"""Inference for the eight released surface-code and BB models."""

from pathlib import Path
import hashlib
import importlib.metadata
import json

import numpy as np
import torch
from torch.nn import functional as F
from safetensors.torch import load_file

MODELS = Path(__file__).resolve().parent / "models"


def list_models():
    """Return the names of the supplied models."""
    return sorted(folder.name for folder in MODELS.iterdir() if (folder / "config.json").is_file())


def model_path(model):
    """Return the folder of a supplied model, given its name or a folder path.

    Names are not case-sensitive, so ``"scd7p001"`` finds ``"SCd7p001"``.
    """
    path = Path(model)
    if (path / "config.json").is_file():
        return path
    names = {name.lower(): name for name in list_models()}
    if str(model).lower() in names:
        return MODELS / names[str(model).lower()]
    raise ValueError(f"Unknown model {model!r}; choose from {list_models()} or pass a model folder")


class Postselector:
    """Load one fixed circuit and return one acceptance score per syndrome.

    ``model`` is a supplied model name, such as ``"SCd5p003"``, or
    a model folder. Larger scores indicate greater confidence in decoder
    success. Each instance owns a BP decoder for BB models; do not share it
    between concurrent workers.
    """

    def __init__(self, model, device="cpu"):
        self.directory = model_path(model)
        self.config = json.loads((self.directory / "config.json").read_text())
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("Use cpu or cuda")
        for name, expected in self.config["sha256"].items():
            if hashlib.sha256((self.directory / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"File checksum mismatch: {name}")
        self.weights = load_file(str(self.directory / self.config.get("weights_file", "model.safetensors")), device=str(self.device))
        with np.load(self.directory / "graph.npz", allow_pickle=False) as data:
            self.graph = {name: data[name] for name in data.files}
        self.static = torch.as_tensor(self.graph["static_features"], device=self.device)
        self.edges = torch.as_tensor(self.graph["edge_index"], device=self.device)
        self.detectors = self.config["detectors"]
        self.bb = self.config["family"] == "bb"
        if self.bb:
            from ldpc import BpDecoder
            from scipy.sparse import csc_matrix

            if importlib.metadata.version("ldpc").split(".")[0] != "2":
                raise ValueError("BB scoring requires ldpc version 2")
            g = self.graph
            self.check_matrix = csc_matrix(
                (np.ones(len(g["h_indices"]), dtype=np.uint8), g["h_indices"], g["h_indptr"]),
                shape=(self.detectors, len(g["priors"])),
            )
            self.bp = BpDecoder(self.check_matrix, error_channel=g["priors"].tolist(),
                                max_iter=30, bp_method="minimum_sum", ms_scaling_factor=0.625,
                                schedule="parallel", omp_thread_count=1,
                                input_vector_type="syndrome", random_schedule_seed=0)
            self.support = torch.as_tensor(g["support_weights"], device=self.device)
        else:
            self.edge_features = torch.as_tensor(self.graph["edge_features"], device=self.device)

    def _linear(self, x, name):
        return F.linear(x, self.weights[name + ".weight"], self.weights[name + ".bias"])

    def _mlp(self, x, name, last=2):
        return self._linear(F.relu(self._linear(x, name + ".0")), name + f".{last}")

    def _batch_norm(self, x, name):
        w = self.weights
        y = F.batch_norm(x.reshape(-1, x.shape[-1]), w[name + ".running_mean"],
                         w[name + ".running_var"], w[name + ".weight"],
                         w[name + ".bias"], training=False, eps=1e-5)
        return y.reshape(x.shape)

    @staticmethod
    def _sum_nodes(x):
        # Preserve the original scatter-sum order, including for nearly
        # uniform embeddings where a different reduction can shift scores.
        batch, nodes, hidden = x.shape
        ids = torch.arange(batch, device=x.device).repeat_interleave(nodes)
        return x.new_zeros(batch, hidden).index_add_(0, ids, x.reshape(-1, hidden))

    def _bp_features(self, syndromes, *, return_llrs=False):
        g = self.graph
        count, columns = len(syndromes), len(g["priors"])
        llrs = np.empty((count, columns), dtype=np.float32)
        bits = np.empty((count, columns), dtype=np.uint8)
        full_llrs = np.empty((count, columns), dtype=np.float64) if return_llrs else None
        for i, syndrome in enumerate(syndromes):
            if syndrome.any():
                bits[i] = self.bp.decode(np.ascontiguousarray(syndrome, dtype=np.uint8))
                llrs[i] = self.bp.log_prob_ratios
                if return_llrs:
                    full_llrs[i] = self.bp.log_prob_ratios
            else:
                # ldpc's empty-syndrome shortcut leaves its posterior buffer stale.
                # Use the fixed circuit's verified one-iteration BP state instead.
                llrs[i], bits[i] = g["zero_bp_llrs"], g["zero_bp_bits"]
                if return_llrs:
                    full_llrs[i] = g["zero_bp_llrs"]  # Empty shots need no LSD.
        if not np.isfinite(llrs).all():
            raise ValueError("BP produced non-finite posterior LLRs")
        # Match the posterior storage precision used for the paper's evaluation.
        llrs = np.clip(llrs, -60000, 60000).astype(np.float16).astype(np.float32)
        mapping = g["factor_to_column"]
        prior = g["prior_llrs"].astype(np.float32)
        features = np.stack((np.clip(llrs[:, mapping], -60, 60) / 20,
                             bits[:, mapping].astype(np.float32),
                             np.clip((llrs - prior)[:, mapping], -60, 60) / 20), axis=-1)
        return (features, full_llrs) if return_llrs else features

    def _forward(self, x):
        if self.bb:
            h = torch.cat((F.relu(self._linear(x[:, :self.detectors], "det_proj.0")),
                           F.relu(self._linear(x[:, self.detectors:], "factor_proj.0"))), dim=1)
        else:
            h = F.relu(self._linear(x, "input_proj.0"))
        src, dst = self.edges
        for layer in range(self.config["layers"]):
            prefix = f"mpnn_blocks.{layer}"
            parts = [h[:, dst], h[:, src]]
            if not self.bb:
                parts.append(self.edge_features.unsqueeze(0).expand(len(x), -1, -1))
            messages = self._mlp(torch.cat(parts, dim=-1), prefix + ".mp.phi")
            aggregate = torch.zeros_like(h).index_add_(1, dst, messages)
            update = self._mlp(torch.cat((h, aggregate), dim=-1), prefix + ".mp.psi")
            h = h + F.relu(self._batch_norm(update, prefix + ".mp.bn"))
            context = (self._sum_nodes(h) / h.shape[1]).unsqueeze(1).expand_as(h)
            update = self._mlp(torch.cat((h, context), dim=-1), prefix + ".global_update.mlp", last=3)
            h = h + self._batch_norm(update, prefix + ".global_update.bn")
        if self.bb:
            pooled = torch.einsum("bfh,fk->bkh", h[:, self.detectors:], self.support)
            return self._mlp(pooled, "head", last=3).squeeze(-1)
        gates = self._mlp(h, "pool.gate_nn")
        attention = (gates - gates.amax(dim=1, keepdim=True)).exp()
        attention = attention / (self._sum_nodes(attention).unsqueeze(1) + 1e-16)
        return self._mlp(self._sum_nodes(attention * h), "head", last=3)

    @torch.inference_mode()
    def score(self, syndromes, batch_size=8):
        """Score an unpacked binary array shaped (shots, detectors)."""
        sy = np.asarray(syndromes)
        if sy.ndim != 2 or sy.shape[1] != self.detectors:
            raise ValueError(f"Expected shape (shots, {self.detectors})")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        scores = np.empty(len(sy), dtype=np.float32)
        for start in range(0, len(sy), batch_size):
            batch = sy[start:start + batch_size]
            if not np.all((batch == 0) | (batch == 1)):
                raise ValueError("Syndromes must contain only zero and one")
            scores[start:start + len(batch)] = self._score_batch(batch)
        return scores

    @torch.inference_mode()
    def _score_batch(self, batch, bp_features=None):
        x = self.static.unsqueeze(0).repeat(len(batch), 1, 1)
        x[:, :self.detectors, 0] = torch.as_tensor(np.array(batch, dtype=np.float32), device=self.device)
        if self.bb:
            if bp_features is None:
                bp_features = self._bp_features(batch)
            x[:, self.detectors:, 3:6] = torch.as_tensor(bp_features, device=self.device)
        logits = self._forward(x)
        result = F.logsigmoid(logits).sum(dim=-1) if self.bb else logits[:, 0]
        if not torch.isfinite(result).all():
            raise ValueError("Model produced non-finite scores")
        return result.cpu().numpy()

"""Decode shots and estimate a post-selected logical error curve."""

import numpy as np
import stim

from .inference import Postselector


class Evaluator:
    """Use the paper's decoder and share BB posteriors with neural scoring."""

    def __init__(self, selector):
        self.selector = selector
        self.circuit = stim.Circuit.from_file(selector.directory / "circuit.stim")
        self.num_observables = self.circuit.num_observables
        if selector.bb:
            from ldpc.lsd_decoder import LsdDecoder

            self.lsd = LsdDecoder(selector.check_matrix, bits_per_step=1,
                                  lsd_order=0, lsd_method="LSD_0")
            # Support pooling records the logical action of every fault.
            # Map it to the merged BP columns used by the fixed circuit.
            g = selector.graph
            mapping = g["factor_to_column"]
            support = (g["support_weights"] > 0).astype(np.uint8)
            columns = len(g["priors"])
            if not np.array_equal(np.unique(mapping), np.arange(columns)):
                raise ValueError("Incomplete mapping to decoder columns")
            self.logical_action = np.zeros((self.num_observables, columns), dtype=np.uint8)
            self.logical_action[:, mapping] = support.T
            if not np.array_equal(self.logical_action[:, mapping].T, support):
                raise ValueError("Conflicting logical actions for a decoder column")
        else:
            import pymatching

            dem = self.circuit.detector_error_model(decompose_errors=True)
            self.matching = pymatching.Matching.from_detector_error_model(dem)

    def score_and_decode(self, syndromes, batch_size=8):
        """Return acceptance scores and predicted logical observables.

        Decode every shot once so the same labels can be reused across the
        rejection sweep, including the unselected baseline. This is offline
        evaluation; deployed post-selection only decodes retained shots.
        """
        sy = np.asarray(syndromes)
        selector = self.selector
        if sy.ndim != 2 or sy.shape[1] != selector.detectors:
            raise ValueError(f"Expected shape (shots, {selector.detectors})")
        if not np.all((sy == 0) | (sy == 1)):
            raise ValueError("Syndromes must contain only zero and one")
        if not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        sy = np.ascontiguousarray(sy, dtype=np.uint8)
        if not selector.bb:
            return selector.score(sy, batch_size), self.matching.decode_batch(sy)

        scores = np.empty(len(sy), dtype=np.float32)
        predictions = np.zeros((len(sy), self.num_observables), dtype=np.uint8)
        for start in range(0, len(sy), batch_size):
            batch = sy[start:start + batch_size]
            features, llrs = selector._bp_features(batch, return_llrs=True)
            scores[start:start + len(batch)] = selector._score_batch(batch, features)
            for offset, syndrome in enumerate(batch):
                if syndrome.any():
                    # Run LSD even where BP converges, as with ldpc's
                    # always_run_lsd=True. Keep native FP64 LLRs.
                    correction = self.lsd.decode(syndrome, llrs[offset])
                    predictions[start + offset] = (self.logical_action @ correction) % 2
        return scores, predictions


def simulate(model, shots, *, seed=None, device="cpu", batch_size=32, chunk_size=10_000,
             progress=True):
    """Sample shots from a model's circuit, then score and decode every shot.

    ``model`` is a model name, a model folder or a Postselector. Returns a
    dict with the model name, its number of rounds, one acceptance score per
    shot and one failure flag per shot. A shot fails if the decoder predicts
    any logical observable wrongly. Shots are processed in chunks of
    ``chunk_size``, so memory does not grow with the full experiment.
    """
    if isinstance(shots, bool) or not isinstance(shots, (int, np.integer)) or shots < 1:
        raise ValueError("shots must be a positive integer")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, (int, np.integer)) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    selector = model if isinstance(model, Postselector) else Postselector(model, device=device)
    evaluator = Evaluator(selector)
    sampler = evaluator.circuit.compile_detector_sampler(seed=seed)
    scores = np.empty(shots, dtype=np.float32)
    failures = np.empty(shots, dtype=bool)
    for start in range(0, shots, chunk_size):
        stop = min(start + chunk_size, shots)
        syndromes, actual = sampler.sample(stop - start, separate_observables=True)
        scores[start:stop], predicted = evaluator.score_and_decode(syndromes, batch_size=batch_size)
        failures[start:stop] = np.any(predicted != actual, axis=1)
        if progress:
            print(f"{stop:,}/{shots:,} shots; {failures[:stop].sum():,} logical failures", flush=True)
    return dict(model=selector.config["model"], rounds=selector.config["rounds"],
                scores=scores, failures=failures)


def rejection_curve(scores, failures, rejection_rates):
    """Rank scores and return failure counts, shot LER and Wilson intervals.

    Intervals use z=1.96 and are pointwise, not a simultaneous confidence band.
    Ties retain input order. Report actual rejection after rounding shot counts.
    ``raw_ler_shot`` is the LER of all shots, without post-selection.
    """
    scores = np.asarray(scores)
    failures = np.asarray(failures)
    rates = np.asarray(rejection_rates, dtype=float)
    if scores.ndim != 1 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("scores must be a nonempty finite vector")
    if failures.shape != scores.shape or not np.all((failures == 0) | (failures == 1)):
        raise ValueError("failures must contain one binary label per score")
    if rates.ndim != 1 or not np.isfinite(rates).all() or np.any((rates < 0) | (rates >= 1)):
        raise ValueError("rejection rates must lie in [0, 1)")
    retained = np.rint((1 - rates) * len(scores)).astype(np.int64)
    if np.any(retained == 0):
        raise ValueError("Each operating point must retain at least one shot")
    order = np.argsort(-scores, kind="stable")
    counts = np.r_[0, np.cumsum(failures[order], dtype=np.int64)][retained]
    ler = counts / retained
    z = 1.96
    denominator = 1 + z * z / retained
    centre = (ler + z * z / (2 * retained)) / denominator
    half_width = z * np.sqrt(ler * (1 - ler) / retained + z * z / (4 * retained**2)) / denominator
    return dict(rejection=1 - retained / len(scores), retained=retained,
                failures=counts, ler_shot=ler,
                low_shot=np.where(counts == 0, 0, np.maximum(0, centre - half_width)),
                high_shot=np.where(counts == retained, 1, np.minimum(1, centre + half_width)),
                raw_ler_shot=failures.mean())


def per_cycle(probability, rounds):
    """Paper reporting convention: 1 - (1 - P_shot)**(1 / rounds)."""
    p = np.asarray(probability, dtype=float)
    if not np.isfinite(rounds) or rounds <= 0:
        raise ValueError("rounds must be positive")
    if not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must lie in [0, 1]")
    with np.errstate(divide="ignore"):
        return -np.expm1(np.log1p(-p) / rounds)

"""Command-line tools: mpnn4qec-score and mpnn4qec-curve."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .evaluation import per_cycle, rejection_curve, simulate
from .inference import Postselector, list_models


def _common(parser):
    parser.add_argument("--model", required=True,
                        help=f"Model name ({', '.join(list_models())}) or model folder")
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:0")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)


def _setup(parser, args):
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def score():
    """Score saved syndromes, or sample example shots from the model's circuit."""
    parser = argparse.ArgumentParser(prog="mpnn4qec-score", description=score.__doc__)
    _common(parser)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Unpacked binary syndromes in a .npy file")
    source.add_argument("--shots", type=int, help="Instead sample this many example shots")
    parser.add_argument("--output", type=Path, default=Path("scores.npy"))
    args = parser.parse_args()
    if args.shots is not None and args.shots < 1:
        parser.error("--shots must be positive")
    if args.output.suffix != ".npy":
        parser.error("--output must end in .npy")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    _setup(parser, args)
    selector = Postselector(args.model, device=args.device)
    if args.input:
        syndromes = np.load(args.input, mmap_mode="r", allow_pickle=False)
    else:
        import stim
        circuit = stim.Circuit.from_file(selector.directory / "circuit.stim")
        syndromes = circuit.compile_detector_sampler(seed=args.seed).sample(args.shots)
    scores = selector.score(syndromes, batch_size=args.batch_size)
    np.save(args.output, scores)
    print(f"Saved {len(scores)} scores to {args.output}; higher scores favour acceptance.")


def curve():
    """Simulate, score and decode shots, then save a post-selected LER curve."""
    parser = argparse.ArgumentParser(prog="mpnn4qec-curve", description=curve.__doc__)
    _common(parser)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--max-rejection", type=float, default=0.5,
                        help="Largest rejection rate, as a fraction (default 0.5)")
    parser.add_argument("--points", type=int, default=51, help="Rejection rates on the curve")
    parser.add_argument("--chunk-size", type=int, default=10_000)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Folder for the figure and data (default: results/MODEL)")
    args = parser.parse_args()
    if args.shots < 1:
        parser.error("--shots must be positive")
    if not 0 < args.max_rejection < 1:
        parser.error("--max-rejection must lie strictly between 0 and 1")
    if args.points < 2:
        parser.error("--points must be at least 2")
    if args.chunk_size < 1:
        parser.error("--chunk-size must be positive")
    _setup(parser, args)
    output = args.output_dir or Path("results") / Path(args.model).name
    names = ["ler_curve.csv", "ler_curve.png", "ler_curve.pdf", "scores_and_failures.npz"]
    existing = [name for name in names if (output / name).exists()]
    if existing:
        parser.error(f"{output} already contains {', '.join(existing)}")

    result = simulate(args.model, args.shots, seed=args.seed, device=args.device,
                      batch_size=args.batch_size, chunk_size=args.chunk_size)
    data = rejection_curve(result["scores"], result["failures"],
                           np.linspace(0, args.max_rejection, args.points))
    rounds = result["rounds"]
    cycle = {name: per_cycle(data[name + "_shot"], rounds) for name in ("ler", "low", "high")}

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .plotting import plot_ler_curve

    output.mkdir(parents=True, exist_ok=True)
    ax = plot_ler_curve(data, rounds, label=result["model"])
    ax.set(title=f"{result['model']}, {args.shots:,} shots", xlim=(0, 100 * args.max_rejection))
    ax.figure.tight_layout()
    ax.figure.savefig(output / "ler_curve.png", dpi=160, bbox_inches="tight")
    ax.figure.savefig(output / "ler_curve.pdf", bbox_inches="tight")
    plt.close(ax.figure)
    np.savez_compressed(output / "scores_and_failures.npz", scores=result["scores"],
                        failures=result["failures"], model=result["model"], seed=args.seed,
                        rounds=rounds)
    columns = ["rejection", "retained", "failures", "ler_shot", "low_shot", "high_shot"]
    table = np.column_stack([data[name] for name in columns] + [cycle["ler"], cycle["low"], cycle["high"]])
    np.savetxt(output / "ler_curve.csv", table, delimiter=",", comments="", fmt="%.10g",
               header=",".join(columns + ["ler_cycle", "low_cycle", "high_cycle"]))
    print(f"Raw LER per cycle {per_cycle(data['raw_ler_shot'], rounds):.4g}; saved the curve to {output}")

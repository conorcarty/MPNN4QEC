"""Plot post-selected logical error rate against rejection rate."""

import numpy as np

from .evaluation import per_cycle


def plot_ler_curve(curve, rounds=None, *, ax=None, label="MPNN", color=None, show_raw=True,
                   raw_label="Raw LER"):
    """Plot a rejection curve with its pointwise 95% Wilson interval.

    ``curve`` comes from ``rejection_curve``. With ``rounds``, rates use the
    paper's per-cycle convention; without it, they are per shot. Points with no
    observed failures are left off the line, but their interval is still
    shaded. ``label`` names the post-selected curve in the legend only. With
    ``show_raw``, a dashed line labelled ``raw_label`` marks the decoder's LER
    without post-selection. Returns the matplotlib axes, so several curves can
    share one plot.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 4.2))

    def rate(values):
        return per_cycle(values, rounds) if rounds is not None else np.asarray(values, dtype=float)

    rejection = 100 * np.asarray(curve["rejection"])
    observed = np.asarray(curve["failures"]) > 0
    (line,) = ax.plot(rejection, np.where(observed, rate(curve["ler_shot"]), np.nan),
                      "o-", color=color, linewidth=1, markersize=3, label=label)
    ax.fill_between(rejection, rate(curve["low_shot"]), rate(curve["high_shot"]),
                    color=line.get_color(), alpha=0.18, linewidth=0)
    if show_raw and curve["raw_ler_shot"] > 0:
        ax.axhline(rate(curve["raw_ler_shot"]), color=line.get_color(), linestyle="--",
                   linewidth=1, label=raw_label)
    ax.set(xlabel="Rejection rate (%)", yscale="log",
           ylabel="Post-selected LER per cycle" if rounds is not None else "Post-selected LER per shot")
    ax.legend(frameon=False, fontsize=9)
    return ax

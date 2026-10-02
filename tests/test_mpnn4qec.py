"""Quick checks of the installed package. Run with ``python -m pytest tests``."""

import matplotlib
import numpy as np
import pytest

import mpnn4qec

matplotlib.use("Agg")


@pytest.mark.parametrize("name", mpnn4qec.list_models())
def test_every_model_loads_and_scores(name):
    selector = mpnn4qec.Postselector(name)
    syndromes = mpnn4qec.Evaluator(selector).circuit.compile_detector_sampler(seed=1).sample(4)
    scores = selector.score(syndromes)
    assert scores.shape == (4,) and np.isfinite(scores).all()


def test_unknown_model_lists_choices():
    with pytest.raises(ValueError, match="SCd5p003"):
        mpnn4qec.Postselector("SCd11p001")


def test_model_names_ignore_case():
    assert mpnn4qec.model_path("scd7p001") == mpnn4qec.model_path("SCd7p001")
    assert mpnn4qec.model_path("bb144P003").name == "BB144p003"


@pytest.mark.parametrize("name, shots", [("SCd5p003", 2000), ("BB72p001", 100)])
def test_simulate_and_plot(name, shots):
    result = mpnn4qec.simulate(name, shots, seed=1, chunk_size=shots // 2 + 1, progress=False)
    assert result["scores"].shape == result["failures"].shape == (shots,)
    curve = mpnn4qec.rejection_curve(result["scores"], result["failures"], np.linspace(0, 0.5, 11))
    assert curve["failures"][0] == result["failures"].sum()
    assert np.all(np.diff(curve["failures"]) <= 0)
    ax = mpnn4qec.plot_ler_curve(curve, result["rounds"], label=result["model"])
    assert ax.get_yscale() == "log"
    legend = [text.get_text() for text in ax.get_legend().get_texts()]
    assert legend[0] == name and legend[1:] in ([], ["Raw LER"])


def test_rejection_curve_drops_lowest_scores():
    scores = np.arange(100.0)
    failures = scores < 5
    curve = mpnn4qec.rejection_curve(scores, failures, [0, 0.05, 0.1])
    assert curve["retained"].tolist() == [100, 95, 90]
    assert curve["failures"].tolist() == [5, 0, 0]
    assert curve["low_shot"][0] <= curve["ler_shot"][0] <= curve["high_shot"][0]
    assert curve["raw_ler_shot"] == 0.05


def test_per_cycle_convention():
    assert mpnn4qec.per_cycle(0.2, 1) == pytest.approx(0.2)
    assert mpnn4qec.per_cycle(1 - 0.9**5, 5) == pytest.approx(0.1)

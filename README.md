# MPNN4QEC

Pretrained graph neural post-selectors from *Graph Neural Post-selection for Quantum Error Correction*. Install the package to simulate shots with the paper's models and plot post-selected logical error rate (LER) against rejection rate.

The models are trained for the specific circuits and noise described below. We will add the training code to this repository soon, so that you can train models for your own settings.

## Install

```bash
pip install git+https://github.com/conorcarty/MPNN4QEC
```

The package needs Python 3.10 or newer; we have tested Python 3.12 and 3.14. A CPU is sufficient.

## Quick start

Simulate 100,000 shots of the distance-5 surface code at p = 0.003, then plot the post-selected LER per cycle:

```python
import numpy as np
import matplotlib.pyplot as plt
import mpnn4qec

result = mpnn4qec.simulate("SCd5p003", shots=100_000)
curve = mpnn4qec.rejection_curve(result["scores"], result["failures"], np.linspace(0, 0.2, 41))
mpnn4qec.plot_ler_curve(curve, result["rounds"], label=result["model"])
plt.show()
```

The model name passed to `simulate` sets the code, circuit, noise and weights; names are not case-sensitive. `label` only names the curve in the legend. The dashed line is the raw LER: the decoder's own LER without post-selection. `plot_ler_curve` returns the matplotlib axes. Pass `ax=` to draw several models on one plot, and `raw_label=` to rename each dashed line.

The same from the command line, saving a PNG, a PDF, a CSV of the curve and the raw scores and failure labels to `results/SCd5p003`:

```bash
mpnn4qec-curve --model SCd5p003 --shots 100000
```

The [tutorial notebook](tutorial.ipynb) explains each step.

## Models

| Name | Code | Rounds | Training noise, p | Decoder | Weights |
|---|---|---|---|---|---|
| `SCd5p001` | Rotated surface, d = 5 | 5 | 0.001 | MWPM | [Download](https://github.com/conorcarty/MPNN4QEC/raw/main/src/mpnn4qec/models/SCd5p001/SCd5p001.safetensors) |
| `SCd5p003` | Rotated surface, d = 5 | 5 | 0.003 | MWPM | [Download](https://github.com/conorcarty/MPNN4QEC/raw/main/src/mpnn4qec/models/SCd5p003/SCd5p003.safetensors) |
| `SCd7p001` | Rotated surface, d = 7 | 7 | 0.001 | MWPM | [Download](https://github.com/conorcarty/MPNN4QEC/raw/main/src/mpnn4qec/models/SCd7p001/SCd7p001.safetensors) |
| `SCd7p003` | Rotated surface, d = 7 | 7 | 0.003 | MWPM | [Download](https://github.com/conorcarty/MPNN4QEC/raw/main/src/mpnn4qec/models/SCd7p003/SCd7p003.safetensors) |
| `SCd9p001` | Rotated surface, d = 9 | 9 | 0.001 | MWPM | [Download](https://github.com/conorcarty/MPNN4QEC/raw/main/src/mpnn4qec/models/SCd9p001/SCd9p001.safetensors) |
| `SCd9p003` | Rotated surface, d = 9 | 9 | 0.003 | MWPM | [Download](https://github.com/conorcarty/MPNN4QEC/raw/main/src/mpnn4qec/models/SCd9p003/SCd9p003.safetensors) |
| `BB72p001` | BB [[72,12,6]] | 6 | 0.001 | BP+LSD | [Download](https://github.com/conorcarty/MPNN4QEC/raw/main/src/mpnn4qec/models/BB72p001/BB72p001.safetensors) |
| `BB144p003` | BB [[144,12,12]] | 12 | 0.003 | BP+LSD | [Download](https://github.com/conorcarty/MPNN4QEC/raw/main/src/mpnn4qec/models/BB144p003/BB144p003.safetensors) |

The installed package includes every model, and `mpnn4qec.list_models()` lists their names. Each model folder holds its weights, configuration, circuit and fixed graph data.

## How the models are configured

Each model was trained on one circuit, one noise setting and one decoder:

- **Surface codes.** Stim's `surface_code:rotated_memory_z` circuit at distance d with d rounds. Circuit-level depolarising noise applies the same p to all four of Stim's channels: after Clifford gates, on data qubits before each round, before measurements and after resets. Decoding uses PyMatching MWPM.
- **BB codes.** The Z-basis memory circuit of [Gong et al.](https://arxiv.org/abs/2403.18901), with 6 rounds for [[72,12,6]] and 12 for [[144,12,12]], and the same p on every noise channel. Decoding uses parallel min-sum BP (scaling 0.625, at most 30 iterations) followed by order-0 LSD on every nonempty shot, including shots where BP has already converged (`always_run_lsd=True` in `ldpc`).

The circuit for each model is in its folder as `circuit.stim`. The models have not been fine-tuned for other regimes, such as other noise strengths or decoders, so expect them to perform less well there.

## Plotting post-selected LER curves

`simulate` samples shots from a model's circuit, scores each shot, and decodes it once with the model's decoder. A shot fails if any logical observable is predicted wrongly. `rejection_curve` then keeps the highest-scoring shots at each rejection rate and gives the post-selected LER. The paper reports LER per cycle, `1 - (1 - LER_shot)**(1 / rounds)`; `per_cycle` applies this convention, as does `plot_ler_curve` when given the number of rounds.

Small post-selected LERs need many shots. With default settings on one Apple M4 Pro CPU, simulation, scoring and decoding run at roughly:

| Model | Shots per second |
|---|---|
| Surface, d = 5 | 3,700 |
| Surface, d = 7 | 1,200 |
| Surface, d = 9 | 440 |
| BB [[72,12,6]] | 370 |
| BB [[144,12,12]] | 30 |

For comparison, the paper's BB [[72,12,6]] curve uses about 10^8 shots.

## Credits

The surface circuits are generated by [Stim](https://github.com/quantumlib/Stim). The BB circuits use the circuit builder of [Gong et al.](https://arxiv.org/abs/2403.18901) ([SlidingWindowDecoder](https://github.com/gongaa/SlidingWindowDecoder)), as distributed in [Lee et al.](https://doi.org/10.1038/s41534-026-01242-x)'s [ldpc-post-selection](https://github.com/seokhyung-lee/ldpc-post-selection); the builder's licence is in `THIRD_PARTY_LICENSE`. Decoding uses [PyMatching](https://github.com/oscarhiggott/PyMatching) and [ldpc](https://github.com/quantumgizmos/ldpc). This package is released under the MIT licence.

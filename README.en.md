# LOPT detector reference implementation

[中文](README.md)

This repository provides the CPU-side Permutation LOPT detector and an optional PyTorch logM2 feature function. Callers supply clean history and consecutive layer-wise observations. The detector returns scores, path traces, and alarm decisions.

This is a code-only release of an archived detection core. The core source is unchanged. Example data, benchmark tables, and model integration are being prepared separately.

## Contents

| File | Purpose |
| --- | --- |
| `paper_lopt.py` | Calibration, mean gate, three layer orders, path scoring, Cauchy aggregation, and joint decisions |
| `features.py` | Optional tensor-to-logM2 reference function |
| `requirements.txt` | CPU detector dependencies |
| `VERSION.md` | Source identity, release scope, and verification record |
| `SHA256SUMS.txt` | File checksums |

Experiment data, weights, fault-injection code and configuration, server access details, and credentials are excluded. Direct, Block, and CUSUM baselines and device-side fused kernels are outside this release.

## Installation

Python 3.11 is recommended. Create a virtual environment and activate it:

```sh
python -m venv .venv
```

On Windows, run `.venv\Scripts\Activate.ps1`. On Linux/macOS, run `source .venv/bin/activate`. Then install the CPU dependencies:

```sh
python -m pip install -r requirements.txt
```

The detector uses NumPy and SciPy. The optional feature function requires a PyTorch installation appropriate for the caller's training environment. PyTorch is unnecessary when processing existing scalar observations.

## Inputs and usage

Each stream uses its own calibration state. All arrays use the same layer order:

- `history`: clean observations collected before evaluation, shape `(history_steps, layer_count)`.
- `previous`: features at the preceding observation time, shape `(layer_count,)`.
- `current`: features at the current observation time, shape `(layer_count,)`.

The variables below are supplied by the caller; no observation data are bundled.

```python
from paper_lopt import Config, calibrate, test_stream, joint_decision

config = Config(layer_count=36, history_steps=50, stream_count=3)
calibration = calibrate(history, config)
result = test_stream(current, previous, calibration, config)
print(result["status"], result["p_value"], result["alarm"])
```

For multiple streams, call `test_stream` with each stream's calibration and pass a dictionary of complete results to `joint_decision(stream_results)`. Set `stream_count` to the frozen protocol's stream count. Keep calibration and configuration fixed during evaluation. The core does not maintain or update a temporal buffer.

## Detection rule

1. Estimate per-layer means and standard deviations from clean history; calibrate `sigma`, `d0`, and the mean-gate boundary.
2. Normalize both observations with the same calibration and compute **same-layer differences between consecutive times**.
3. Evaluate paths when the normalized mean increase exceeds the mean-gate boundary.
4. Use three deterministic layer orders: forward, reverse, and the second half in forward order followed by the first half in reverse order. Each path chooses the next term's direction from the sign of its accumulated score.
5. Combine the three path tail probabilities with equal-weight Cauchy aggregation. A stream alarms when the combined value is strictly below `alpha_total / stream_count`.
6. The joint decision is `True` if any stream alarms.

The archived defaults are 36 layers, 50 historical times, 8 circular shifts, quantiles of 0.90, `lambda_weight=beta_weight=1`, a total threshold of 0.003, and 3 streams. Freeze the intended protocol before evaluation.

| Status | Meaning | `alarm` |
| --- | --- | --- |
| `GATE_REJECT` | Mean gate failed; paths were not evaluated | `False` |
| `PATH_EVALUATED` | Paths evaluated and combined | `True` or `False` |
| `NONFINITE` | An input contains NaN or Inf | `None` |

With no positive alarm and at least one `None`, `joint_decision` returns `None`. The caller records nonfinite outcomes separately from statistical alarms.

## Optional feature function

```python
from features import global_logm2

value = global_logm2(observed_tensor)
```

The function converts the detached tensor to FP32 and computes:

$$
z = \log_{10}\!\left(\operatorname{mean}(x^2) + 10^{-30}\right).
$$

It preserves the archived operation order and does not write to the training tensor. `.item()` returns a host scalar and may synchronize device execution. This function specifies the feature and provides a reference adapter; fused statistics, asynchronous transfer, and low-overhead model integration are separate engineering components. Use the same feature, logarithm base, and precision convention for calibration and observations.

## Verification and release status

The detector source matches the archived SHA256. CPU import and basic API checks pass; see [VERSION.md](VERSION.md) for their scope and environment. Example data and benchmark tables will be added after selection. A line-by-line match to the current manuscript and the selected experimental protocol is pending.

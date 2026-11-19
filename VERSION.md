# 版本与验证 / Version and verification

## 本版范围 / Release scope

- 日期 / Date: 2026-11-19.
- 类型 / Type: code-only detector reference.
- `paper_lopt.py` is copied byte-for-byte from the archived detector-only candidate.
- Detector SHA256: `83b636bc1fbee4c74d2463c1994eb86d24fee2872989c02b2c4915659f030a26`.
- `features.py` extracts only the existing `global_logm2` function, preserving its signature and body.
- Existing experimental archives and frozen releases remain unchanged.

## 本地检查 / Local checks

CPU environment: Python 3.11.1, NumPy 2.4.2, SciPy 1.17.1. `requirements.txt` lists dependencies without pinning; these versions identify the environment used for the checks below.

The release checks cover source identity, syntax, imports, calibration dimensions, identical-observation gate rejection, finite path evaluation, nonfinite status, and joint decision behavior. API checks use transient, artificial inputs and include no archived experiment records.

The optional PyTorch feature function is checked for syntax and source equivalence. PyTorch is absent from the packaging environment, so no tensor-runtime check is included in this release.

## 后续发布项 / Pending release items

- Select example data and a matching evaluation protocol.
- Verify the detector version against the current manuscript, including calibration, mean gating, path scoring, and tail probabilities.
- Qualify model/device integration separately if that implementation is to be released.

本版未附带基准结果或性能结论。仓库尚未选择开源许可证。

No benchmark results or performance claims are bundled. An open-source license has not been selected.

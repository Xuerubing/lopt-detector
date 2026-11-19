# LOPT 检测参考实现

[English](README.en.md)

本仓库提供 CPU 端 Permutation LOPT 检测核心，以及可选的 PyTorch logM2 特征提取函数。使用者提供按层排列的正常历史和相邻时刻观测，得到评分、路径明细及报警判定。

当前版本为归档检测核心的代码整理版。检测源文件保持原样；配套实验数据、结果表及模型集成代码另行整理。

## 文件

| 文件 | 内容 |
| --- | --- |
| `paper_lopt.py` | 历史校准、均值门、三种层序评分、Cauchy 合并和多流联合判定 |
| `features.py` | 可选的 PyTorch 张量到 logM2 标量提取 |
| `requirements.txt` | CPU 检测依赖 |
| `VERSION.md` | 代码来源、版本范围和验证记录 |
| `SHA256SUMS.txt` | 本版文件校验清单 |

本仓库不包含实验数据、模型权重、故障注入代码或配置、服务器入口及凭据。Direct、Block、CUSUM 对比方法和设备端融合算子也未纳入本版。

## 环境

建议使用 Python 3.11，在独立环境中安装：

```sh
python -m venv .venv
```

Windows：

```powershell
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Linux/macOS：

```sh
source .venv/bin/activate
python -m pip install -r requirements.txt
```

CPU 检测仅依赖 NumPy 和 SciPy。`features.py` 另外需要与使用者训练环境匹配的 PyTorch；检测已有标量记录时无需安装 PyTorch。

## 输入与调用

每个检测流有独立的校准状态，层顺序在历史、前一时刻和当前时刻之间保持一致。

| 输入 | 形状 | 含义 |
| --- | --- | --- |
| `history` | `(history_steps, layer_count)` | 评价开始前取得的正常历史 |
| `previous` | `(layer_count,)` | 同一流前一时刻的各层特征 |
| `current` | `(layer_count,)` | 同一流当前时刻的各层特征 |

下面的变量由使用者提供，仓库不附带观测数据：

```python
from paper_lopt import Config, calibrate, test_stream, joint_decision

config = Config(layer_count=36, history_steps=50, stream_count=3)
calibration = calibrate(history, config)
result = test_stream(current, previous, calibration, config)

print(result["status"], result["p_value"], result["alarm"])
```

多流使用各自的校准状态执行 `test_stream`，然后传入结果字典：

```python
joint_alarm = joint_decision(stream_results)
```

`stream_results` 的键是流名，值为对应的完整 `test_stream` 返回值。`stream_count` 应与冻结方案的流数量一致。评价期间固定配置及正常历史；该核心不会自动更新历史或管理时序缓存。

## 检测步骤

1. 按层从正常历史估计均值、标准差，并校准 `sigma`、`d0` 和均值门边界。
2. 使用同一校准状态标准化 `current` 和 `previous`，计算**同层相邻时刻差值**。
3. 当前与前一时刻的标准化均值差超过均值门边界时，进入路径评分。
4. 依次使用正序、逆序，以及“后半段正序接前半段逆序”三种确定性层序。每条路径由已有累计评分的符号决定下一项的方向。
5. 将三条路径的尾概率做等权 Cauchy 合并；合并值严格小于 `alpha_total / stream_count` 时，该流报警。
6. 多流中任一流报警，联合判定为 `True`。

默认参数直接保留在 `Config` 中：36 层、50 个历史时刻、8 个循环位移、两个分位数均为 0.90、`lambda_weight=beta_weight=1`、总阈值 0.003、3 个流。实际应用应先冻结对应配置，再进行评价。

返回状态：

| 状态 | 含义 | `alarm` |
| --- | --- | --- |
| `GATE_REJECT` | 均值门未通过，未执行路径评分 | `False` |
| `PATH_EVALUATED` | 已评分并合并路径尾概率 | `True` 或 `False` |
| `NONFINITE` | 输入包含 NaN 或 Inf | `None` |

`joint_decision` 在没有报警但存在 `None` 时返回 `None`。非有限值处理与统计报警分别由调用端记录。

## 可选特征提取

```python
from features import global_logm2

value = global_logm2(observed_tensor)
```

该函数先将观测张量转为 FP32，然后计算：

$$
z = \log_{10}\!\left(\operatorname{mean}(x^2) + 10^{-30}\right).
$$

函数沿用归档代码的运算顺序，使用 `detach()` 读取特征，不写回训练张量。`.item()` 将标量返回到主机，设备张量可能因此同步。本函数用于特征定义和参考接入；融合统计、异步搬运及低开销模型集成属于独立工程实现。

所有历史和观测应使用同一种特征、对数底数及精度约定。

## 本版验证与后续

源文件 SHA256 与归档一致；CPU 导入及基本接口检查通过。详细环境和检查范围见 [VERSION.md](VERSION.md)。实验示例数据及对应基准表待选定后加入。当前文稿与具体实验协议的逐项版本核对留在后续发布阶段完成。

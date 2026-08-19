# DSFlow

**DSFlow: Dual Supervision and Step-Aware Architecture for One-Step Flow Matching Speech Synthesis**

单步流匹配语音合成：用"双监督 + 步感知"训练一个既能多步 ODE 采样、也能**单步**从噪声直接生成梅尔谱的端到端 TTS 模型。

## Idea

流匹配（Flow Matching）语音合成（如 VoiceFlow / E2-TTS / F5-TTS）通常需要在推理时做几十步 ODE 采样。DSFlow 的目标是把推理压到**单步**，同时保持多步采样的能力：

- **Dual Supervision（双监督）**：解码器共享主干、双输出头。
  - 头 A 预测流速度场 `v = x₁ − x₀`，用条件流匹配（CFM）MSE 监督——负责多步 Euler 采样；
  - 头 B 直接预测干净数据 `x₀`，用 L1 监督——负责单步生成（`t=1` 时直接输出目标谱）。
  - 两个头互相促进：速度场给出精确的输运方向，直接头让模型在数据流形上收敛更快。
- **Step-Aware Architecture（步感知架构）**：
  - DiT 风格解码器，时间步 `t` 通过 AdaLN 调制每一层；
  - 每个 Transformer 块带一个**步感知门控**（`sigmoid` 投影的时间步标量），决定该块在当前步贡献多少修正量；
  - 训练时按概率（`step_dropout`，默认 0.1）把随机采样的 `t` 置为 1，让模型直接暴露在推理时的单步分布上。

## Architecture

```text
text ──> TextEncoder (Transformer) ──> DurationPredictor ──> per-token durations (mel frames)
                                             │
                                             ▼
                      repeat-by-duration ──> text features
                                             │
noise x_t ──> linear proj + pos emb ──> ─────┤
                                             ▼
                        StepAwareDecoder (DiT ×N, t-conditioned AdaLN + step gate)
                                             │
                                   ┌─────────┴─────────┐
                                   ▼                   ▼
                             v-head (CFM)        x0-head (direct)
```

推理时有两种路径：

- 单步：`mel = x0_head(x₁, t=1)`，一次前向；
- 多步：Euler 迭代 `x ← x − dt · v_head(x, t)`，任意步数。

## Install

```bash
pip install -e .          # 依赖: torch, torchaudio, soundfile, g2p-en, einops, tqdm, scipy
pip install -e ".[dev]"   # 额外安装 pytest
```

> 音素化默认使用纯 Python 的 `g2p-en`（ARPAbet），不依赖 espeak-ng；若不可用自动回退到字符级。

## Data preparation

首次运行会自动下载 LJSpeech-1.1（约 2.7 GB，数据来自 https://data.keithito.com/data/speech/ ）并解压。

```bash
python -m dsflow.data.prepare --data-root data --cache-dir data/cache --workers 8
```

产出：

- `data/LJSpeech-1.1/`：原始数据集
- `data/cache/mel/*.pt`：每句 80 频段 log10 幅度梅尔谱
- `data/cache/metadata.json`：文本、音素 token、时长等记录
- `data/cache/vocab.json`：音素词表

## Training

```bash
python -m dsflow.train \
  --steps 2000 --batch-size 8 --lr 1e-4 \
  --decoder-layers 8 --decoder-dim 512 \
  --step-dropout 0.1 --lambda-cfm 1.0 --lambda-direct 1.0 \
  --ckpt-dir checkpoints --device cuda
```

- 每 `--ckpt-every` 步保存 `step_N.pt` 与 `last.pt`（含模型、优化器、配置与词表），自动断点续训；
- 双监督权重由 `--lambda-cfm` / `--lambda-direct` 控制，`--step-dropout` 控制单步暴露概率。

## Inference

```bash
# 单步生成
python -m dsflow.infer --ckpt checkpoints/last.pt \
  --text "The examination and testimony of the experts enabled the Commission to conclude that five shots may have been fired" \
  --out outputs/one_step.wav --steps 1 --device cuda

# 多步生成（Euler）
python -m dsflow.infer --ckpt checkpoints/last.pt --text "..." --out outputs/multi.wav --steps 32 --device cuda
```

- 输出 `*.wav`（22.05 kHz）与 `*.mel.pt`（生成梅尔谱）；
- `--duration-scale` 可调节语速（预测时长 × 缩放）；
- 声码器优先使用缓存的 HiFi-GAN（`jaketae/hifigan-lj-v1`，44.1 kHz 原生输出自动重采样到 22.05 kHz，权重缓存于 `data/vocoder/`），下载失败时自动回退到 torchaudio Griffin-Lim。

## Short-run verification (2026-08-18)

在 1 张 NVIDIA L4 上对 LJSpeech 全量 13100 句做 2000 步短训练验证：

| 项目 | 数值 |
| --- | --- |
| 模型 | 51.9M 参数（8 层 512 维步感知解码器） |
| 训练 | 2000 步 / batch 8 / 13:45，loss 299.3 → 77.2（cfm 47.1, direct 30.0, dur 0.003） |
| 推理 | 单步 / 4 步 / 32 步均可合成 ~8.9 s WAV，无削波 |
| 一致性 | 单步 vs 多步梅尔相关性 0.71–0.78 |

说明：2000 步仅为流水线验证。单步 x0 头此时重建相关性（0.39）低于多步（0.60），继续训练（建议 ≥ 50k 步并配合 EMA/更长 step-dropout 曝光）会进一步收敛，这也是本仓库留给后续工作的主线。

## Chatterbox S3Gen 单步化（2026-08-18）

把 [ResembleAI/chatterbox](https://huggingface.co/ResembleAI/chatterbox)（HF 上 AXERA-TECH 板端部署对应的 S3Gen 流匹配声码器）从 **10 步 Euler 蒸馏成 1 步**：

- **目标**：官方 S3Gen 用 `CausalConditionalCFM` 以 10 步 Euler（余弦时间表 + CFG 双 batch 混合）把 S3 语音 token 解码为 mel。
- **方法**（DSFlow 思路移植）：学生共享主干、双输出头（速度头继承教师权重 + 新增零初始化直接头）；每步采新鲜噪声 z，教师 5 步 ODE 端点作 reflow 目标；双监督 = CFM MSE + 直接 L1；step dropout 把 t 置 0（S3Gen 约定 t=0=纯噪声，即单步输入）。
- **验证**（留出 20 句）：教师 10 步 corr 0.887 / L1 0.784 / 0.41s；教师直接 1 步 corr 0.870 / L1 1.228；**学生 1 步 corr 0.891 / L1 0.691 / 0.062s**（vocoder ~6.6x、端到端 TTS ~13x 加速）。
- **复现**：
  ```bash
  pip install --no-deps git+https://github.com/Resemble-AI/chatterbox.git@master
  pip install -r requirements-chatterbox.txt
  python -m dsflow.chatterbox.distill --fresh-z --steps 12000 --step-dropout 0.5 \
    --lambda-direct 1.5 --lambda-band 2.0 --lambda-stft 2.0 --lambda-level 1.0 \
    --prompt-mix 0.7 --teacher-steps 5 --lr 2e-5
  python -m dsflow.chatterbox.eval --student-ckpt checkpoints/chatterbox/last.pt
  python -m dsflow.chatterbox.demo_tts   # 端到端：教师 10 步 vs 学生 1 步 WAV
  ```

#### 蒸馏微创新：波形域多尺度 STFT loss

纯 mel 域 L1/CFM 收敛后，学生单步 mel 谱形已接近教师（corr ≈ 0.89），但经 HiFT 声码器合成后波形仍明显"毛糙"：
浊音帧强周期比例仅 6%（教师 44%）、整体电平低 5.6x、高频（12k+）能量缺 ~3.8x，听感上"不像同一音色且有噪音"。
原因是 mel 域损失天然偏向平均/平滑解，管不到最终波形的谐波细节。

做法（`dsflow/chatterbox/distill.py`）：

- **可微声码器前向 `vocode_mel`**：复刻 `HiFT.inference`（mel→f0→source→decode）但不带 `torch.inference_mode`，
  冻结教师声码器参数，让梯度穿过声码器直达学生 mel——损失直接作用在"最终听到的波形"上；
- **多尺度 STFT loss**（256/512/1024/2048/4096 五尺度，hop=fft/4）：每尺度 L1 + log-幅度 L1 + 谱收敛
  `||X−Y||_F/||Y||_F`。4096 尺度（~5.9Hz 分辨率）专门抓 F0/谐波区，log-幅度项均衡响亮与微弱频带；
- **配套项**：`masked_band_level_loss`（80 频带逐带平均 log 幅度对齐，治整体电平/频响塌陷）、
  `wav_level_loss`（vocoded 波形 log-RMS 对齐，治响度差）、`--prompt-mix`（训练 batch 按概率混入内置音色
  prompt 条件——真实推理路径带 prompt，纯无 prompt 训练会在此泛化失败）。

效果（demo 句同噪声 A/B，step 12000；留出 20 句）：

  | 指标 | step 2000（原始） | step 12000（+STFT） | 教师 10 步 |
  | --- | --- | --- | --- |
  | mel corr vs 教师 | 0.893 | **0.969** | 1.0 |
  | WAV 电平比（师/学） | 5.6x | **1.15x** | 1.0 |
  | 浊音帧强周期比例 | 6% | **46%** | 45% |
  | 留出集 corr / L1 | 0.891 / 0.691 | **0.894 / 0.681** | 0.887 / 0.784 |

上板（Magnetar/AX650C，U8 INT8）：ONNX 对分 cosine 1.0；板端 ax_run_model 3 样本 cosine **0.9932**（旧模型
0.9876）；板端端到端 181ms/512 帧（RTF 0.0177）。波形级 STFT loss 让 INT8 量化精度也顺带过 0.99 门限。

- **已知限制**：单步学生与教师同噪声下波形样本级相关仍低（~0.04，声码器相位敏感所致，听感已基本一致）；
  进一步逼近需换谱锐度/谐波约束或上 QAT。
## Repository layout

```text
dsflow/
├── config.py        # 数据/模型/训练/推理配置
├── text.py          # g2p-en 音素化 + 字符回退 + 词表
├── audio.py         # wav I/O + log10 幅度梅尔谱（slaney 标度）
├── losses.py        # 双监督损失（CFM + 直接 L1）+ 时长损失
├── model/
│   ├── text_encoder.py   # 文本 Transformer
│   ├── duration.py       # 时长预测器
│   ├── decoder.py        # 步感知 DiT 解码器（AdaLN + step gate，v/x0 双头）
│   └── dsflow.py         # 模型组装与前向
├── train.py          # 训练循环（step dropout、双监督、checkpoint/resume）
├── infer.py          # 单步/多步推理
├── vocoder.py        # HiFi-GAN（缓存）+ Griffin-Lim 兜底
└── data/
    ├── ljspeech.py   # 下载/解压/多进程预处理
    ├── dataset.py    # 比例时长对齐数据集 + collate
    └── prepare.py    # 预处理 CLI
```

`dsflow/chatterbox/`：Chatterbox S3Gen 单步化（教师/学生模型、蒸馏、评估、TTS demo）。

## Known limitations & next steps

- 当前为单说话人（LJSpeech）；多说话人/零样本需引入说话人条件与更大数据；
- 时长对齐使用简单的比例时长（F5-TTS 风格），未做显式对齐；可替换为 MFA/attention aligner 提升韵律；
- 未实现 EMA、classifier-free guidance、蒸馏式单步目标（如一致性损失）——这些都是下一步实验方向；
- 训练脚本目前单卡；多卡 DDP 可在此基础上添加。

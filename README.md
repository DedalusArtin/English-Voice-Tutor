# English Voice Tutor v2.0

中英文双语英语对话教师 — 语音/文字双模式 + 语法纠错 + TTS 朗读

## 架构

```
麦克风/键盘 → faster-whisper ASR → LM Studio (Qwen3.8) → Edge-TTS → 播放
```

## 功能

- **双模输入**: `/v` 语音模式 / `/t` 文字模式，持续同一模式直到命令切换
- **双语教学**: 每条回复 = 英文 + 中文翻译 + 语法纠错
- **智能纠错**: "eat apple" → "eat an apple"，中文解释原因
- **TTS 朗读**: Edge-TTS 自动朗读英文部分
- **自动检测**: 自动识别 LM Studio 加载的模型

## 快速开始

```bash
# 1. 启动 LM Studio, 加载模型, Start Server (端口 1234)
# 2. 运行:
python voice_tutor.py
# 或双击 run.bat
```

## 命令

| 命令 | 说明 |
|------|------|
| `/v` | 切换到语音模式 |
| `/t` | 切换到文字模式 |
| `/q` | 退出 |

## 配置 (voice_tutor.py 顶部)

| 参数 | 默认 | 说明 |
|------|------|------|
| `WHISPER_MODEL` | base | tiny/base/small |
| `TTS_VOICE_EN` | en-US-AriaNeural | 英语语音 |
| `TTS_ENABLED` | True | 语音输出开关 |
| `SILENCE_DURATION` | 1.5 | 静音停止秒数 |

## 依赖

```bash
pip install faster-whisper edge-tts sounddevice numpy requests
```

## 环境

- Windows + Python 3.10+
- LM Studio (本地 API, 端口 1234)
- 麦克风 (语音模式需要)
- 网络 (Edge-TTS 和首次下载 Whisper 模型需要)

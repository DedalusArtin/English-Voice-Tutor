"""
English Voice Tutor v2.0 - 英语语音/文字双模对话教师
=====================================================
模式A: 语音对话 (麦克风 → Whisper ASR → LM Studio → TTS)
模式B: 文字对话 (键盘输入 → LM Studio → TTS)

特色: 中英文双语教学, 纠错+翻译+例句
"""

import os
import sys
import time
import queue
import tempfile
import requests
import numpy as np
import asyncio

# 国内镜像 + 模型缓存到 F 盘
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HOME', r'F:\LM_model\models\huggingface')
os.environ.setdefault('HF_HUB_CACHE', r'F:\LM_model\models\huggingface')

# ===================== 配置 =====================
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODELS_URL = "http://localhost:1234/v1/models"
MODEL_NAME = ""  # 自动检测

# Whisper ASR
WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"

# TTS
TTS_VOICE_EN = "en-US-AriaNeural"   # 英语
TTS_VOICE_ZH = "zh-CN-XiaoxiaoNeural"  # 中文
TTS_RATE = "+0%"
TTS_ENABLED = True  # 语音输出开关

# 录音
SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5
MAX_RECORD = 30

# 系统提示词 (中英文双语教师)
SYSTEM_PROMPT = """You are a friendly bilingual (Chinese-English) conversation tutor. You help users practice English through natural dialogue.

## Response Format (VERY IMPORTANT - follow exactly):
Every response must follow this structure:

**[English]** (your natural conversational reply, 1-3 sentences)
**[中文翻译]** (Chinese translation of your reply)
**[纠错]** (if user made mistakes: briefly explain in Chinese; if no mistakes: "Perfect! / 完美!")

## Rules:
1. Keep English responses SHORT (1-3 sentences) for quick listening.
2. Always correct grammar/vocabulary mistakes gently in the 纠错 section.
3. When correcting, explain WHY in simple Chinese (e.g., "apple是可数名词,前面要加an").
4. Ask follow-up questions to keep conversation going.
5. If user speaks Chinese, translate their meaning to English first, then respond in English.
6. Use simple English appropriate for a learner (HSK4-6 level).
7. Occasionally introduce useful phrases or idioms with explanations.

## Example:
User: "I want to eat apple"
You:
**[English]** That's great! But we should say "I want to eat an apple". What kind of apple do you like?
**[中文翻译]** 很好！但我们应该说"I want to eat an apple"。你喜欢吃什么苹果？
**[纠错]** "apple"是可数名词,单数前面要加冠词"an"。正确说法: eat an apple。

User: "昨天我去了公园"
You:
**[English]** Nice! You can say "I went to the park yesterday". What did you do there?
**[中文翻译]** 不错！你可以说"I went to the park yesterday"。你在那里做了什么？
**[纠错]** 中文"昨天我去了公园"→ 英文语序是"主语+动词+地点+时间": I went to the park yesterday.
"""

# ===================== TTS =====================
def speak_en(text):
    """英语 TTS"""
    if not TTS_ENABLED:
        return
    try:
        import edge_tts
        tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        tmp_path = tmp.name
        tmp.close()
        communicate = edge_tts.Communicate(text, TTS_VOICE_EN, rate=TTS_RATE)
        asyncio.run(communicate.save(tmp_path))
        os.system(f'start "" "{tmp_path}"')
        time.sleep(max(1.5, len(text) * 0.07))
        try:
            os.unlink(tmp_path)
        except:
            pass
    except Exception as e:
        print(f"  [TTS-ERR] {e}")


# ===================== ASR =====================
whisper_model = None

def load_whisper():
    global whisper_model
    if whisper_model is not None:
        return
    print("  [LOAD] 加载 Whisper 语音识别...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print(f"  [OK] Whisper '{WHISPER_MODEL}' 就绪")


def record_and_transcribe():
    """录音 + 识别"""
    import sounddevice as sd
    from faster_whisper import WhisperModel
    
    load_whisper()
    
    print("\n  [MIC] 请说英语... (说完自动停, 也可以说中文)")
    
    audio_queue = queue.Queue()
    frames = []
    silence_start = None
    is_recording = False
    
    def callback(indata, frames_count, time_info, status):
        audio_queue.put(indata.copy())
    
    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='float32', callback=callback):
        start = time.time()
        while True:
            try:
                chunk = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            
            vol = np.abs(chunk).mean()
            if vol > SILENCE_THRESHOLD:
                is_recording = True
                silence_start = None
                frames.append(chunk)
            elif is_recording:
                if silence_start is None:
                    silence_start = time.time()
                    frames.append(chunk)
                elif time.time() - silence_start < SILENCE_DURATION:
                    frames.append(chunk)
                else:
                    break
            
            if time.time() - start > MAX_RECORD:
                break
    
    if not frames:
        return ""
    
    audio = np.concatenate(frames, axis=0).flatten()
    
    print("  [ASR] 识别中...")
    segments, _ = whisper_model.transcribe(audio, beam_size=5, vad_filter=True)
    text = " ".join(seg.text.strip() for seg in segments)
    return text.strip()


# ===================== LLM =====================
def chat(user_text, history):
    """调用 LM Studio"""
    history.append({"role": "user", "content": user_text})
    
    payload = {
        "model": MODEL_NAME or "default",
        "messages": history,
        "temperature": 0.7,
        "max_tokens": 300,
        "stream": False
    }
    
    try:
        resp = requests.post(LM_STUDIO_URL, json=payload, timeout=60)
        resp.raise_for_status()
        reply = resp.json()["choices"][0]["message"]["content"].strip()
        history.append({"role": "assistant", "content": reply})
        return reply
    except requests.exceptions.ConnectionError:
        print("  [ERROR] LM Studio 连接失败!")
        return None
    except Exception as e:
        print(f"  [ERROR] {e}")
        return None


def parse_reply(reply):
    """解析双语回复, 提取英文部分用于 TTS"""
    en_text = ""
    for line in reply.split("\n"):
        line = line.strip()
        if line.startswith("**[English]**"):
            en_text = line.replace("**[English]**", "").strip()
            break
        elif line.startswith("[English]"):
            en_text = line.replace("[English]", "").strip()
            break
    
    if not en_text:
        # fallback: 取第一行非空英文
        for line in reply.split("\n"):
            if line.strip() and not line.startswith("**[") and not line.startswith("["):
                en_text = line.strip()
                break
    
    return en_text


# ===================== LM Studio 检测 =====================
def check_lm_studio():
    global MODEL_NAME
    print("  [CHECK] 检查 LM Studio...")
    try:
        resp = requests.get(LM_MODELS_URL, timeout=5)
        models = resp.json().get("data", [])
        if not models:
            print("  [ERROR] LM Studio 没有加载模型! 请先加载 Qwen3.8")
            return False
        MODEL_NAME = models[0]["id"]
        print(f"  [OK] 模型: {MODEL_NAME}")
        return True
    except:
        print("  [ERROR] LM Studio 未启动!")
        print("  请: 1.打开 LM Studio  2.加载模型  3.Start Server (端口1234)")
        return False


# ===================== 主流程 =====================
def print_reply(reply):
    """格式化显示回复"""
    print()
    for line in reply.split("\n"):
        line = line.rstrip()
        if line.startswith("**[English]**") or line.startswith("[English]"):
            content = line.replace("**[English]**", "").replace("[English]", "").strip()
            print(f"\n  \033[92m[EN] {content}\033[0m")
        elif line.startswith("**[中文翻译]**") or line.startswith("[中文翻译]"):
            content = line.replace("**[中文翻译]**", "").replace("[中文翻译]", "").strip()
            print(f"  \033[96m[中] {content}\033[0m")
        elif line.startswith("**[纠错]**") or line.startswith("[纠错]"):
            content = line.replace("**[纠错]**", "").replace("[纠错]", "").strip()
            print(f"  \033[93m[纠] {content}\033[0m")
        elif line.strip():
            print(f"  {line}")
    print()


def main():
    print()
    print("=" * 52)
    print("   English Voice Tutor v2.0 - 英语双语对话教师")
    print("=" * 52)
    
    if not check_lm_studio():
        input("\n  按回车退出...")
        return
    
    print(f"  ASR:  faster-whisper {WHISPER_MODEL} (CPU {WHISPER_COMPUTE})")
    print(f"  TTS:  {'开 (英文朗读)' if TTS_ENABLED else '关'}")
    print(f"  LLM:  {MODEL_NAME}")
    print("=" * 52)
    print("  命令:")
    print("    /v    切换到语音模式 (麦克风)")
    print("    /t    切换到文字模式 (键盘)")
    print("    /q    退出")
    print("  回复: 英文 + 中文翻译 + 语法纠错")
    print("=" * 52)
    
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    
    # 开场白
    opener = "**[English]** Hi! I'm your English tutor. Let's practice English!**[中文翻译]** 你好! 我是你的英语老师,来练习英语吧!**[纠错]** 输入 /v 用语音, /t 用打字, /q 退出。"
    print_reply(opener)
    speak_en("Hi! I'm your English tutor. Let's practice English.")
    
    # 默认文字模式
    current_mode = 'T'
    
    while True:
        print("  " + "-" * 46)
        
        if current_mode == 'T':
            user_text = input(f"  \033[90m[打字]\033[0m ").strip()
            if not user_text:
                continue
            
            # 命令处理
            cmd = user_text.lower()
            if cmd in ['/q', '/quit', '/exit', '/bye']:
                speak_en("Goodbye! Great practice today!")
                print("\n  再见! 今天的练习很棒!")
                break
            elif cmd in ['/v', '/voice', '/语音']:
                current_mode = 'V'
                print("  \033[92m>> 已切换到语音模式, 请说话\033[0m")
                continue
            elif cmd in ['/t', '/text', '/打字']:
                print("  \033[92m>> 已经在文字模式\033[0m")
                continue
        
        elif current_mode == 'V':
            user_text = record_and_transcribe()
            if not user_text:
                print("  [SKIP] 没识别到, 继续...")
                continue
            
            # 语音命令检测
            low = user_text.lower()
            if any(w in low for w in ['switch to text', 'text mode', '打字', '文字模式']):
                current_mode = 'T'
                print("  \033[92m>> 已切换到文字模式\033[0m")
                continue
            if any(w in low for w in ['goodbye', 'bye bye', 'see you', '退出', '再见']):
                speak_en("Goodbye! Great practice today!")
                print("\n  再见!")
                break
            
            print(f"\n  \033[95m[你说] {user_text}\033[0m")
        
        # LLM
        print("  [AI] 思考中...")
        reply = chat(user_text, history)
        if reply is None:
            continue
        
        # 显示
        print_reply(reply)
        
        # TTS 只朗读英文部分
        en_text = parse_reply(reply)
        if en_text and TTS_ENABLED:
            speak_en(en_text)


if __name__ == "__main__":
    main()

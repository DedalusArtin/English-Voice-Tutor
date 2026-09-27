"""
English Voice Tutor v2.1 - 英语语音/文字双模对话教师
=====================================================
模式: 语音(V) / 文字(T) / 暂停菜单(Ctrl+C)
特色: 中英文双语教学 + 语法纠错 + 在线搜索 + TTS
"""

import os
import sys
import time
import queue
import tempfile
import requests
import numpy as np
import asyncio
import threading

# 国内镜像 + 模型缓存到 F 盘
os.environ.setdefault('HF_ENDPOINT', 'https://hf-mirror.com')
os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HOME', r'F:\LM_model\models\huggingface')
os.environ.setdefault('HF_HUB_CACHE', r'F:\LM_model\models\huggingface')

# ===================== 配置 =====================
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODELS_URL = "http://localhost:1234/v1/models"
MODEL_NAME = ""

WHISPER_MODEL = "base"
WHISPER_DEVICE = "cpu"
WHISPER_COMPUTE = "int8"

TTS_VOICE_EN = "en-US-AriaNeural"
TTS_ENABLED = True
TTS_RATE = "+0%"

SAMPLE_RATE = 16000
SILENCE_THRESHOLD = 0.01
SILENCE_DURATION = 1.5
MAX_RECORD = 30

SYSTEM_PROMPT = """You are a friendly bilingual (Chinese-English) conversation tutor. You help users practice English through natural dialogue.

## Response Format (follow exactly):
**[English]** (natural conversational reply, 1-3 sentences)
**[中文翻译]** (Chinese translation)
**[纠错]** (if mistakes: explain in Chinese; if none: "Perfect! / 完美!")

## Rules:
1. Keep English SHORT (1-3 sentences).
2. Correct grammar/vocabulary gently in the correction section.
3. Explain corrections in simple Chinese.
4. Ask follow-up questions.
5. If user speaks Chinese, translate to English first.
6. Use simple English for learners.
"""

# ===================== TTS =====================
def speak_en(text):
    if not TTS_ENABLED or not text:
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
        try: os.unlink(tmp_path)
        except: pass
    except Exception as e:
        print(f"  [TTS-ERR] {e}")

# ===================== ASR =====================
whisper_model = None

def load_whisper():
    global whisper_model
    if whisper_model is not None:
        return
    print("  [LOAD] 加载 Whisper...")
    from faster_whisper import WhisperModel
    whisper_model = WhisperModel(WHISPER_MODEL, device=WHISPER_DEVICE, compute_type=WHISPER_COMPUTE)
    print(f"  [OK] Whisper '{WHISPER_MODEL}' 就绪")

def record_and_transcribe():
    import sounddevice as sd
    load_whisper()
    
    print("\n  \033[91m[MIC]\033[0m 请说话... (Ctrl+C 暂停)")
    
    audio_queue = queue.Queue()
    frames = []
    silence_start = None
    is_recording = False
    
    def callback(indata, fc, ti, status):
        audio_queue.put(indata.copy())
    
    try:
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
    except KeyboardInterrupt:
        raise  # 让上层处理 Ctrl+C
    
    if not frames:
        return ""
    
    audio = np.concatenate(frames, axis=0).flatten()
    print("  [ASR] 识别中...")
    segments, _ = whisper_model.transcribe(audio, beam_size=5, vad_filter=True)
    return " ".join(seg.text.strip() for seg in segments).strip()

# ===================== Web Search =====================
SEARCH_KEYWORDS = [
    'latest', 'newest', 'current', 'today', 'now', 'recent', '2025', '2026',
    'news', 'price', 'stock', 'weather', 'who is', 'what is the latest',
    '最新', '现在', '今天', '新闻', '股价', '天气', '目前', '当前',
]

def needs_search(text):
    return any(kw in text.lower() for kw in SEARCH_KEYWORDS)

def web_search(query, max_results=3):
    try:
        from ddgs import DDGS
        with DDGS(proxy='http://127.0.0.1:7890') as d:
            results = list(d.text(query, max_results=max_results))
        return '\n'.join(f"{r.get('title','')}: {r.get('body','')}" for r in results)
    except Exception as e:
        print(f"  [SEARCH-ERR] {e}")
        return ''

def search_and_inject(text):
    if not needs_search(text):
        return text
    print("  [SEARCH] 联网搜索中...")
    results = web_search(text)
    if results:
        return f"[Web Search Results]\n{results}\n\n[User Question]\n{text}\n\nAnswer based on search results if relevant."
    return text

# ===================== LLM =====================
def chat(user_text, history):
    history.append({"role": "user", "content": user_text})
    payload = {"model": MODEL_NAME or "default", "messages": history, "temperature": 0.7, "max_tokens": 300, "stream": False}
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

import re

def parse_reply(reply):
    """用正则提取英文句子用于 TTS"""
    # 1. 优先找 [English] 标记, 到下一个标记截断
    for sep in [r'\*\*\[English\]\*\*', r'\[English\]', r'\*\*\(English\)\*\*']:
        m = re.search(sep + r'\s*(.+?)(?=\s*\*?\*?\[(中文翻译|Translation|纠错|Correction|Example|例句)\]|$)', reply, re.DOTALL)
        if m:
            text = m.group(1).strip().rstrip('*').strip()
            if text and len(text) > 3:
                return text
    
    # 2. 找真正的英文句子 (跳过词典条目/markdown)
    for line in reply.split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('**[') or line.startswith('|') or line.startswith('---') or line.startswith('```'):
            continue
        clean = re.sub(r'\*\*|__|~~|`', '', line).strip()
        # 跳过: 发音标记 / 音标 / 词典格式
        if '/' in clean and len(clean) < 30:
            continue
        if re.match(r'^[a-zA-Z]+\s*/.*?/$', clean):
            continue
        # 必须是完整句子: 有空格, >15字符, 含动词/常见词
        if re.search(r'[a-zA-Z]{3,}', clean) and len(clean) > 15 and ' ' in clean:
            return clean
    return ""


def parse_examples(reply):
    """提取例句"""
    examples = []
    # 找 [Example] 标记
    for m in re.finditer(r'\*\*\[Example\]\*\*\s*(.+)', reply):
        text = m.group(1).strip().rstrip('*').strip()
        if text:
            examples.append(text)
    # 找 [例句] 标记
    for m in re.finditer(r'\*\*\[例句\]\*\*\s*(.+)', reply):
        text = m.group(1).strip().rstrip('*').strip()
        if text:
            examples.append(text)
    # fallback: 找引号里的句子
    if not examples:
        for m in re.finditer(r'["""](.+?)["""]', reply):
            text = m.group(1).strip()
            if len(text) > 5 and re.search(r'[a-zA-Z]{3,}', text):
                examples.append(text)
    return examples


def print_reply(reply):
    """格式化显示, 跳过 markdown 噪音"""
    print()
    for line in reply.split('\n'):
        line = line.rstrip()
        if not line.strip():
            continue
        
        clean = re.sub(r'\*\*|__|~~', '', line).strip()
        
        # 标题
        if line.startswith('#'):
            clean = re.sub(r'^#+\s*', '', clean)
            print(f"  \033[95m{clean}\033[0m")
            continue
        # 表格分隔线
        if re.match(r'^\|[\s\-:|]+\|$', line) or line.startswith('---'):
            continue
        # 格式化输出
        if re.match(r'\[English\]|\(English\)', clean):
            content = re.sub(r'^\[(English|English)\]|\(English\)', '', clean).strip()
            print(f"  \033[92m[EN] {content}\033[0m")
        elif re.match(r'\[中文翻译\]|\[Translation\]|\[译文\]', clean):
            content = re.sub(r'^\[(中文翻译|Translation|译文)\]', '', clean).strip()
            print(f"  \033[96m[中] {content}\033[0m")
        elif re.match(r'\[纠错\]|\[Correction\]', clean):
            content = re.sub(r'^\[(纠错|Correction)\]', '', clean).strip()
            print(f"  \033[93m[纠] {content}\033[0m")
        elif re.match(r'\[Example\]|\[例句\]', clean):
            content = re.sub(r'^\[(Example|例句)\]', '', clean).strip()
            print(f"  \033[96m[例] {content}\033[0m")
        elif clean.startswith('|'):
            # 表格行, 简化显示
            cells = [c.strip() for c in clean.split('|') if c.strip()]
            if cells and len(cells) >= 2:
                print(f"  \033[90m{cells[0]} → {cells[1]}\033[0m")
            continue
        else:
            print(f"  {clean}")
    print()

# ===================== LM Studio 检测 =====================
def check_lm_studio():
    global MODEL_NAME
    print("  [CHECK] 检查 LM Studio...")
    try:
        resp = requests.get(LM_MODELS_URL, timeout=5)
        models = resp.json().get("data", [])
        if not models:
            print("  [ERROR] LM Studio 没有加载模型!")
            return False
        MODEL_NAME = models[0]["id"]
        print(f"  [OK] 模型: {MODEL_NAME}")
        return True
    except:
        print("  [ERROR] LM Studio 未启动! 请先: 打开LM Studio → 加载模型 → Start Server")
        return False

# ===================== 暂停菜单 =====================
def pause_menu(current_mode):
    """Ctrl+C 暂停菜单, 返回新模式 ('V'/'T'/'Q')"""
    print()
    print("  \033[93m===== 暂停 =====\033[0m")
    print("  当前模式:", "语音" if current_mode == 'V' else "文字")
    print()
    print("  [1] 继续 (回到当前模式)")
    print("  [2] 切换到文字模式")
    print("  [3] 切换到语音模式")
    print("  [4] 开/关 语音朗读")
    print("  [q] 退出")
    
    choice = input("\n  选择: ").strip().lower()
    if choice == '1' or choice == '':
        return current_mode
    elif choice == '2':
        return 'T'
    elif choice == '3':
        return 'V'
    elif choice == '4':
        global TTS_ENABLED
        TTS_ENABLED = not TTS_ENABLED
        print(f"  语音朗读: {'开' if TTS_ENABLED else '关'}")
        return current_mode
    elif choice == 'q':
        return 'Q'
    return current_mode

# ===================== 主流程 =====================
def main():
    global TTS_ENABLED
    print()
    print("=" * 54)
    print("   English Voice Tutor v2.1 - 英语双语对话教师")
    print("=" * 54)
    
    if not check_lm_studio():
        input("\n  按回车退出...")
        return
    
    print(f"  ASR:  faster-whisper {WHISPER_MODEL} (CPU {WHISPER_COMPUTE})")
    print(f"  TTS:  {'开' if TTS_ENABLED else '关'}")
    print(f"  LLM:  {MODEL_NAME}")
    print(f"  搜索: DuckDuckGo (自动检测联网问题)")
    print("=" * 54)
    print()
    print("  \033[93m===== 快捷键 =====\033[0m")
    print("  \033[96mCtrl+C\033[0m  随时暂停, 弹出菜单 (切换模式/退出)")
    print()
    print("  \033[92m文字模式命令:\033[0m")
    print("    /v      切换到语音模式")
    print("    /s      朗读上一条回复")
    print("    /ex     生成3个例句并逐句播放")
    print("    /mute   关闭自动朗读")
    print("    /unmute 开启自动朗读")
    print("    /q      退出")
    print()
    print("  \033[92m语音模式:\033[0m")
    print("    说 'switch to text' / '打字' → 切到文字")
    print("    说 'goodbye' / '再见' → 退出")
    print()
    print("  回复格式: 英文 + 中文翻译 + 语法纠错")
    print("=" * 54)
    
    history = [{"role": "system", "content": SYSTEM_PROMPT}]
    last_reply = ""
    
    opener = "**[English]** Hi! I'm your English tutor. Let's practice English!**[中文翻译]** 你好! 我是你的英语老师,来练习英语吧!**[纠错]** 按 Ctrl+C 可以随时暂停或切换模式。"
    print_reply(opener)
    speak_en("Hi! I'm your English tutor. Let's practice English.")
    
    current_mode = 'T'  # 默认文字模式
    
    while True:
        print("  " + "-" * 48)
        print(f"  \033[90m模式: {'语音 [V]' if current_mode == 'V' else '文字 [T]'} | Ctrl+C 暂停\033[0m")
        
        try:
            if current_mode == 'T':
                user_text = input("  \033[90m[打字]\033[0m ").strip()
                if not user_text:
                    continue
                
                cmd = user_text.lower()
                if cmd in ['/q', '/quit', '/exit', '/bye']:
                    speak_en("Goodbye! Great practice today!")
                    print("\n  再见!")
                    break
                elif cmd in ['/v', '/voice']:
                    current_mode = 'V'
                    print("  \033[92m>> 已切换到语音模式\033[0m")
                    continue
                elif cmd in ['/t', '/text']:
                    print("  \033[92m>> 已在文字模式\033[0m")
                    continue
                elif cmd in ['/speak', '/s', '/读']:
                    # 重读上一条回复
                    if last_reply:
                        en_text = parse_reply(last_reply)
                        if en_text:
                            print(f"  \033[92m[朗读] {en_text}\033[0m")
                            speak_en(en_text)
                        else:
                            print("  [!] 没有可朗读的内容")
                    else:
                        print("  [!] 还没有回复内容")
                    continue
                elif cmd in ['/mute', '/静音']:
                    TTS_ENABLED = False
                    print("  \033[93m>> 语音朗读已关闭\033[0m")
                    continue
                elif cmd in ['/unmute', '/开声音']:
                    TTS_ENABLED = True
                    print("  \033[92m>> 语音朗读已开启\033[0m")
                    continue
                elif cmd in ['/ex', '/例句', '/example']:
                    # 生成例句并播放
                    if last_reply:
                        ex_text = parse_reply(last_reply)
                        if ex_text:
                            print("  \033[96m[例句生成中...]\033[0m")
                            ex_prompt = f'Give 3 simple example sentences using key phrases from this reply. Format each as: **[Example]** sentence. Only give examples, nothing else.\n\nReply: {ex_text}'
                            ex_reply = chat(ex_prompt, [{"role": "system", "content": "You generate example English sentences. Output 3 sentences, each on its own line starting with **[Example]**. No extra text."}])
                            if ex_reply:
                                ex_list = parse_examples(ex_reply)
                                if not ex_list:
                                    # fallback: 按行拆
                                    ex_list = [l.strip().rstrip('*').strip() for l in ex_reply.split('\n') if l.strip() and 'Example' not in l and l.strip()]
                                for i, ex in enumerate(ex_list[:3], 1):
                                    print(f"  \033[96m[例句{i}] {ex}\033[0m")
                                
                                # 逐句播放, 按回车继续
                                for i, ex in enumerate(ex_list[:3], 1):
                                    input(f"  \033[90m按回车播放例句{i}...\033[0m")
                                    speak_en(ex)
                    else:
                        print("  [!] 还没有回复内容")
                    continue
            
            elif current_mode == 'V':
                user_text = record_and_transcribe()
                if not user_text:
                    print("  [SKIP] 没识别到, 继续...")
                    continue
                
                low = user_text.lower()
                if any(w in low for w in ['switch to text', 'text mode', '打字', '文字模式']):
                    current_mode = 'T'
                    print("  \033[92m>> 已切换到文字模式\033[0m")
                    continue
                if any(w in low for w in ['goodbye', 'bye bye', 'see you', '再见', '退出']):
                    speak_en("Goodbye! Great practice today!")
                    print("\n  再见!")
                    break
                
                print(f"\n  \033[95m[你说] {user_text}\033[0m")
        
        except KeyboardInterrupt:
            # Ctrl+C → 暂停菜单
            new_mode = pause_menu(current_mode)
            if new_mode == 'Q':
                speak_en("Goodbye!")
                print("\n  再见!")
                break
            current_mode = new_mode
            continue
        
        # 搜索增强
        enhanced_text = search_and_inject(user_text)
        
        # LLM
        try:
            print("  [AI] 思考中...")
            reply = chat(enhanced_text, history)
        except KeyboardInterrupt:
            continue
        
        if reply is None:
            continue
        
        last_reply = reply
        print_reply(reply)
        
        en_text = parse_reply(reply)
        if en_text and TTS_ENABLED:
            try:
                speak_en(en_text)
            except KeyboardInterrupt:
                pass


if __name__ == "__main__":
    main()

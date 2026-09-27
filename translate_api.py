"""
DeepLX 兼容本地翻译 API
========================
用 LM Studio 的小模型做翻译, 暴露 DeepLX 格式接口。
给截图里的翻译工具用: 接口地址填 http://localhost:8080/translate

用法:
  python translate_api.py
  
  # 测试:
  curl -X POST http://localhost:8080/translate \
    -H "Content-Type: application/json" \
    -d '{"text":"Hello world","target_lang":"ZH"}'
"""

import json
import requests
from http.server import HTTPServer, BaseHTTPRequestHandler

# ===================== 配置 =====================
LISTEN_HOST = "127.0.0.1"
LISTEN_PORT = 8080

# LM Studio API (翻译模型)
LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODELS_URL = "http://localhost:1234/v1/models"

# 如果 LM Studio 里加载了单独的翻译模型, 填模型名
# 留空 = 用当前加载的模型
TRANSLATE_MODEL = ""

# 翻译提示词
TRANSLATE_PROMPT = """You are a professional translator. Translate the following text.
Rules:
- Output ONLY the translation, no explanations
- Preserve formatting and punctuation
- If text is already in target language, return as-is

Target language: {target}
Source language: {source}

Text to translate:
{text}"""


# ===================== 语言映射 =====================
LANG_MAP = {
    "ZH": "Chinese (中文)",
    "EN": "English",
    "JA": "Japanese (日本語)",
    "KO": "Korean (한국어)",
    "FR": "French (Français)",
    "DE": "German (Deutsch)",
    "ES": "Spanish (Español)",
    "RU": "Russian (Русский)",
    "AUTO": "auto-detect",
}


def detect_lang_hint(text):
    """简单语言检测"""
    # 中文字符
    cjk = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    if cjk > len(text) * 0.1:
        return "ZH"
    # 日文假名
    kana = sum(1 for c in text if '\u3040' <= c <= '\u30ff')
    if kana > 0:
        return "JA"
    return "EN"


def translate(text, source_lang="AUTO", target_lang="EN"):
    """调用 LM Studio 翻译"""
    if not text.strip():
        return ""
    
    # 自动检测源语言
    if source_lang == "AUTO":
        source_lang = detect_lang_hint(text)
    
    src_name = LANG_MAP.get(source_lang, source_lang)
    tgt_name = LANG_MAP.get(target_lang, target_lang)
    
    prompt = TRANSLATE_PROMPT.format(
        source=src_name, target=tgt_name, text=text
    )
    
    payload = {
        "model": TRANSLATE_MODEL or "default",
        "messages": [
            {"role": "system", "content": "You are a translation engine. Output only the translation."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 2000,
        "stream": False
    }
    
    resp = requests.post(LM_STUDIO_URL, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()["choices"][0]["message"]["content"].strip()
    
    # 清理可能的多余输出
    for prefix in ["Translation:", "译文:", "翻译:"]:
        if result.startswith(prefix):
            result = result[len(prefix):].strip()
    
    return result


# ===================== HTTP Handler =====================
class DeepLXHandler(BaseHTTPRequestHandler):
    
    def do_POST(self):
        if self.path == "/translate":
            try:
                content_length = int(self.headers['Content-Length'])
                body = json.loads(self.rfile.read(content_length))
                
                text = body.get("text", "")
                source_lang = body.get("source_lang", "AUTO").upper()
                target_lang = body.get("target_lang", "EN").upper()
                
                print(f"  [{source_lang} -> {target_lang}] {text[:60]}...")
                
                result = translate(text, source_lang, target_lang)
                
                response = {
                    "code": 200,
                    "id": 123456789,
                    "data": result,
                    "alternatives": [result],
                    "source_lang": source_lang,
                    "target_lang": target_lang,
                }
                
            except Exception as e:
                print(f"  [ERROR] {e}")
                response = {"code": 500, "message": str(e), "data": ""}
            
            self._send_json(response)
    
    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok", "service": "DeepLX Local Translate"})
        else:
            self._send_json({"code": 404, "message": "Not found"})
    
    def _send_json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))
    
    def log_message(self, format, *args):
        pass  # 静默日志


# ===================== 启动 =====================
def main():
    print("=" * 48)
    print("   DeepLX 本地翻译 API")
    print("=" * 48)
    
    # 检查 LM Studio
    try:
        resp = requests.get(LM_MODELS_URL, timeout=5)
        models = resp.json().get("data", [])
        if models:
            model_name = models[0]["id"]
            if TRANSLATE_MODEL:
                model_name = TRANSLATE_MODEL
            print(f"  LM Studio: 在线, 模型: {model_name}")
        else:
            print("  [WARN] LM Studio 没有加载模型!")
    except:
        print("  [ERROR] LM Studio 未启动! 翻译会失败")
    
    print(f"  监听: http://{LISTEN_HOST}:{LISTEN_PORT}")
    print(f"  接口: http://{LISTEN_HOST}:{LISTEN_PORT}/translate")
    print("=" * 48)
    print("  在翻译工具里填: http://localhost:8080/translate")
    print("  API Key 留空")
    print("=" * 48)
    
    server = HTTPServer((LISTEN_HOST, LISTEN_PORT), DeepLXHandler)
    print("  服务已启动, Ctrl+C 停止\n")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  服务已停止")
        server.server_close()


if __name__ == "__main__":
    main()

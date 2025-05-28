import requests

proxies = {
    "http": "http://127.0.0.1:1080",
    "https": "http://127.0.0.1:1080",
}

try:
    r = requests.get("https://wandb.ai", proxies=proxies, timeout=10)
    print("连接成功，状态码：", r.status_code)
except Exception as e:
    print("连接失败：", e)
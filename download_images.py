import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

# ———— 配置区 ————
# 要抓取的网页 URL（页面上的所有图片链接会被下载）
BASE_URL = "https://www.medetec.co.uk/slide%20scans/pressure-ulcer-images-b/index.html"
# 本地保存图片的目录（请自行修改为想要存放图片的路径）
SAVE_DIR = r"your_path\data\download_ulcers"  # 例如：r"C:\Users\YourName\Pictures\PressureUlcers"

# 支持的图片扩展名，用来过滤非图片链接
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tif", ".tiff"}

def ensure_directory(path: str):
    """
    如果目录不存在，就递归创建
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

def is_image_url(url: str) -> bool:
    """
    判断给定的 URL 是否以有效图片后缀结尾
    """
    lower = url.lower()
    for ext in VALID_EXTENSIONS:
        if lower.endswith(ext):
            return True
    return False

def download_image(img_url: str, save_folder: str):
    """
    下载单张图片到指定目录
    """
    try:
        response = requests.get(img_url, stream=True, timeout=10)
        response.raise_for_status()
        # 从 URL 中提取文件名
        filename = os.path.basename(img_url.split("?")[0])
        save_path = os.path.join(save_folder, filename)
        # 如果文件已存在，则跳过
        if os.path.exists(save_path):
            print(f"[跳过] {filename} 已存在")
            return

        # 将内容写入到本地文件
        with open(save_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:
                    f.write(chunk)
        print(f"[下载完成] {filename}")
    except Exception as e:
        print(f"[下载失败] {img_url}  错误信息：{e}")

def main():
    # 1. 确保保存目录存在
    ensure_directory(SAVE_DIR)

    # 2. 获取网页内容
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/100.0.0.0 Safari/537.36"
        }
        resp = requests.get(BASE_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        html = resp.text
    except Exception as e:
        print(f"[错误] 无法获取页面：{BASE_URL}\n{e}")
        return

    # 3. 解析 HTML，提取所有可能的图片链接
    soup = BeautifulSoup(html, "html.parser")
    img_urls = set()

    # 3.1 先查找 <img> 标签
    for img_tag in soup.find_all("img"):
        src = img_tag.get("src")
        if not src:
            continue
        # 拼接成绝对 URL
        full_url = urljoin(BASE_URL, src)
        if is_image_url(full_url):
            img_urls.add(full_url)

    # 3.2 有些页面可能直接在 <a> 标签里链接到高清大图，这里也一并查找
    for a_tag in soup.find_all("a"):
        href = a_tag.get("href")
        if not href:
            continue
        full_url = urljoin(BASE_URL, href)
        if is_image_url(full_url):
            img_urls.add(full_url)

    if not img_urls:
        print("未在页面中找到任何图片链接，请确认网页结构是否发生变化。")
        return

    print(f"共找到 {len(img_urls)} 张图片，开始下载……\n")
    # 4. 遍历集合，逐张下载
    for url in sorted(img_urls):
        download_image(url, SAVE_DIR)

if __name__ == "__main__":
    main()

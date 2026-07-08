import time
from pathlib import Path
from playwright.sync_api import sync_playwright


def unlock_and_convert(input_html_path, output_pdf_path):
    # 1. 路径处理
    input_path = Path(input_html_path).resolve()
    file_url = f"file://{input_path}"

    with sync_playwright() as p:
        print("启动浏览器...")
        browser = p.chromium.launch(channel="msedge", headless=True)
        page = browser.new_page()

        print(f"正在加载: {input_html_path}")
        page.goto(file_url, wait_until="networkidle")

        # --- 💥 核心修复：暴力破解 CSS 滚动锁 ---
        print("正在暴力解除网页滚动限制...")
        page.add_style_tag(content="""
            html, body {
                overflow: visible !important;
                overflow-y: visible !important;
                height: auto !important;
                max-height: none !important;
                position: static !important;
            }
            /* 解锁可能存在的内部滚动容器 */
            div, section, main, article {
                overflow: visible !important;
                max-height: none !important;
            }
            /* 隐藏可能遮挡视线的固定弹窗 */
            header, nav, .fixed, .sticky {
                position: absolute !important;
            }
        """)

        # 等待样式生效
        time.sleep(1)

        # 重新计算高度
        dimensions = page.evaluate("""() => {
            return {
                height: document.documentElement.scrollHeight,
                bodyHeight: document.body.scrollHeight
            }
        }""")

        # 取最大的那个高度
        real_height = max(dimensions['height'], dimensions['bodyHeight'])
        print(f"破解后检测到的页面高度: {real_height}px")

        if real_height < 2000:
            print("⚠️ 警告：页面高度依然很小，说明源文件里可能真的没有数据！")

        # 生成 PDF
        print("正在生成长图 PDF...")
        page.pdf(
            path=output_pdf_path,
            print_background=True,
            width="1280px",
            height=f"{real_height + 100}px",  # 加上余量
            margin={"top": "0", "bottom": "0", "left": "0", "right": "0"}
        )

        browser.close()
        print(f"转换完成: {output_pdf_path}")


if __name__ == "__main__":
    # ⚠️ 替换为你刚才确认过路径的文件
    input_file = r"C:\Users\CTOS\Downloads\LMArena ｜ Benchmark & Compare the Best AI Models (2026_1_14 17：55：08).html"
    output_file = "unlocked_result.pdf"

    import os

    if os.path.exists(input_file):
        unlock_and_convert(input_file, output_file)
    else:
        print("❌ 找不到文件，请检查路径！")
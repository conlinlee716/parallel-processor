#!/usr/bin/env python3
"""
生成Mermaid图表图片的脚本

使用方法：
python generate_mermaid_image.py

注意：需要安装selenium和webdriver-manager
"""

import os
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager

def generate_mermaid_image():
    # 配置Chrome选项
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')  # 无头模式
    options.add_argument('--disable-gpu')
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    
    try:
        # 初始化浏览器
        print("初始化Chrome浏览器...")
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()),
            options=options
        )
        
        # 读取HTML文件
        html_path = os.path.abspath('mermaid_chart.html')
        file_url = f"file://{html_path}"
        
        # 打开HTML文件
        print(f"打开HTML文件: {file_url}")
        driver.get(file_url)
        
        # 等待Mermaid渲染完成
        print("等待Mermaid图表渲染...")
        time.sleep(2)  # 等待2秒确保渲染完成
        
        # 找到Mermaid图表元素
        mermaid_element = driver.find_element(By.CLASS_NAME, 'mermaid')
        
        # 截取图表部分的屏幕截图
        screenshot_path = os.path.abspath('mermaid_chart.png')
        print(f"生成截图: {screenshot_path}")
        mermaid_element.screenshot(screenshot_path)
        
        print("图表生成成功！")
        print(f"图片路径: {screenshot_path}")
        
    except Exception as e:
        print(f"生成图表时出错: {e}")
        print("请确保已安装Chrome浏览器和相关依赖：")
        print("pip install selenium webdriver-manager")
        
        # 如果无法使用selenium，提供备选方案
        print("\n备选方案：")
        print("1. 手动打开mermaid_chart.html文件")
        print("2. 使用浏览器的截图功能保存为图片")
        
    finally:
        try:
            driver.quit()
        except:
            pass

if __name__ == "__main__":
    generate_mermaid_image()
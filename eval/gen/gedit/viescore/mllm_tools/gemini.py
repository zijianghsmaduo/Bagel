import base64
import requests
from io import BytesIO
from typing import Union, Optional, Tuple, List
from PIL import Image, ImageOps
import os
import google.generativeai as genai
import time

# --- 辅助函数 (大部分保持不变) ---

def get_api_key(file_path):
    """从文件的第一行读取 API 密钥。"""
    with open(file_path, 'r') as file:
        return file.readline().strip()

def pick_next_item(current_item, item_list):
    """从列表中循环选择下一个项目。"""
    if current_item not in item_list:
        raise ValueError("Current item is not in the list")
    current_index = item_list.index(current_item)
    next_index = (current_index + 1) % len(item_list)
    return item_list[next_index]

def load_image(image: Union[str, Image.Image], format: str = "RGB", size: Optional[Tuple] = None) -> Image.Image:
    """从路径、URL或PIL对象加载图像。"""
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(f"Incorrect path or url: {image}")
    elif not isinstance(image, Image.Image):
        raise ValueError("Incorrect format for image. Should be a url, a local path, or a PIL image.")
    
    image = ImageOps.exif_transpose(image)
    image = image.convert(format)
    if size is not None:
        image = image.resize(size, Image.LANCZOS)
    return image

# --- Gemini 模型封装类 ---

class GeminiProVision:
    def __init__(self, api_key_path='keys/gemini.env', model_name="gemini-1.5-pro-latest"):
        """Google Gemini Pro Vision 模型封装
        Args:
            api_key_path (str or list): API密钥文件路径或路径列表。
            model_name (str): 要使用的 Gemini 模型名称。
        """
        self.multiple_api_keys = False
        self.current_key_file = None
        self.key_lists = None
        
        if isinstance(api_key_path, list):
            self.key_lists = api_key_path
            self.current_key_file = api_key_path[0]
            self.api_key = get_api_key(self.current_key_file)
            self.multiple_api_keys = True
        else:
            # 兼容直接传入 key 字符串或单个文件路径
            if os.path.exists(api_key_path):
                self.api_key = get_api_key(api_key_path)
            else:
                self.api_key = api_key_path
        
        if not self.api_key:
            raise ValueError("Gemini API key not found or provided.")

        self.model_name = model_name
        self.update_client()

    def update_client(self):
        """使用当前的 API 密钥配置 genai 客户端并创建模型实例。"""
        print(f"Configuring Gemini with a new key...")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    def prepare_prompt(self, image_links: List = [], text_prompt: str = "") -> list:
        """为 Gemini API 准备 prompt 列表。
        
        Gemini可以直接接受PIL Image对象，无需base64编码。
        """
        prompt_content = []
        # 文本部分必须在前面
        if text_prompt:
            prompt_content.append(text_prompt)

        if not isinstance(image_links, list):
            image_links = [image_links]
        
        for image_link in image_links:
            image = load_image(image_link)
            prompt_content.append(image)
            
        return prompt_content

    def get_parsed_output(self, prompt: list, max_retries=2):
        """发送请求到 Gemini API 并处理响应，包含重试和密钥切换逻辑。"""
        current_retry = 0
        while current_retry <= max_retries:
            try:
                response = self.model.generate_content(prompt)
                # 成功获取响应后直接返回文本
                return response.text
            except genai.types.generation_types.BlockedPromptException as e:
                print(f"Prompt was blocked by safety settings: {e}")
                return "Error: The prompt was blocked due to safety concerns."
            except Exception as e:
                # 捕获所有 google.api_core.exceptions 里的错误，例如 ResourceExhausted
                error_str = str(e).lower()
                if 'resource has been exhausted' in error_str or 'rate limit' in error_str:
                    print(f"Rate limit or quota exceeded. Error: {e}")
                    if self.multiple_api_keys and current_retry < max_retries:
                        print("Attempting to switch API key...")
                        self.switch_to_next_key()
                        current_retry += 1
                        print("Retrying with new key...")
                        time.sleep(2) # 等待2秒再重试
                    else:
                        print("No more keys to try or max retries reached.")
                        return f"Error: Rate limit or quota exceeded. No more retries. Last error: {e}"
                else:
                    # 其他类型的API错误
                    print(f"An unexpected API error occurred: {e}")
                    return f"Error: An unexpected API error occurred: {e}"
        return "Error: Failed to get a response after multiple retries."

    def switch_to_next_key(self):
        """切换到下一个 API 密钥并更新客户端。"""
        new_key_file = pick_next_item(self.current_key_file, self.key_lists)
        self.api_key = get_api_key(new_key_file)
        self.current_key_file = new_key_file
        print(f"Switched to new key from file: {new_key_file}")
        self.update_client()

# --- 主程序入口 ---

if __name__ == "__main__":
    # 假设您有一个名为 'keys/gemini.env' 的文件，其中包含您的 Gemini API 密钥
    # 或者一个包含多个密钥文件路径的列表
    # key_files = ['keys/gemini_key1.env', 'keys/gemini_key2.env']
    
    try:
        # 使用单个密钥文件
        model = GeminiProVision(api_key_path='keys/gemini.env')
        
        # 或者使用多个密钥文件列表
        # model = GeminiProVision(api_key_path=['keys/gemini_key1.env', 'keys/gemini_key2.env'])

        prompt_list = model.prepare_prompt(
            image_links=['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 
            text_prompt='What is the difference between these two images?'
        )
        
        print("--- Prompt Content ---")
        # 打印文本和图像对象类型
        for item in prompt_list:
            if isinstance(item, str):
                print(f"Text: {item}")
            elif isinstance(item, Image.Image):
                print(f"Image: <PIL.Image object size={item.size}>")
        print("-" * 22)

        res = model.get_parsed_output(prompt_list)
        
        print("\n--- Gemini Pro 1.5 Result ---")
        print(res)

    except FileNotFoundError:
        print("\nError: API key file not found. Please create a file (e.g., 'keys/gemini.env') and place your Gemini API key in it.")
    except Exception as e:
        print(f"\nAn error occurred: {e}")
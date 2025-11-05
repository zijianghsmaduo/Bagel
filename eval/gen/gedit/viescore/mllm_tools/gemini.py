import base64
import requests
from io import BytesIO, StringIO
from typing import Union, Optional, Tuple, List
from PIL import Image, ImageOps
import os
import json
from google import genai
from google.genai import types


def get_api_key(file_path):
    # Read the API key from the first line of the file
    with open(file_path, 'r') as file:
        return file.readline().strip()

def pick_next_item(current_item, item_list):
    if current_item not in item_list:
        raise ValueError("Current item is not in the list")
    current_index = item_list.index(current_item)
    next_index = (current_index + 1) % len(item_list)

    return item_list[next_index]

# Function to encode a PIL image
def encode_pil_image(pil_image):
    # Create an in-memory binary stream
    image_stream = BytesIO()
    
    # Save the PIL image to the binary stream in JPEG format (you can change the format if needed)
    pil_image.save(image_stream, format='JPEG')
    
    # Get the binary data from the stream and encode it as base64
    image_data = image_stream.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    return base64_image


def load_image(image: Union[str, Image.Image], format: str = "RGB", size: Optional[Tuple] = None) -> Image.Image:
    """
    Load an image from a given path or URL and convert it to a PIL Image.

    Args:
        image (Union[str, Image.Image]): The image path, URL, or a PIL Image object to be loaded.
        format (str, optional): Desired color format of the resulting image. Defaults to "RGB".
        size (Optional[Tuple], optional): Desired size for resizing the image. Defaults to None.

    Returns:
        Image.Image: A PIL Image in the specified format and size.

    Raises:
        ValueError: If the provided image format is not recognized.
    """
    if isinstance(image, str):
        if image.startswith("http://") or image.startswith("https://"):
            image = Image.open(requests.get(image, stream=True).raw)
        elif os.path.isfile(image):
            image = Image.open(image)
        else:
            raise ValueError(
                f"Incorrect path or url, URLs must start with `http://` or `https://`, and {image} is not a valid path"
            )
    elif isinstance(image, Image.Image):
        image = image
    else:
        raise ValueError(
            "Incorrect format used for image. Should be an url linking to an image, a local path, or a PIL image."
        )
    image = ImageOps.exif_transpose(image)
    image = image.convert(format)
    if (size != None):
        image = image.resize(size, Image.LANCZOS)
    return image

class Gemini():
    def __init__(self, api_key_path='keys/secret.env', are_images_encoded=False, model_name="gemini-2.5-flash"):
        """Google Gemini model wrapper via Google AI Studio
        Args:
            api_key_path (str): Path to the API key file or API key string. Defaults to 'keys/secret.env'.
            are_images_encoded (bool): Whether the images are encoded in base64. Defaults to False.
            model_name (str): Gemini model name. Defaults to "gemini-2.5-flash".
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
            # Check if it's a file path or direct API key
            if os.path.isfile(api_key_path):
                self.api_key = get_api_key(api_key_path)
                self.current_key_file = api_key_path
            else:
                # Assume it's a direct API key string
                self.api_key = api_key_path
        
        if not self.api_key:
            print("API key not found.")
            exit(1)

        self.model_name = model_name
        self.use_encode = are_images_encoded

        # Initialize the Gemini API client using the new API
        self.client = genai.Client(api_key=self.api_key)

    def prepare_prompt(self, image_links: List = [], text_prompt: str = ""):
        """Prepare prompt content for Gemini API in the new format"""
        if not isinstance(image_links, list):
            image_links = [image_links]
        
        # Build contents list for the new API format
        # According to documentation, contents should be a list of parts
        contents = []
        
        # Add images first
        for image_link in image_links:
            # If it's already a PIL Image, use it directly
            if isinstance(image_link, Image.Image):
                image = image_link
                # Determine format from PIL Image
                if image.format:
                    format_lower = image.format.lower()
                    if format_lower == 'png':
                        mime_type = 'image/png'
                        save_format = 'PNG'
                    elif format_lower in ['jpeg', 'jpg']:
                        mime_type = 'image/jpeg'
                        save_format = 'JPEG'
                    elif format_lower == 'webp':
                        mime_type = 'image/webp'
                        save_format = 'WEBP'
                    else:
                        mime_type = 'image/jpeg'
                        save_format = 'JPEG'
                else:
                    mime_type = 'image/jpeg'
                    save_format = 'JPEG'
            else:
                # Otherwise, load it using the load_image function
                image = load_image(image_link)
                # Try to determine format from file extension
                if isinstance(image_link, str) and os.path.isfile(image_link):
                    ext = os.path.splitext(image_link)[1].lower()
                    if ext == '.png':
                        mime_type = 'image/png'
                        save_format = 'PNG'
                    elif ext in ['.jpg', '.jpeg']:
                        mime_type = 'image/jpeg'
                        save_format = 'JPEG'
                    elif ext == '.webp':
                        mime_type = 'image/webp'
                        save_format = 'WEBP'
                    else:
                        mime_type = 'image/jpeg'
                        save_format = 'JPEG'
                else:
                    mime_type = 'image/jpeg'
                    save_format = 'JPEG'
            
            # Convert PIL Image to bytes for the new API
            image_stream = BytesIO()
            image.save(image_stream, format=save_format)
            image_bytes = image_stream.getvalue()
            
            # Use types.Part.from_bytes() as per documentation
            contents.append(types.Part.from_bytes(
                data=image_bytes,
                mime_type=mime_type
            ))
        
        # Add text prompt after images (as recommended in documentation)
        if text_prompt:
            contents.append(text_prompt)
        
        # Return contents as a list (not types.Content object)
        return contents

    def get_parsed_output(self, prompt):
        """Get parsed output from Gemini API using the new API"""
        try:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt
            )
            return self.extract_response(response)
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower() or "quota" in error_str.lower():
                print(f"Rate limit exceeded: {error_str}")
                if self.multiple_api_keys == True:
                    new_key = pick_next_item(self.current_key_file, self.key_lists)
                    self.update_key(new_key)
                    self.current_key_file = new_key
                    print("New key is from the file: ", new_key)
                return "rate_limit_exceeded"
            else:
                print(f"Error in Gemini API: {error_str}")
                return ""
    
    def extract_response(self, response):
        """Extract text response from Gemini API response"""
        try:
            # New API returns response.text directly
            if hasattr(response, 'text') and response.text:
                return response.text
            # Fallback: try to get from candidates
            elif hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        if hasattr(part, 'text'):
                            return part.text
            return ""
        except Exception as e:
            print(f"Error extracting response: {e}")
            return ""

    def update_key(self, key, load_from_file=True):
        """Update API key"""
        if load_from_file:
            self.api_key = get_api_key(key)
        else:
            self.api_key = key
        
        # Reinitialize the client with new key
        self.client = genai.Client(api_key=self.api_key)

if __name__ == "__main__":
    # Use provided API key directly
    api_key = "AIzaSyAj3vjB9PrPDNPkAd2cSgxgkccVaZI_CCM"
    model = Gemini(api_key, model_name="gemini-2.5-flash")
    prompt = model.prepare_prompt(['https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/DiffEdit/sample_34_1.jpg', 'https://chromaica.github.io/Museum/ImagenHub_Text-Guided_IE/input/sample_34_1.jpg'], 'What is difference between two images?')
    print("prompt prepared")
    res = model.get_parsed_output(prompt)
    print("result : \n", res)


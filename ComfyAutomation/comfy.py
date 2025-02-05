
from PIL import Image
import json

def raw_image_info(filename: str) -> dict[str, str]:
    img = Image.open(filename)
    if filename[-4:] == ".png":
        info = img.load()
        return img.info
    # This should just return exif_data and parse elsewhere
    elif filename[-4:] == ".jpg":
        exif_data = img._getexif()
        return exif_data[37510].decode('utf-8').replace('\x00', '').replace('UNICODE', '')

def comfy_prompt(filename: str) -> dict[str, str]:
    return json.loads(raw_image_info(filename)['prompt'])

def comfy_prompt(filename: str) -> dict[str, str]:
    return json.loads(raw_image_info(filename)['prompt'])

def comfy_nodes(filename: str) -> dict[str, str]:
    return {id: value for (id, value) in comfy_prompt(filename).items()}

def comfy_unique_node_types(filename: str) -> dict[str, str]:
    node_types = [value['class_type'] for (_, value) in comfy_prompt(filename).items()]
    return sorted(set(node_types))



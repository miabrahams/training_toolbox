from typing import Optional
import os
from lib.comfy_analysis import ComfyImage

# Test single image

# TODO: Use sqlite dump
data_path = 'data/'
images = [img for img in os.listdir(data_path) if img[-4:] == '.png']
print(f"Processing {len(images)} images")
ImageObjects = [ComfyImage(data_path + img) for img in images]
img = [img for img in ImageObjects if 'aerys' in img.filename][0]
img.workflow['nodes']
img.prompt


# Define a type for a reader function that takes a dict (the node's JSON) and returns a value.





# Python
def extract_positive_prompt(json_data: dict, filename: str) -> str:
    """
    Attempts to extract the positive prompt string from a ComfyUI workflow JSON structure.

    The function uses a series of heuristics:
      1. Look for nodes with common types where the prompt is reliably stored (e.g. "Power Prompt (rgthree)" or "PCLazyTextEncode").
      2. If not found, search for a KSampler node and then traverse its incoming connections to find a CLIPTextEncode (or Primitive)
         node that holds the positive text.
      3. If nothing is found, raise an error including the provided filename.

    :param json_data: The JSON data loaded as a dict representing the workflow.
    :param filename: The filename (used in error messages).
    :return: The extracted positive prompt string.
    :raises Exception: If no positive prompt is found.
    """
    # Heuristic 1: Direct lookup by expected node types
    for node_id, node in json_data.items():
        node_type = node.get("class_type", "")
        if node_type in ("Power Prompt (rgthree)", "PCLazyTextEncode", "CLIPTextEncode"):
            prompt_text = node.get("inputs", {}).get("text")
            if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                return prompt_text

    # Heuristic 2: Traverse the graph starting from a KSampler node:
    # Find a node that configures the sampler and try to locate its related CLIPTextEncode or Primitive node.
    for node_id, node in json_data.items():
        node_type = node.get("class_type", "")
        # Check for a KSampler-like node (could be labeled differently in some workflows)
        if "KSampler" in node_type:
            inputs = node.get("inputs", {})
            # Try each input that looks like a reference ([node_id, index])
            for key, value in inputs.items():
                if isinstance(value, list) and len(value) >= 1:
                    ref_id = value[0]
                    if ref_id in json_data:
                        parent = json_data[ref_id]
                        parent_type = parent.get("class_type", "")
                        if "CLIPTextEncode" in parent_type or "Primitive" in parent_type:
                            prompt_text = parent.get("inputs", {}).get("text")
                            if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                                return prompt_text

    # If nothing is found, raise an error mentioning the file name.
    raise Exception(f"Positive prompt not found in file {filename}")


# Example usage:
if __name__ == "__main__":
    import json

    # Assume we have imported a JSON workflow into 'workflow_data'
    # For example:
    # with open("path/to/workflow.json", "r", encoding="utf-8") as f:
    #     workflow_data = json.load(f)
    # Here we simulate with a sample dict:
    workflow_data = {
        "340": {
            "inputs": {"prompt": "<lora:G3NSHIN IL.safetensors:1> additional text"},
            "class_type": "Power Prompt (rgthree)"
        },
        "445": {
            "inputs": {"steps_total": 28, "sampler_name": "euler_smea_dy"},
            "class_type": "KSampler Config (rgthree)"
        },
        "172": {
            "inputs": {"text": "This is an example prompt from CLIPTextEncode node"},
            "class_type": "CLIPTextEncode"
        }
    }

    try:
        result = extract_positive_prompt(workflow_data, "example_workflow.json")
        print("Extracted positive prompt:", result)
    except Exception as e:
        print("Error:", e)




print("Positive Node value:", positive_node.read_value(example_node_data))
print("LoRAs Node value:", loras_node.read_value(example_node_data))
print("Seed Node value:", seed_node.read_value(example_node_data))
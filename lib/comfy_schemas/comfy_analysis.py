from lib.metadata import read_comfy_metadata
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple
from copy import deepcopy
from pathlib import Path
import re


# [print(nt) for nt in comfy_unique_node_types(FILE_PATH)]
PROMPT_NODE_TYPES = {
    'DPCombinatorialGenerator': 'text',
    'CLIPTextEncode': 'text',
    'Power Prompt (rgthree)': 'prompt',
    'PCLazyTextEncode': 'text'
}

def scrape_node(node):
    class_type = node['class_type']
    if class_type in PROMPT_NODE_TYPES:
        return node['inputs'][PROMPT_NODE_TYPES[class_type]]
    else:
        raise Exception('Cannot parse node: ' + str(node))

def fileToComfyImage(filename: Path) -> 'ComfyImage':
    prompt, workflow = read_comfy_metadata(filename)
    return ComfyImage(filename, prompt, workflow)

class ComfyImage:
    def __init__(self, filename: Path, prompt: dict, workflow: dict):
        self.prompt = prompt
        self.workflow = workflow
        self.filename = filename
    def nodes(self):
        return {id: value for (id, value) in self.prompt.items()}
    def unique_nodes(self):
        node_types = [value['class_type'] for (_, value) in self.prompt.items()]
        return sorted(set(node_types))
    def text_nodes(self):
        return [value for (_, value) in self.prompt if value['class_type'] in PROMPT_NODE_TYPES]
    def text_values(self):
        return map(scrape_node, self.text_nodes())


ReaderFunc = Callable[[Dict[str, Any]], Any]

@dataclass
class SchemaNode:
    role: str
    node_id: str
    input_name: str
    node_type: str
    reader: ReaderFunc = field(default=lambda node: node.get('inputs', {}).get('default'))

    def read_value(self, node_data: Dict[str, Any]) -> Any:
        """
        Applies the reader function to the given node's data.
        If input_name is provided and no custom reader is set,
        the default reader returns node_data["inputs"][input_name].
        """
        # If a custom input_name is provided and the default lambda is in use,
        # create a default reader function on the fly.
        if self.reader == (lambda node: node.get('inputs', {}).get('default')) and self.input_name:
            return node_data.get('inputs', {}).get(self.input_name)
        return self.reader(node_data)

@dataclass
class Schema:
    version: str
    checkpoint_node: SchemaNode
    seed_node: SchemaNode
    positive_node: SchemaNode
    negative_node: SchemaNode
    loras_node: SchemaNode
    steps_node: SchemaNode
    sampler_node: SchemaNode
    scheduler_node: SchemaNode
    aspect_ratio_node: SchemaNode
    swap_dimensions_node: SchemaNode





# SPECIFIC STUFF HERE

# Schema 3 (most recent)
Schema3 = Schema(
    version="3.0",
    checkpoint_node=SchemaNode(
        role="checkpoint",
        node_id="474",
        input_name="ckpt_name",
        node_type="Checkpoint Loader",
    ),
    seed_node=SchemaNode(
        role="seed",
        node_id="463",
        input_name="seed",
        node_type="CR Seed",
    ),
    positive_node=SchemaNode(
        role="positive",
        node_id="553",
        input_name="text",
        node_type="PCLazyTextEncode",
    ),
    negative_node=SchemaNode(
        role="negative",
        node_id="448",
        input_name="customtext",
        node_type="CR Prompt Text",
    ),
    loras_node=SchemaNode(
        role="loras",
        node_id="340",
        input_name="prompt",
        node_type="Power Prompt (rgthree)",
    ),
    steps_node=SchemaNode(
        role="steps",
        node_id="445",
        input_name="steps_total",
        node_type="KSampler Config (rgthree)",
    ),
    sampler_node=SchemaNode(
        role="sampler",
        node_id="445",
        input_name="sampler_name",
        node_type="KSampler Config (rgthree)",
    ),
    scheduler_node=SchemaNode(
        role="scheduler",
        node_id="445",
        input_name="scheduler",
        node_type="KSampler Config (rgthree)",
    ),
    aspect_ratio_node=SchemaNode(
        role="aspect_ratio",
        node_id="346",
        input_name="aspect_ratio",
        node_type="CR SDXL Aspect Ratio",
    ),
    swap_dimensions_node=SchemaNode(
        role="swap_dimensions",
        node_id="346",
        input_name="swap_dimensions",
        node_type="CR SDXL Aspect Ratio",
    ),
)



def split_loras(node_data: Dict[str, Any]) -> Tuple[str, str]:
    """
    Separates LoRAs from text 'prompt' field and returns a tuple:
    """
    prompt = node_data.get('inputs', {}).get('prompt', '')

    # Find all LoRA patterns in the prompt.
    lora_matches = re.findall(r"<lora:[^>]+>", prompt)

    # Remove all LoRA patterns from the prompt.
    cleaned_prompt = re.sub(r"<lora:[^>]+>", "", prompt)

    # Normalize whitespace.
    cleaned_prompt = " ".join(cleaned_prompt.split())
    lora_string = " ".join(lora_matches)
    return cleaned_prompt, lora_string

loras_node_gen2 = SchemaNode(
    role="loras",
    node_id="340",
    input_name="prompt",
    node_type="Power Prompt (rgthree)",
    reader=lambda node: split_loras(node)[1],
)
prompt_node_gen2 = SchemaNode(
    role="positive",
    node_id="340",
    input_name="prompt",
    node_type="Power Prompt (rgthree)",
    reader=lambda node: split_loras(node)[0],
)


# Create a deep copy of Schema3 and modify it for Schema2
Schema2 = deepcopy(Schema3)
Schema2.version = "2.0"
Schema2.loras_node = loras_node_gen2
Schema2.positive_node = prompt_node_gen2



#TODO: Identify when PromptToSchedule nodes are used by looking at switch input
prompt_schedule_gen2 = SchemaNode(
    role="prompt",
    node_id="445",
    input_name="text",
    node_type="PromptToSchedule",
)




# Some old workflows used Combinatorial Prompts
prompt_schedule_gen05 = SchemaNode(
    role="prompt",
    node_id="89",
    input_name="text",
    node_type="DPCombinatorialGenerator",
)


# ClipTextEncode
prompt_schedule_gen04 = SchemaNode(
    role="prompt",
    node_id="98",
    input_name="text",
    node_type="CLIPTextEncode",
)



# One step up parent
def get_parent(prompt_graph: dict, node: dict, input_type: Optional[str]) -> Optional[dict]:
    inputs = node.get("inputs", {})
    # Check each input
    for input_name, input_value in inputs.items():
        if input_type and input_type != input_name:
            continue
        if isinstance(input_value, list) and len(input_value) >= 1:
            ref_id = input_value[0]
            if ref_id in prompt_graph:
                return prompt_graph[ref_id]
    return None



# 13054 / 13145 (99.3%) of images found prompt
def extract_positive_prompt(prompt_dict: dict) -> str:
    """
    The function uses a series of heuristics
      1. Look for nodes with common types where the prompt is reliably stored
      2. If not found, search for a KSampler node and then traverse its incoming connections to find a CLIPTextEncode (or Primitive)
         node that holds the positive text.
      3. If nothing is found, raise an error including the provided filename.

    :param json_data: The JSON data loaded as a dict representing the workflow.
    :param filename: The filename (used in error messages).
    :return: The extracted positive prompt string.
    :raises Exception: If no positive prompt is found.
    """
    # Heuristic 1: Direct lookup by expected node types

    for schema in [Schema3.positive_node, Schema2.positive_node, prompt_schedule_gen05, prompt_schedule_gen04]:
        node = prompt_dict.get(schema.node_id, {})
        if node.get("class_type", "") == schema.node_type:
            prompt_text = node.get("inputs", {}).get(schema.input_name)
            if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                return prompt_text

    # Heuristic 2: Traverse the graph starting from a KSampler node:
    ksamplers = []
    for _, node in prompt_dict.items():
        class_type = node.get("class_type", "")
        if class_type in ["KSampler", "CFGGuider"] or "KSamplerAdvanced" in class_type:
            ksamplers.append(node)

    for node in ksamplers:
        input_node = get_parent(prompt_dict, node, "positive")
        if not input_node:
            continue
        # Simple CLIPTextEncode node
        if "CLIPTextEncode" in input_node.get("class_type", ""):
            prompt_text = input_node.get("inputs", {}).get("text")
            if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                return prompt_text
        # ConditioningConcat (get both parent inputs)
        if "ConditioningConcat" in input_node.get("class_type", ""):
            prompt_text = []
            for input_name in ["conditioning_to", "conditioning_from"]:
                cond_input = get_parent(prompt_dict, input_node, input_name)
                if cond_input:
                    prompt_text.append(cond_input.get("inputs", {}).get("text"))
            if all(isinstance(text, str) and text.strip() for text in prompt_text):
                return " ".join(prompt_text)


    # If nothing is found, raise an error mentioning the file name.
    raise Exception("Positive prompt not found")


def extract_negative_prompt(prompt_dict: dict) -> str:
    """
    Extract the negative prompt from ComfyUI workflow data.
    Similar to extract_positive_prompt but looks for negative conditioning nodes.

    :param prompt_dict: The prompt dictionary from ComfyUI workflow.
    :return: The extracted negative prompt string.
    :raises Exception: If no negative prompt is found.
    """
    # Heuristic 1: Direct lookup by expected negative node types
    for schema in [Schema3.negative_node, Schema2.negative_node]:
        node = prompt_dict.get(schema.node_id, {})
        if node.get("class_type", "") == schema.node_type:
            prompt_text = node.get("inputs", {}).get(schema.input_name)
            if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                return prompt_text

    # Heuristic 2: Traverse the graph starting from a KSampler node:
    ksamplers = []
    for _, node in prompt_dict.items():
        class_type = node.get("class_type", "")
        if class_type in ["KSampler", "CFGGuider"] or "KSamplerAdvanced" in class_type:
            ksamplers.append(node)

    for node in ksamplers:
        input_node = get_parent(prompt_dict, node, "negative")
        if not input_node:
            continue
        # Simple CLIPTextEncode node
        if "CLIPTextEncode" in input_node.get("class_type", ""):
            prompt_text = input_node.get("inputs", {}).get("text")
            if prompt_text and isinstance(prompt_text, str) and prompt_text.strip():
                return prompt_text
        # ConditioningConcat (get both parent inputs)
        if "ConditioningConcat" in input_node.get("class_type", ""):
            prompt_text = []
            for input_name in ["conditioning_to", "conditioning_from"]:
                cond_input = get_parent(prompt_dict, input_node, input_name)
                if cond_input:
                    prompt_text.append(cond_input.get("inputs", {}).get("text"))
            if all(isinstance(text, str) and text.strip() for text in prompt_text):
                return " ".join(prompt_text)

    # If nothing is found, return empty string (negative prompts are optional)
    return ""
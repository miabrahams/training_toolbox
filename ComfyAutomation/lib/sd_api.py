
import requests
from typing import List

API = 'http://127.0.0.1:7860/sdapi/v1'

def get_models():
    return requests.get(url=f'{API}/sd-models').json()

def get_options():
    return requests.get(url=f'{API}/options').json()


def get_model_names():
    models = get_models()
    return [m['model_name'] for m in models]


def find_model(name: str, model_names: List[str]) -> str:
    if name in model_names:
        found_model = name
    else:
        import difflib
        def str_simularity(a, b):
            return difflib.SequenceMatcher(None, a, b).ratio()
        max_sim = 0.0
        max_model = model_names[0]
        for model in model_names:
            sim = str_simularity(name, model)
            if sim >= max_sim:
                max_sim = sim
                max_model = model
        found_model = max_model
    return found_model


def get_extra_networks():
    return requests.get(url=f'{API}/extra-networks').json()
.PHONY: analysis clean

load_prompts:
	cd golang && go run cmd/load/main.go -db ../data/prompts.sqlite -dir ~/ComfyUI/output

analysis:
	python tagging/comfy_mass_analysis.py -db data/prompts.sqlite -dir ~/ComfyUI/output

analysis_screen:
	python tagging/comfy_mass_analysis.py --screen-dir /home/abrahams/ComfyUI/output/NoobAI-XL-eps-v1.1

clean:
	rm -rf .pytest_cache build dist *.egg-info __pycache__ .mypy_cache .ruff_cache *.pyc

install:
	uv sync

api:
	uvicorn api:app --reload --host 127.0.0.1 --port 8000
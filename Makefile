.PHONY: analysis clean

load_prompts:
	cd golang && go run cmd/load/load.go -db ../data/prompts.sqlite -config ../config/config.yml

analysis:
	python tagging/comfy_mass_analysis.py -db data/prompts.sqlite -dir ~/ComfyUI/output

analysis_screen:
	python tagging/comfy_mass_analysis.py --screen-dir /home/abrahams/ComfyUI/output/NoobAI-XL-eps-v1.1

clean:
	rm -rf .pytest_cache build dist *.egg-info __pycache__ .mypy_cache .ruff_cache *.pyc

install:
	uv sync

# Example of testing the schema extractor
test_extractor:
	python -m src.lib.comfy_schemas.test_extractor test_schema_v5.png schema_v5.yml

api:
	uvicorn api:app --reload --host 127.0.0.1 --port 8000
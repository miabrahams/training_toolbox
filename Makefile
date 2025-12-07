
.PHONY: load_prompts
load_prompts:
	cd golang && go run cmd/load/load.go -db ../data/prompts.sqlite -config ../config/config.yml

.PHONY: export_prompts
export_prompts:
	python cli.py export_prompt_fields --out prompt_fields.sqlite

.PHONY: analysis
analysis:
	python tagging/comfy_mass_analysis.py -db data/prompts.sqlite -dir ~/ComfyUI/output

.PHONY: analysis_screen
analysis_screen:
	python tagging/comfy_mass_analysis.py --screen-dir /home/abrahams/ComfyUI/output/NoobAI-XL-eps-v1.1

.PHONY: clean
clean:
	rm -rf .pytest_cache build dist *.egg-info __pycache__ .mypy_cache .ruff_cache *.pyc

.PHONY: install
install:
	uv sync

# Example of testing the schema extractor
.PHONY: test_extractor
test_extractor:
	python -m src.lib.comfy_schemas.test_extractor test_schema_v5.png schema_v5.yml

.PHONY: api
api:
	uvicorn api:app --reload --host 127.0.0.1 --port 8000
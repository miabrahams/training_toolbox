.PHONY: analysis

analysis:
	python tagging/comfy_mass_analysis.py -db data/prompts.sqlite -dir ~/ComfyUI/output

analysis_screen:
	python tagging/comfy_mass_analysis.py --screen-dir /home/abrahams/ComfyUI/output/NoobAI-XL-eps-v1.1
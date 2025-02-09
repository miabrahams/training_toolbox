


python diffusers/scripts/convert_original_stable_diffusion_to_diffusers.py \
    --checkpoint_path  /mnt/d/AI/automatic/models/Stable-diffusion/ponyDiffusionV6XL_v6.safetensors \
    --dump_path converted_checkpoints/ponyDiffusionv6XL.diffusers \
    --from_safetensors \
    --pipeline_class_name StableDiffusionXLPipeline
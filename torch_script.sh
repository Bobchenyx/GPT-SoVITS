# export PYTHONPATH="${PYTHONPATH}:/Users/bobchenyx/Downloads/moyoyo/GPT-SoVITS:/Users/bobchenyx/Downloads/moyoyo/GPT-SoVITS/tools:/Users/bobchenyx/Downloads/moyoyo/GPT-SoVITS/GPT_SoVITS"

python GPT_SoVITS/export_torch_script.py \
    --gpt_model doubao/txdb-e15.ckpt \
    --sovits_model doubao/txdb_e12_s204.pth \
    --ref_audio doubao/doubao-ref-ours.wav \
    --ref_text doubao/doubao-ref.txt \
    --output_path doubao-export \
    --export_common_model \
    --device cpu \
    --no-half

# CUDA_VISIBLE_DEVICES=0 python GPT_SoVITS/export_torch_script.py \
#     --gpt_model doubao/txdb-e15.ckpt \
#     --sovits_model doubao/txdb_e12_s204.pth \
#     --ref_audio doubao/doubao-ref-ours.wav \
#     --ref_text doubao/doubao-ref.txt \
#     --output_path doubao-export-gpu \
#     --export_common_model \
#     --device cuda \
#     --no-half
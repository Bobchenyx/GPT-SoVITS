python GPT_SoVITS/export_torch_script.py \
    --gpt_model doubao/txdb-e15.ckpt \
    --sovits_model doubao/txdb_e12_s204.pth \
    --ref_audio doubao/doubao-ref-ours.wav \
    --ref_text doubao/doubao-ref.txt \
    --output_path doubao-export \
    --export_common_model \
    --device cpu \
    --no-half


python GPT_SoVITS/test_torch_script.py \
    --model_dir doubao-export \
    --ref_audio doubao/doubao-ref-ours.wav \
    --ref_text "我叫豆包呀，能陪你聊天解闷，不管是聊生活趣事，知识科普还是帮你出主意，我都在行哦。" \
    --output doubao-export/generated_audio-new.wav \
    --device cpu \
    --language zh \
    --version v2 \
    --target_text "你好呀，我叫豆包,是你的语音助理. 可以帮你解答任何你想知道的问题"

# CUDA_VISIBLE_DEVICES=0 python GPT_SoVITS/test_torch_script.py \
#     --model_dir doubao-export-gpu \
#     --ref_audio doubao/doubao-ref-ours.wav \
#     --ref_text "我叫豆包呀，能陪你聊天解闷，不管是聊生活趣事，知识科普还是帮你出主意，我都在行哦。" \
#     --output doubao-export-gpu/generated_audio-new.wav \
#     --device cuda \
#     --language zh \
#     --version v2 \
#     --target_text "你好呀，我叫豆包,是你的语音助理. 可以帮你解答任何你想知道的问题"


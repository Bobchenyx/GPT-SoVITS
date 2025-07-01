import torch
import torchaudio
import soundfile as sf
import numpy as np
from transformers import AutoTokenizer, AutoModelForMaskedLM
import argparse
import os
import sys
import re
import itertools
import time

# 添加GPT-SoVITS路径以便导入模块
def setup_gpt_sovits_path():
    """设置GPT-SoVITS模块路径"""
    current_dir = os.getcwd()
    possible_paths = [
        current_dir,  # 当前目录
        os.path.join(current_dir, "GPT_SoVITS"),  # 当前目录下的GPT-SoVITS文件夹
        os.path.dirname(current_dir),  # 上级目录
        os.path.join(os.path.dirname(current_dir), "GPT_SoVITS"),  # 上级目录下的GPT-SoVITS
    ]
    
    for path in possible_paths:
        if os.path.exists(os.path.join(path, "text")) and os.path.exists(os.path.join(path, "text", "cleaner.py")):
            if path not in sys.path:
                sys.path.insert(0, path)
            print(f"Added GPT_SoVITS path: {path}")
            return True
    
    return False

# 尝试导入GPT-SoVITS模块
def import_gpt_sovits_modules():
    """导入GPT-SoVITS相关模块"""
    try:
        from text.LangSegmenter import LangSegmenter
        from text.cleaner import clean_text
        from text import cleaned_text_to_sequence
        return LangSegmenter, clean_text, cleaned_text_to_sequence, True
    except ImportError as e:
        print(f"Warning: Could not import GPT-SoVITS modules: {e}")
        return None, None, None, False

def load_audio(audio_path, target_sr=16000):
    """加载音频文件并重采样到目标采样率"""
    audio, sr = torchaudio.load(audio_path)
    if sr != target_sr:
        audio = torchaudio.functional.resample(audio, sr, target_sr)
    return audio.squeeze(0).numpy()

def get_bert_feature_with_exported_model(text, word2ph, bert_model, tokenizer, device):
    """使用导出的BERT模型获取特征"""
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt").to(device)
        word2ph_tensor = torch.tensor(word2ph, dtype=torch.int32, device=device)

        return bert_model(
            inputs["input_ids"],
            inputs["attention_mask"],
            inputs["token_type_ids"],
            word2ph_tensor
        )

def get_bert_inf_with_exported_model(phones, word2ph, norm_text, language, bert_model, tokenizer, device, is_half=False):
    """使用导出的BERT模型获取推理结果"""
    language = language.replace("all_", "")
    if language == "zh":
        bert = get_bert_feature_with_exported_model(norm_text, word2ph, bert_model, tokenizer, device)
    else:
        bert = torch.zeros(
            (1024, len(phones)),
            dtype=torch.float16 if is_half else torch.float32,
        ).to(device)
    return bert

def clean_text_inf(text, language, version, clean_text_func, cleaned_text_to_sequence_func):
    """文本清理函数 - 从原版inference_webui.py复制"""
    language = language.replace("all_", "")
    phones, word2ph, norm_text = clean_text_func(text, language, version)
    phones = cleaned_text_to_sequence_func(phones, version)
    return phones, word2ph, norm_text

def get_phones_and_bert_with_exported_models(text, language, version, LangSegmenter, clean_text_func, 
                                            cleaned_text_to_sequence_func, tokenizer, bert_model, device, 
                                            is_half=False, final=False):
    """使用导出的BERT模型和tokenizer的get_phones_and_bert函数"""
    dtype = torch.float16 if is_half else torch.float32
    
    text = re.sub(r' {2,}', ' ', text)
    textlist = []
    langlist = []
    
    if language == "all_zh":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_yue":
        for tmp in LangSegmenter.getTexts(text, "zh"):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ja":
        for tmp in LangSegmenter.getTexts(text, "ja"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "all_ko":
        for tmp in LangSegmenter.getTexts(text, "ko"):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "en":
        langlist.append("en")
        textlist.append(text)
    elif language == "auto":
        for tmp in LangSegmenter.getTexts(text):
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    elif language == "auto_yue":
        for tmp in LangSegmenter.getTexts(text):
            if tmp["lang"] == "zh":
                tmp["lang"] = "yue"
            langlist.append(tmp["lang"])
            textlist.append(tmp["text"])
    else:
        # 处理 "zh" 这样的简化语言代码
        if language == "zh":
            language = "all_zh"
            for tmp in LangSegmenter.getTexts(text, "zh"):
                langlist.append(tmp["lang"])
                textlist.append(tmp["text"])
        else:
            for tmp in LangSegmenter.getTexts(text):
                if langlist:
                    if (tmp["lang"] == "en" and langlist[-1] == "en") or (tmp["lang"] != "en" and langlist[-1] != "en"):
                        textlist[-1] += tmp["text"]
                        continue
                if tmp["lang"] == "en":
                    langlist.append(tmp["lang"])
                else:
                    langlist.append(language)
                textlist.append(tmp["text"])
    
    print(f"Text segments: {textlist}")
    print(f"Language segments: {langlist}")
    
    phones_list = []
    bert_list = []
    norm_text_list = []
    
    for lang, txt in zip(langlist, textlist):
        phones, word2ph, norm_text = clean_text_inf(txt, lang, version, clean_text_func, cleaned_text_to_sequence_func)
        bert = get_bert_inf_with_exported_model(phones, word2ph, norm_text, lang, bert_model, tokenizer, device,is_half)
        phones_list.append(phones)
        norm_text_list.append(norm_text)
        bert_list.append(bert)
    
    bert = torch.cat(bert_list, dim=1)
    phones = list(itertools.chain.from_iterable(phones_list))
    norm_text = "".join(norm_text_list)

    if not final and len(phones) < 6:
        return get_phones_and_bert_with_exported_models("." + text, language, version, LangSegmenter, 
                                                       clean_text_func, cleaned_text_to_sequence_func, 
                                                       tokenizer, bert_model, device, is_half, final=True)

    return phones, bert.to(dtype), norm_text

class GPTSoVITSInference:
    def __init__(self, model_dir, device='cpu', is_half=False):
        self.device = device
        self.model_dir = model_dir
        self.is_half = is_half
        
        print("Setting up GPT-SoVITS environment...")
        
        # 设置路径并导入模块
        if not setup_gpt_sovits_path():
            print("Warning: Could not find GPT-SoVITS modules path")
        
        self.LangSegmenter, self.clean_text_func, self.cleaned_text_to_sequence_func, self.has_modules = import_gpt_sovits_modules()
        
        if not self.has_modules:
            print("Error: GPT-SoVITS modules not available. Please run this script from GPT-SoVITS directory or add GPT-SoVITS to Python path.")
            raise ImportError("Required GPT-SoVITS modules not found")
        
        print("Loading exported models...")
        
        # 1. 加载导出的SSL模型
        self.ssl_model = torch.jit.load(
            os.path.join(model_dir, "ssl_model.pt"), 
            map_location=device
        )
        self.ssl_model.eval()
        print("✓ SSL model loaded")
        
        # 2. 加载导出的GPT-SoVITS模型
        self.gpt_sovits_model = torch.jit.load(
            os.path.join(model_dir, "gpt_sovits_model.pt"), 
            map_location=device
        )
        self.gpt_sovits_model.eval()
        print("✓ GPT-SoVITS model loaded")
        
        # 3. 加载导出的tokenizer
        tokenizer_path = os.path.join(model_dir, "tokenizer")
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✓ Tokenizer loaded from: {tokenizer_path}")
        
        # 4. 加载导出的BERT模型
        self.bert_model = torch.jit.load(
            os.path.join(model_dir, "bert_model.pt"), 
            map_location=device
        )
        self.bert_model.eval()
        print("✓ Exported BERT model loaded")
        
        print("All models loaded successfully!")
    
    def inference(self, ref_audio_path, ref_text, target_text, output_path, 
                 top_k=5, language="auto", version="v2"):
        """
        使用原版文本处理方法进行语音合成推理
        """
        print(f"Processing reference audio: {ref_audio_path}")
        print(f"Reference text: {ref_text}")
        print(f"Target text: {target_text}")
        print(f"Language: {language}, Version: {version}")
        
        with torch.no_grad():
            # 1. 加载参考音频
            print("Loading reference audio...")
            # ref_audio = torch.tensor([load_audio(ref_audio_path, 16000)]).float().to(self.device)
            ref_audio_np = load_audio(ref_audio_path, 16000)
            ref_audio = torch.from_numpy(ref_audio_np).float().unsqueeze(0).to(self.device)
            print(f"Reference audio shape: {ref_audio.shape}")
            
            # 2. 使用原版音素处理 + 导出的BERT模型处理参考文本
            print("Processing reference text with original phoneme processing + exported BERT model...")
            ref_phones, ref_bert_features, ref_norm_text = get_phones_and_bert_with_exported_models(
                ref_text, language, version, 
                self.LangSegmenter, self.clean_text_func, self.cleaned_text_to_sequence_func,
                self.tokenizer, self.bert_model, self.device, self.is_half
            )
            print(f"Reference - Phones: {len(ref_phones)}, BERT: {ref_bert_features.shape}, Norm text: {ref_norm_text}")

            start = time.time()
            
            # 3. 使用原版音素处理 + 导出的BERT模型处理目标文本
            print("Processing target text with original phoneme processing + exported BERT model...")
            target_phones, target_bert_features, target_norm_text = get_phones_and_bert_with_exported_models(
                target_text, language, version,
                self.LangSegmenter, self.clean_text_func, self.cleaned_text_to_sequence_func,
                self.tokenizer, self.bert_model, self.device, self.is_half
            )
            print(f"Target - Phones: {len(target_phones)}, BERT: {target_bert_features.shape}, Norm text: {target_norm_text}")
            
            # 4. 准备序列输入
            # ref_seq = torch.LongTensor([ref_phones]).to(self.device)
            # target_seq = torch.LongTensor([target_phones]).to(self.device)
            ref_seq = torch.tensor(ref_phones, dtype=torch.long).unsqueeze(0).to(self.device)
            target_seq = torch.tensor(target_phones, dtype=torch.long).unsqueeze(0).to(self.device)
            
            # 5. 使用导出的SSL模型提取特征
            print("Extracting SSL features...")
            ssl_content = self.ssl_model(ref_audio)
            print(f"SSL content shape: {ssl_content.shape}")
            
            # 6. 重采样参考音频到32kHz
            print("Resampling reference audio...")
            ref_audio_32k = self.ssl_model.resample(ref_audio, 16000, 32000)
            print(f"Resampled audio shape: {ref_audio_32k.shape}")
            
            # 7. 设置采样参数
            top_k_tensor = torch.tensor([top_k], dtype=torch.long).to(self.device)
            
            # 8. 使用导出的GPT-SoVITS模型进行推理
            print("Running GPT-SoVITS inference...")
            # print("Input tensor information:")
            # print(f"  ssl_content: {ssl_content.shape} {ssl_content.dtype}")
            # print(f"  ref_audio_32k: {ref_audio_32k.shape} {ref_audio_32k.dtype}")
            # print(f"  ref_seq: {ref_seq.shape} {ref_seq.dtype}")
            # print(f"  target_seq: {target_seq.shape} {target_seq.dtype}")
            # print(f"  ref_bert: {ref_bert_features.shape} {ref_bert_features.dtype}")
            # print(f"  target_bert: {target_bert_features.shape} {target_bert_features.dtype}")
            # print(f"  top_k: {top_k_tensor.shape} {top_k_tensor.dtype}")
            
            generated_audio = self.gpt_sovits_model(
                ssl_content,
                ref_audio_32k, 
                ref_seq,
                target_seq,
                ref_bert_features,
                target_bert_features,
                top_k_tensor
            )

            end = time.time()
            print("生成耗时:", end - start, "秒")
            
            # print(f"Generated audio shape: {generated_audio.shape}")
            
            # 9. 保存结果
            audio_numpy = generated_audio.squeeze().cpu().numpy()
            
            # 音频后处理
            max_val = np.abs(audio_numpy).max()
            if max_val > 1.0:
                audio_numpy = audio_numpy / max_val
            
            sf.write(output_path, audio_numpy, 32000)
            print(f"Audio saved to: {output_path}")
            print(f"Audio duration: {len(audio_numpy)/32000:.2f} seconds")
            
            return audio_numpy

def main():
    parser = argparse.ArgumentParser(description="GPT-SoVITS Inference with Original Text Processing")
    parser.add_argument("--model_dir", required=True, help="Directory containing exported models")
    parser.add_argument("--ref_audio", required=True, help="Reference audio file path")
    parser.add_argument("--ref_text", required=True, help="Reference text")
    parser.add_argument("--target_text", required=True, help="Target text to synthesize")
    parser.add_argument("--output", required=True, help="Output audio file path")
    parser.add_argument("--device", default="cpu", help="Device to use (cpu/cuda)")
    parser.add_argument("--language", default="auto", help="Language setting (auto/all_zh/zh/en/etc.)")
    parser.add_argument("--version", default="v2", help="Model version (v1/v2)")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k sampling parameter")
    parser.add_argument("--no-half", action="store_true", help="Do not use half precision")
    
    args = parser.parse_args()
    
    # 检查导出的模型文件是否存在
    required_files = ["ssl_model.pt", "gpt_sovits_model.pt", "tokenizer", "bert_model.pt"]
    missing_files = []
    
    for file in required_files:
        file_path = os.path.join(args.model_dir, file)
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Error: Required files not found:")
        for file in missing_files:
            print(f"  - {file}")
        return
    
    print("All required model files found!")
    print(f"Model directory: {args.model_dir}")
    print(f"Device: {args.device}")
    print(f"Language: {args.language}")
    print(f"Version: {args.version}")
    print(f"Top-k: {args.top_k}")
    
    # 确定是否使用半精度
    is_half = not args.no_half and torch.cuda.is_available() and args.device == "cuda"
    print(f"Using half precision: {is_half}")
    print("-" * 50)
    
    # 初始化推理器
    try:
        inference = GPTSoVITSInference(args.model_dir, args.device, is_half)
    except Exception as e:
        print(f"Failed to initialize inference model: {e}")
        print("\nMake sure you are running this script from the GPT-SoVITS directory")
        print("Or add GPT-SoVITS to your Python path")
        return
    
    # 进行推理
    try:
        inference.inference(
            ref_audio_path=args.ref_audio,
            ref_text=args.ref_text, 
            target_text=args.target_text,
            output_path=args.output,
            top_k=args.top_k,
            language=args.language,
            version=args.version
        )
        print("=" * 50)
        print("Inference completed successfully!")
        
    except Exception as e:
        print(f"Inference failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
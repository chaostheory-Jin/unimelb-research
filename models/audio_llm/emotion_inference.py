#!/usr/bin/env python3
# emotion_inference.py - 使用微调后的LLM进行情感识别 (针对 Apple Silicon M4 优化)

import os
import json
import argparse
import sys
import platform

# 检查是否为 Apple Silicon
is_apple_silicon = platform.system() == "Darwin" and platform.machine().startswith(("arm", "aarch"))

# 导入必要的库
import torch
import numpy as np
import librosa
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from tqdm import tqdm

# 确定默认设备
if is_apple_silicon:
    print(f"Detected Apple Silicon: {platform.machine()}")
    if torch.backends.mps.is_available():
        print("MPS (Metal Performance Shaders) is available")
        default_device = "mps"
    else:
        print("Warning: MPS is not available, falling back to CPU")
        default_device = "cpu"
else:
    default_device = "cuda" if torch.cuda.is_available() else "cpu"

# 模型和数据路径
DATASET_DIR = "/Users/jinhongyu/Documents/GitHub/unimelb-research/dataset"
MODEL_DIR = os.path.join(DATASET_DIR, "emotion_llm_model")
TEST_DATA_PATH = os.path.join(DATASET_DIR, "llm_emotion_data", "crema_d_test.json")

def extract_audio_features(audio_path):
    """从音频文件中提取特征"""
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        
        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # 计算统计量
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        
        # 创建特征描述
        features_desc = "Audio features: "
        for i, (mean, std) in enumerate(zip(mfccs_mean, mfccs_std)):
            features_desc += f"MFCC{i+1}(mean={mean:.2f}, std={std:.2f}) "
        
        return features_desc
    except Exception as e:
        print(f"Error extracting features from {audio_path}: {e}")
        return "Audio features could not be extracted."

def create_prompt(audio_path, instruction="Identify the emotion in this audio clip."):
    """创建用于情感识别的提示"""
    # 提取音频特征
    features_desc = extract_audio_features(audio_path)
    
    # 创建提示
    prompt = f"### Instruction:\n{instruction}\n\n"
    prompt += f"### Input:\nThis is an audio clip of a person speaking. "
    prompt += f"The audio has the following characteristics: {features_desc}\n\n"
    prompt += "### Response:\n"
    
    return prompt

def load_model(model_path, base_model_name=None, device=default_device):
    """加载微调后的模型"""
    # 检查是否是LoRA模型
    is_lora = os.path.exists(os.path.join(model_path, "adapter_config.json"))
    
    try:
        if is_lora and base_model_name:
            print(f"Loading LoRA model from {model_path} with base model {base_model_name}")
            # 加载基础模型
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                low_cpu_mem_usage=True
            )
            # 加载LoRA适配器
            model = PeftModel.from_pretrained(model, model_path)
        else:
            print(f"Loading full model from {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if device != "cpu" else torch.float32,
                low_cpu_mem_usage=True
            )
        
        # 针对 Apple Silicon 的特殊处理
        if device == "mps" and is_apple_silicon:
            model = model.to("mps")
        elif device != "auto":
            model = model.to(device)
        
        # 加载分词器
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def predict_emotion(model, tokenizer, audio_path, device=default_device):
    """预测音频的情感"""
    # 创建提示
    prompt = create_prompt(audio_path)
    
    # 标记化
    inputs = tokenizer(prompt, return_tensors="pt")
    
    # 将输入移动到正确的设备
    if device != "auto":
        inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 生成回答
    with torch.no_grad():
        outputs = model.generate(
            inputs["input_ids"],
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True
        )
    
    # 解码回答
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # 提取情感预测
    response = response.replace(prompt, "")
    
    return response

def evaluate_model(model, tokenizer, test_data_path, device=default_device, limit=None):
    """评估模型在测试集上的性能"""
    # 加载测试数据
    try:
        with open(test_data_path, 'r') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"Error loading test data: {e}")
        return
    
    if limit and limit > 0:
        test_data = test_data[:limit]
    
    print(f"Evaluating model on {len(test_data)} test samples...")
    
    correct = 0
    total = 0
    results = []
    
    for sample in tqdm(test_data):
        # 获取音频路径
        file_id = sample['metadata']['file_id']
        audio_path = os.path.join(DATASET_DIR, "CREMA-D", "AudioWAV", f"{file_id}.wav")
        
        if os.path.exists(audio_path):
            # 预测情感
            prediction = predict_emotion(model, tokenizer, audio_path, device)
            
            # 获取真实情感
            true_emotion = sample['emotion']
            
            # 检查预测是否正确
            is_correct = true_emotion.lower() in prediction.lower()
            if is_correct:
                correct += 1
            
            total += 1
            
            # 保存结果
            results.append({
                "file_id": file_id,
                "true_emotion": true_emotion,
                "prediction": prediction,
                "correct": is_correct
            })
            
            # 打印进度
            if total % 10 == 0:
                print(f"Processed {total}/{len(test_data)} samples. Current accuracy: {correct/total:.4f}")
    
    # 计算准确率
    accuracy = correct / total if total > 0 else 0
    print(f"\nFinal accuracy: {accuracy:.4f} ({correct}/{total})")
    
    # 保存结果
    results_path = os.path.join(os.path.dirname(model_path), "evaluation_results.json")
    with open(results_path, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "results": results
        }, f, indent=2)
    
    print(f"Results saved to {results_path}")
    
    return accuracy

def main():
    parser = argparse.ArgumentParser(description="Emotion recognition with fine-tuned LLM (Apple Silicon M4 optimized)")
    parser.add_argument("--model_path", type=str, default=MODEL_DIR, 
                        help="Path to the fine-tuned model")
    parser.add_argument("--base_model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="Base model name (for LoRA models)")
    parser.add_argument("--audio", type=str, help="Path to audio file for prediction")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model on test set")
    parser.add_argument("--device", type=str, default=default_device, 
                        choices=["cpu", "cuda", "mps", "auto"], 
                        help=f"Device to use (default: {default_device})")
    parser.add_argument("--limit", type=int, default=10, 
                        help="Limit number of test samples (default: 10)")
    
    args = parser.parse_args()
    
    # 加载模型
    model_path = args.model_path
    print(f"Loading model from {model_path}...")
    model, tokenizer = load_model(model_path, args.base_model, args.device)
    
    if args.audio:
        # 单个音频预测
        if not os.path.exists(args.audio):
            print(f"Error: Audio file not found: {args.audio}")
            return
        
        print(f"Predicting emotion for {args.audio}...")
        prediction = predict_emotion(model, tokenizer, args.audio, args.device)
        print(f"\nPredicted emotion: {prediction}")
    
    if args.evaluate:
        # 评估模型
        if not os.path.exists(TEST_DATA_PATH):
            print(f"Error: Test data not found at {TEST_DATA_PATH}")
            print("Please run 'python dataset/prepare_llm_data.py' first to prepare the data.")
            return
        
        evaluate_model(model, tokenizer, TEST_DATA_PATH, args.device, args.limit)

if __name__ == "__main__":
    main()
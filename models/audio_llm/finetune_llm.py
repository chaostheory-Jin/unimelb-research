#!/usr/bin/env python3
# finetune_llm.py - 使用情感语音数据微调大语言模型

import os
# 设置环境变量以禁用tokenizers并行处理
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import argparse
import sys
import platform
import torch
# 检查是否为 Apple Silicon
is_apple_silicon = platform.system() == "Darwin" and platform.machine().startswith(("arm", "aarch"))
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

# 检查必要的库是否已安装
required_packages = ["torch", "transformers", "datasets", "peft", "accelerate"]
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        missing_packages.append(package)

if missing_packages:
    print(f"Error: Missing required packages: {', '.join(missing_packages)}")
    print("Please run 'python models/audio_llm/install_requirements.py' to install them.")
    sys.exit(1)

# 导入必要的库
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

# 数据集路径
DATASET_DIR = "/Users/jinhongyu/Documents/GitHub/unimelb-research/dataset"
LLM_DATA_DIR = os.path.join(DATASET_DIR, "llm_emotion_data")
OUTPUT_DIR = os.path.join(DATASET_DIR, "emotion_llm_model")

def check_data_availability():
    """检查数据集是否可用"""
    train_path = os.path.join(LLM_DATA_DIR, "crema_d_train.json")
    val_path = os.path.join(LLM_DATA_DIR, "crema_d_val.json")
    
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"Error: Training data not found at {LLM_DATA_DIR}")
        print("Please run 'python dataset/prepare_llm_data.py' first to prepare the data.")
        return False
    
    return True

def load_dataset(dataset_path):
    """加载数据集"""
    try:
        with open(dataset_path, 'r') as f:
            data = json.load(f)
        
        # 将数据转换为适合训练的格式
        formatted_data = []
        for item in data:
            # 创建指令格式的文本
            text = f"### Instruction:\n{item['instruction']}\n\n"
            text += f"### Input:\n{item['input']}\n\n"
            text += f"### Response:\n{item['output']}"
            
            formatted_data.append({
                "text": text,
                "emotion": item["emotion"]
            })
        
        return Dataset.from_list(formatted_data)
    except Exception as e:
        print(f"Error loading dataset from {dataset_path}: {e}")
        sys.exit(1)

def tokenize_function(examples, tokenizer, max_length=1024):
    """将文本标记化"""
    result = tokenizer(
        examples["text"],
        truncation=True,
        max_length=max_length,
        padding="max_length"
    )
    return result

def finetune_llm(args):
    """微调大语言模型"""
    # 检查数据可用性
    if not check_data_availability():
        return
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, "logs"), exist_ok=True)
    
    # 加载数据集
    print("Loading datasets...")
    train_dataset = load_dataset(os.path.join(LLM_DATA_DIR, "crema_d_train.json"))
    val_dataset = load_dataset(os.path.join(LLM_DATA_DIR, "crema_d_val.json"))
    
    print(f"Loaded {len(train_dataset)} training samples and {len(val_dataset)} validation samples")
    
    # 加载模型和分词器
    print(f"Loading model: {args.model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        print("If you're using a Hugging Face model, make sure you're logged in:")
        print("huggingface-cli login")
        return
    
    # 标记化数据集
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    tokenized_val = val_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    
    # 确定设备映射
    if args.device == "auto":
        device_map = "auto"
    else:
        device_map = args.device
    
    # 加载模型
    try:
        print(f"Loading model to {device_map} (this may take a while)...")
        
        # 针对 Apple Silicon 的优化
        if is_apple_silicon and device_map == "mps":
            # 对于 MPS 设备，我们需要特别处理
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16 if device_map != "cpu" else torch.float32,
                low_cpu_mem_usage=True
            )
            # 手动将模型移动到 MPS 设备
            if torch.backends.mps.is_available():
                model = model.to("mps")
        else:
            # 对于其他设备，使用标准加载方式
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                torch_dtype=torch.float16 if device_map != "cpu" else torch.float32,
                device_map=device_map,
                low_cpu_mem_usage=True
            )
    except Exception as e:
        print(f"Error loading model: {e}")
        print("If you're using a Hugging Face model, make sure you're logged in and have access to the model.")
        print("For Apple Silicon, try using a smaller model or --device cpu")
        return
    
    # 配置LoRA
    if args.use_lora:
        print("Setting up LoRA...")
        try:
            # 针对不同模型调整目标模块
            if "llama" in args.model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            elif "mistral" in args.model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]
            else:
                # 默认目标模块
                target_modules = ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"]
            
            peft_config = LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules
            )
            
            # 对于 Apple Silicon，我们需要特别处理
            if is_apple_silicon:
                # 先将模型移到 CPU 进行 PEFT 准备
                original_device = next(model.parameters()).device
                model = model.to("cpu")
                model = prepare_model_for_kbit_training(model)
                model = get_peft_model(model, peft_config)
                # 然后再移回原来的设备
                model = model.to(original_device)
            else:
                model = prepare_model_for_kbit_training(model)
                model = get_peft_model(model, peft_config)
                
            print("LoRA setup complete")
            
            # 打印可训练参数
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%} of total)")
            
        except Exception as e:
            print(f"Error setting up LoRA: {e}")
            print("Falling back to full model fine-tuning")
            args.use_lora = False
    
    # 设置训练参数
    fp16_setting = not is_apple_silicon  # Apple Silicon 不使用 fp16
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=100,
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        logging_steps=10,
        fp16=fp16_setting,
        report_to="none",  # 禁用Wandb等报告
        # Apple Silicon 特定设置
        dataloader_num_workers=4 if is_apple_silicon else 0,
        dataloader_pin_memory=not is_apple_silicon,
    )
    
    # 创建训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )
    
    # 开始训练
    print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Error during training: {e}")
        print("Training failed. Check error message above.")
        return
    
    # 保存模型
    print(f"Saving model to {OUTPUT_DIR}")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    print("Training complete!")
    print(f"Model saved to {OUTPUT_DIR}")
    print("\nTo use the model for inference, run:")
    print(f"python models/audio_llm/emotion_inference.py --model_path {OUTPUT_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM for emotion recognition (Apple Silicon optimized)")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
                        help="Hugging Face model name (default: TinyLlama-1.1B for Apple Silicon)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1 for Apple Silicon)")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--use_lora", action="store_true", help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--device", type=str, default=default_device, 
                        choices=["cpu", "cuda", "mps", "auto"], 
                        help=f"Device to use (default: {default_device})")
    parser.add_argument("--gradient_accumulation", type=int, default=8, 
                        help="Gradient accumulation steps (default: 8 for Apple Silicon)")
    parser.add_argument("--eval_steps", type=int, default=100, help="Evaluation steps")
    parser.add_argument("--save_steps", type=int, default=100, help="Save steps")
    
    args = parser.parse_args()
    
    # 针对 Apple Silicon 的建议
    if is_apple_silicon:
        if args.batch_size > 2 and args.device != "cpu":
            print(f"Warning: Batch size {args.batch_size} may be too large for Apple Silicon.")
            print("Consider using a smaller batch size (1-2) with higher gradient accumulation.")
        
        if "llama" in args.model_name.lower() and "tiny" not in args.model_name.lower():
            print(f"Warning: {args.model_name} may be too large for Apple Silicon.")
            print("Consider using TinyLlama/TinyLlama-1.1B-Chat-v1.0 or another smaller model.")
    
    finetune_llm(args)

if __name__ == "__main__":
    main()
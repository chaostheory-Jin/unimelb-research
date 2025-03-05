#!/usr/bin/env python3
# prepare_llm_data.py - 准备用于LLM微调的情感语音数据

import os
import numpy as np
import pandas as pd
import json
import random
from tqdm import tqdm
import base64
import librosa

# 数据集路径
DATASET_DIR = "/Users/jinhongyu/Documents/GitHub/unimelb-research/dataset"
CREMA_D_DIR = os.path.join(DATASET_DIR, "CREMA-D")
OUTPUT_DIR = os.path.join(DATASET_DIR, "llm_emotion_data")

# 确保输出目录存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 情感描述映射
EMOTION_DESCRIPTIONS = {
    "A": "anger",
    "D": "disgust",
    "F": "fear",
    "H": "happiness",
    "N": "neutral",
    "S": "sadness",
    "ANG": "anger",
    "DIS": "disgust",
    "FEA": "fear",
    "HAP": "happiness",
    "NEU": "neutral",
    "SAD": "sadness"
}

# 情感详细描述
EMOTION_DETAILS = {
    "anger": "The speaker sounds angry, showing intense displeasure or hostility.",
    "disgust": "The speaker expresses disgust, showing strong aversion or repulsion.",
    "fear": "The speaker sounds fearful, expressing anxiety or terror about a threat.",
    "happiness": "The speaker sounds happy, expressing joy, pleasure, or contentment.",
    "neutral": "The speaker sounds neutral, without any strong emotional expression.",
    "sadness": "The speaker sounds sad, expressing grief, unhappiness, or sorrow."
}

# 指令模板
INSTRUCTION_TEMPLATES = [
    "Identify the emotion in this audio clip.",
    "What emotion is expressed in this audio?",
    "Analyze the emotional content of this audio.",
    "Determine the speaker's emotional state in this recording.",
    "What feeling is conveyed in this audio sample?",
    "Detect the emotion in this voice recording.",
    "What is the emotional tone of this audio clip?",
    "Classify the emotion expressed in this audio.",
    "Recognize the emotional state in this voice sample.",
    "What emotion does the speaker convey in this audio?"
]

def encode_audio_file(audio_path):
    """将音频文件编码为base64字符串"""
    try:
        with open(audio_path, "rb") as audio_file:
            audio_bytes = audio_file.read()
            base64_encoded = base64.b64encode(audio_bytes).decode('utf-8')
            return base64_encoded
    except Exception as e:
        print(f"Error encoding audio file {audio_path}: {e}")
        return None

def create_audio_description(metadata):
    """创建音频样本的文本描述"""
    emotion = EMOTION_DESCRIPTIONS.get(metadata['emotion'], metadata['emotion'])
    
    # 基本描述
    description = f"This is an audio clip of a person speaking with {emotion} emotion."
    
    # 添加详细情感描述
    if emotion in EMOTION_DETAILS:
        description += f" {EMOTION_DETAILS[emotion]}"
    
    # 添加说话者信息（如果有）
    if 'sex' in metadata and metadata['sex'] is not None:
        gender = "male" if metadata['sex'].lower() == 'm' else "female"
        description += f" The speaker is {gender}."
    
    if 'age' in metadata and metadata['age'] is not None:
        description += f" The speaker is approximately {metadata['age']} years old."
    
    # 添加句子内容（如果有）
    if 'sentence' in metadata and metadata['sentence']:
        sentence_map = {
            "IEO": "It's eleven o'clock",
            "TIE": "That is exactly what happened",
            "IOM": "I'm on my way to the meeting",
            "IWW": "I wonder what this is about",
            "TAI": "The airplane is almost full",
            "MTI": "Maybe tomorrow it will be cold",
            "IWL": "I would like a new alarm clock",
            "ITH": "I think I have a doctor's appointment",
            "DFA": "Don't forget a jacket",
            "ITS": "I think I've seen this before",
            "TSI": "The surface is slick",
            "WSI": "We'll stop in a couple of minutes"
        }
        sentence_text = sentence_map.get(metadata['sentence'], metadata['sentence'])
        description += f" The spoken text is: \"{sentence_text}\"."
    
    return description

def create_llm_training_sample(audio_path, metadata, include_audio=False):
    """创建用于LLM训练的样本"""
    # 随机选择一个指令模板
    instruction = random.choice(INSTRUCTION_TEMPLATES)
    
    # 获取情感标签
    emotion = EMOTION_DESCRIPTIONS.get(metadata['emotion'], metadata['emotion'])
    
    # 创建详细回答
    detailed_answer = f"The emotion expressed in this audio is {emotion}. {EMOTION_DETAILS.get(emotion, '')}"
    
    # 创建简短回答
    short_answer = f"The emotion is {emotion}."
    
    # 创建训练样本
    sample = {
        "instruction": instruction,
        "input": create_audio_description(metadata),
        "output": detailed_answer,
        "short_output": short_answer,
        "emotion": emotion,
        "metadata": metadata
    }
    
    # 如果需要，添加音频编码
    if include_audio:
        audio_base64 = encode_audio_file(audio_path)
        if audio_base64:
            sample["audio_base64"] = audio_base64
    
    return sample

def prepare_crema_d_data():
    """准备CREMA-D数据集"""
    print("Preparing CREMA-D dataset for LLM fine-tuning...")
    
    # 加载处理后的数据
    metadata_path = os.path.join(CREMA_D_DIR, "processed", "metadata.csv")
    if not os.path.exists(metadata_path):
        print(f"Error: Metadata file not found at {metadata_path}")
        return
    
    metadata_df = pd.read_csv(metadata_path)
    print(f"Loaded {len(metadata_df)} samples from CREMA-D")
    
    # 加载训练/验证/测试分割
    train_indices = np.load(os.path.join(CREMA_D_DIR, "processed", "train_indices.npy"))
    val_indices = np.load(os.path.join(CREMA_D_DIR, "processed", "val_indices.npy"))
    test_indices = np.load(os.path.join(CREMA_D_DIR, "processed", "test_indices.npy"))
    
    # 创建训练/验证/测试数据
    train_data = []
    val_data = []
    test_data = []
    
    # 处理训练集
    print("Processing training set...")
    for idx in tqdm(train_indices):
        if idx < len(metadata_df):
            row = metadata_df.iloc[idx]
            file_id = row['file_id']
            audio_path = os.path.join(CREMA_D_DIR, "AudioWAV", f"{file_id}.wav")
            
            if os.path.exists(audio_path):
                # 将行转换为字典
                metadata = row.to_dict()
                
                # 创建训练样本 (不包含音频编码以减小文件大小)
                sample = create_llm_training_sample(audio_path, metadata, include_audio=False)
                train_data.append(sample)
    
    # 处理验证集
    print("Processing validation set...")
    for idx in tqdm(val_indices):
        if idx < len(metadata_df):
            row = metadata_df.iloc[idx]
            file_id = row['file_id']
            audio_path = os.path.join(CREMA_D_DIR, "AudioWAV", f"{file_id}.wav")
            
            if os.path.exists(audio_path):
                metadata = row.to_dict()
                sample = create_llm_training_sample(audio_path, metadata, include_audio=False)
                val_data.append(sample)
    
    # 处理测试集 (包含音频编码)
    print("Processing test set...")
    for idx in tqdm(test_indices[:100]):  # 只处理100个测试样本以节省空间
        if idx < len(metadata_df):
            row = metadata_df.iloc[idx]
            file_id = row['file_id']
            audio_path = os.path.join(CREMA_D_DIR, "AudioWAV", f"{file_id}.wav")
            
            if os.path.exists(audio_path):
                metadata = row.to_dict()
                sample = create_llm_training_sample(audio_path, metadata, include_audio=True)
                test_data.append(sample)
    
    # 保存数据
    print(f"Saving {len(train_data)} training samples...")
    with open(os.path.join(OUTPUT_DIR, "crema_d_train.json"), 'w') as f:
        json.dump(train_data, f, indent=2)
    
    print(f"Saving {len(val_data)} validation samples...")
    with open(os.path.join(OUTPUT_DIR, "crema_d_val.json"), 'w') as f:
        json.dump(val_data, f, indent=2)
    
    print(f"Saving {len(test_data)} test samples...")
    with open(os.path.join(OUTPUT_DIR, "crema_d_test.json"), 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("CREMA-D dataset prepared for LLM fine-tuning!")

def main():
    """主函数"""
    # 准备CREMA-D数据集
    prepare_crema_d_data()
    
    # 这里可以添加其他数据集的处理
    # prepare_msp_podcast_data()
    # prepare_iemocap_data()
    
    print("All datasets prepared for LLM fine-tuning!")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
# preprocess.py

import os
import pandas as pd
import numpy as np
import librosa
import json
from tqdm import tqdm

DATASET_DIR = "/Users/jinhongyu/Documents/GitHub/unimelb-research/dataset/CREMA-D"
PROCESSED_DIR = os.path.join(DATASET_DIR, "processed")

def extract_features(file_path, sr=16000, n_mfcc=13, n_fft=2048, hop_length=512):
    """从音频文件中提取MFCC特征"""
    try:
        # 加载音频文件
        y, sr = librosa.load(file_path, sr=sr)
        
        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        
        # 计算统计量
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)
        mfccs_max = np.max(mfccs, axis=1)
        mfccs_min = np.min(mfccs, axis=1)
        
        # 合并特征
        features = np.concatenate((mfccs_mean, mfccs_std, mfccs_max, mfccs_min))
        
        return features
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

# 添加一个函数来将NumPy类型转换为Python原生类型
def convert_to_serializable(obj):
    """将NumPy类型转换为Python原生类型，以便JSON序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    return obj

def main():
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)
    
    # 检查必要的文件是否存在
    tabulated_votes_path = os.path.join(DATASET_DIR, "processedResults", "tabulatedVotes.csv")
    if not os.path.exists(tabulated_votes_path):
        # 尝试其他可能的位置
        tabulated_votes_path = os.path.join(DATASET_DIR, "finishedEmoResponses.csv")
        if not os.path.exists(tabulated_votes_path):
            print("Error: Could not find tabulatedVotes.csv or finishedEmoResponses.csv")
            return
    
    # 加载标签数据
    print(f"Loading emotion labels from {tabulated_votes_path}...")
    try:
        votes_df = pd.read_csv(tabulated_votes_path)
        print(f"Loaded {len(votes_df)} rows. Column names: {votes_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading {tabulated_votes_path}: {e}")
        return
    
    # 加载演员人口统计数据
    demographics_path = os.path.join(DATASET_DIR, "VideoDemographics.csv")
    print(f"Loading demographics from {demographics_path}...")
    try:
        demographics_df = pd.read_csv(demographics_path)
        print(f"Loaded {len(demographics_df)} rows. Column names: {demographics_df.columns.tolist()}")
    except Exception as e:
        print(f"Error loading {demographics_path}: {e}")
        return
    
    # 创建特征和标签列表
    features_list = []
    metadata_list = []
    
    # 确定文件名列和情感列
    file_col = None
    emo_col = None
    
    # 检查列名以确定正确的列
    if 'fileName' in votes_df.columns:
        file_col = 'fileName'
    elif 'FileName' in votes_df.columns:
        file_col = 'FileName'
    elif 'ClipName' in votes_df.columns:
        file_col = 'ClipName'
    
    if 'emoVote' in votes_df.columns:
        emo_col = 'emoVote'
    elif 'EmoVote' in votes_df.columns:
        emo_col = 'EmoVote'
    elif 'Emotion' in votes_df.columns:
        emo_col = 'Emotion'
    
    if file_col is None or emo_col is None:
        print(f"Error: Could not identify file name or emotion columns in {votes_df.columns.tolist()}")
        return
    
    print(f"Using {file_col} for file names and {emo_col} for emotions")
    
    # 处理每个音频文件
    print("Extracting features from audio files...")
    processed_count = 0
    error_count = 0
    
    for index, row in tqdm(votes_df.iterrows(), total=len(votes_df)):
        try:
            file_name = row[file_col]
            
            # 处理文件名以匹配实际文件
            if isinstance(file_name, str):
                if file_name.endswith('.wav'):
                    file_id = file_name[:-4]  # 移除.wav扩展名
                else:
                    file_id = file_name
            else:
                continue  # 跳过非字符串文件名
            
            # 构建音频文件路径
            audio_path = os.path.join(DATASET_DIR, "AudioWAV", f"{file_id}.wav")
            
            if not os.path.exists(audio_path):
                # 尝试其他可能的路径
                audio_path = os.path.join(DATASET_DIR, "AudioWAV", file_id)
                if not os.path.exists(audio_path):
                    print(f"Warning: Audio file not found: {file_id}")
                    error_count += 1
                    continue
            
            # 提取特征
            features = extract_features(audio_path)
            
            if features is not None:
                features_list.append(features)
                
                # 获取演员ID - 通常是文件名的前4个字符
                actor_id = file_id[:4] if len(file_id) >= 4 else file_id
                
                # 获取演员人口统计数据
                actor_demo = demographics_df[demographics_df['ActorID'] == int(actor_id)] if actor_id.isdigit() else pd.DataFrame()
                
                # 解析文件名获取情感和级别
                parts = file_id.split('_')
                sentence = parts[1] if len(parts) > 1 else ""
                emotion = parts[2] if len(parts) > 2 else ""
                level = parts[3] if len(parts) > 3 else ""
                
                # 从标签数据获取情感
                label_emotion = row[emo_col]
                
                # 获取协议列（如果存在）
                agreement = row['agreement'] if 'agreement' in row else None
                
                # 保存元数据 - 确保所有值都是可序列化的
                metadata = {
                    'file_id': file_id,
                    'actor_id': actor_id,
                    'sentence': sentence,
                    'emotion_from_filename': emotion,
                    'level': level,
                    'emotion': label_emotion,
                    'agreement': convert_to_serializable(agreement)
                }
                
                # 添加人口统计数据（如果有）
                if not actor_demo.empty:
                    for col in actor_demo.columns:
                        if col != 'ActorID':
                            metadata[col.lower()] = convert_to_serializable(actor_demo[col].values[0])
                
                metadata_list.append(metadata)
                processed_count += 1
                
                # 每处理100个文件打印一次进度
                if processed_count % 100 == 0:
                    print(f"Processed {processed_count} files...")
        
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            error_count += 1
    
    if len(features_list) == 0:
        print("Error: No features extracted. Check file paths and data format.")
        return
    
    # 将特征保存为numpy数组
    features_array = np.array(features_list)
    np.save(os.path.join(PROCESSED_DIR, "features.npy"), features_array)
    
    # 检查特征维度
    print(f"Feature shape: {features_array.shape}")
    
    # 创建特征列名
    feature_dim = features_array.shape[1]
    n_mfcc = feature_dim // 4  # 因为我们有mean, std, max, min四组特征
    
    feature_columns = []
    for stat in ['mean', 'std', 'max', 'min']:
        for i in range(1, n_mfcc + 1):
            feature_columns.append(f"mfcc_{stat}_{i}")
    
    # 将元数据保存为JSON文件 - 确保所有值都是可序列化的
    print("Saving metadata to JSON...")
    with open(os.path.join(PROCESSED_DIR, "metadata.json"), 'w') as f:
        # 使用自定义的JSON编码器
        json.dump(metadata_list, f, default=convert_to_serializable)
    
    # 将元数据保存为CSV文件
    print("Saving metadata to CSV...")
    metadata_df = pd.DataFrame(metadata_list)
    metadata_df.to_csv(os.path.join(PROCESSED_DIR, "metadata.csv"), index=False)
    
    # 创建训练/验证/测试集分割
    print("Creating train/val/test splits...")
    # 按演员ID分割，确保同一演员的数据不会同时出现在训练和测试集中
    actor_ids = list(set(meta['actor_id'] for meta in metadata_list))
    np.random.shuffle(actor_ids)
    
    train_actors = actor_ids[:int(0.7 * len(actor_ids))]
    val_actors = actor_ids[int(0.7 * len(actor_ids)):int(0.85 * len(actor_ids))]
    test_actors = actor_ids[int(0.85 * len(actor_ids)):]
    
    train_idx = [i for i, meta in enumerate(metadata_list) if meta['actor_id'] in train_actors]
    val_idx = [i for i, meta in enumerate(metadata_list) if meta['actor_id'] in val_actors]
    test_idx = [i for i, meta in enumerate(metadata_list) if meta['actor_id'] in test_actors]
    
    # 保存分割索引
    np.save(os.path.join(PROCESSED_DIR, "train_indices.npy"), np.array(train_idx))
    np.save(os.path.join(PROCESSED_DIR, "val_indices.npy"), np.array(val_idx))
    np.save(os.path.join(PROCESSED_DIR, "test_indices.npy"), np.array(test_idx))
    
    print(f"CREMA-D dataset processed successfully!")
    print(f"Total samples: {len(metadata_list)}")
    print(f"Training samples: {len(train_idx)}")
    print(f"Validation samples: {len(val_idx)}")
    print(f"Testing samples: {len(test_idx)}")
    print(f"Errors encountered: {error_count}")

if __name__ == "__main__":
    main()

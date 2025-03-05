#!/usr/bin/env python3
# data_loader.py

import os
import numpy as np
import json
import torch
from torch.utils.data import Dataset, DataLoader

class EmotionDataset(Dataset):
    def __init__(self, features_file, metadata_file, indices_file, emotion_map=None):
        """
        初始化情感数据集
        
        参数:
            features_file: 特征文件路径
            metadata_file: 元数据文件路径
            indices_file: 索引文件路径
            emotion_map: 情感标签到数字的映射字典
        """
        self.features = np.load(features_file)
        
        with open(metadata_file, 'r') as f:
            self.metadata = json.load(f)
        
        self.indices = np.load(indices_file)
        
        # 如果没有提供情感映射，则创建一个
        if emotion_map is None:
            emotions = set(meta['emotion'] for meta in self.metadata)
            self.emotion_map = {emotion: i for i, emotion in enumerate(sorted(emotions))}
        else:
            self.emotion_map = emotion_map
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        index = self.indices[idx]
        features = self.features[index]
        emotion = self.metadata[index]['emotion']
        label = self.emotion_map[emotion]
        
        return torch.FloatTensor(features), label

def get_dataloaders(dataset_dir, batch_size=32, num_workers=4):
    """
    创建训练、验证和测试数据加载器
    
    参数:
        dataset_dir: 处理后数据集的目录
        batch_size: 批次大小
        num_workers: 数据加载的工作线程数
    
    返回:
        train_loader, val_loader, test_loader
    """
    features_file = os.path.join(dataset_dir, "features.npy")
    metadata_file = os.path.join(dataset_dir, "metadata.json")
    
    train_indices_file = os.path.join(dataset_dir, "train_indices.npy")
    val_indices_file = os.path.join(dataset_dir, "val_indices.npy")
    test_indices_file = os.path.join(dataset_dir, "test_indices.npy")
    
    # 加载元数据以创建情感映射
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    
    emotions = set(meta['emotion'] for meta in metadata)
    emotion_map = {emotion: i for i, emotion in enumerate(sorted(emotions))}
    
    # 创建数据集
    train_dataset = EmotionDataset(features_file, metadata_file, train_indices_file, emotion_map)
    val_dataset = EmotionDataset(features_file, metadata_file, val_indices_file, emotion_map)
    test_dataset = EmotionDataset(features_file, metadata_file, test_indices_file, emotion_map)
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader, emotion_map

if __name__ == "__main__":
    # 测试MSP-Podcast数据加载器
    msp_dir = "/Users/jinhongyu/Documents/GitHub/unimelb-research/dataset/MSP-Podcast/processed"
    train_loader, val_loader, test_loader, emotion_map = get_dataloaders(msp_dir)
    
    print(f"MSP-Podcast情感映射: {emotion_map}")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")
    
    # 测试IEMOCAP数据加载器
    iemocap_dir = "/Users/jinhongyu/Documents/GitHub/unimelb-research/dataset/IEMOCAP/processed"
    train_loader, val_loader, test_loader, emotion_map = get_dataloaders(iemocap_dir)
    
    print(f"IEMOCAP情感映射: {emotion_map}")
    print(f"训练集大小: {len(train_loader.dataset)}")
    print(f"验证集大小: {len(val_loader.dataset)}")
    print(f"测试集大小: {len(test_loader.dataset)}")

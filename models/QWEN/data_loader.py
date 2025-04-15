import os
import json
import torch
import random
import numpy as np
import librosa
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoProcessor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 情感标签映射
EMOTION_MAPPING = {
    "Neutral state": "neutral",
    "Frustration": "frustrated", 
    "Anger": "angry",
    "Happiness": "happy",
    "Excited": "happy",  # 将excited合并到happy
    "Sadness": "sad",
    "Fear": "anxious",
    "Surprise": "surprised",
    "Disgust": "angry",  # 将disgust映射到angry
    "Other": "neutral",  # 默认映射
    "neutral": "neutral",
    "frustrated": "frustrated",
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "anxious": "anxious",
    "surprised": "surprised",
    # IEMOCAP原始标签
    "neu": "neutral",
    "fru": "frustrated", 
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",  # 将excited合并到happy
    "sad": "sad",
    "fea": "fearful",
    "sur": "surprised",
    "dis": "disgusted",
    "oth": "neutral"  # 默认映射
}

class IEMOCAPDataset(Dataset):
    """IEMOCAP数据集加载类"""
    
    def __init__(
        self, 
        json_file: str, 
        iemocap_root: str,
        processor: AutoProcessor,
        split: str = "train",
        max_audio_length: int = 30,
        sample_rate: Optional[int] = None,
        seed: int = 42,
        train_ratio: float = 0.8,
        debug: bool = False
    ):
        self.iemocap_root = iemocap_root
        self.processor = processor
        self.split = split
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate or processor.feature_extractor.sampling_rate
        self.debug = debug
        
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 加载并处理数据
        self.all_samples = self._load_data(json_file)
        
        # 划分训练集和验证集
        self._split_data(train_ratio)
        
        logger.info(f"加载了 {len(self.samples)} 个样本用于 {split} 集")
        
        # 统计情感分布
        self._log_emotion_distribution()
    
    def _load_data(self, json_file: str) -> List[Dict]:
        """加载并预处理数据集"""
        logger.info(f"从 {json_file} 加载数据...")
        
        # 加载JSON数据
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"JSON文件中共有 {len(data)} 个样本")
        
        # 处理样本
        valid_samples = []
        skipped_samples = 0
        path_failures = 0
        
        for item in tqdm(data, desc="处理数据集"):
            # 获取音频路径 - 特别注意这里使用'audio'字段
            rel_path = item.get('audio_path') or item.get('path') or item.get('wav_path') or item.get('audio')
            if not rel_path:
                if self.debug:
                    logger.warning(f"样本 {item.get('id', 'unknown')} 没有音频路径")
                skipped_samples += 1
                continue
            
            # 获取情感标签
            if 'emotion' not in item:
                skipped_samples += 1
                continue
            
            emotions = item['emotion']
            if isinstance(emotions, str):
                emotions = [emotions]
            
            # 如果没有有效的情感标签，跳过
            if not emotions:
                skipped_samples += 1
                continue
            
            # 计算多数投票的情感标签
            emotion = self._get_majority_emotion(emotions)
            
            # 转换相对路径
            abs_path = self._convert_path(rel_path)
            
            # 检查文件是否存在
            if not os.path.exists(abs_path):
                if self.debug:
                    logger.warning(f"找不到音频文件: {abs_path} (原始路径: {rel_path})")
                path_failures += 1
                continue
            
            # 添加到有效样本
            valid_samples.append({
                "id": item.get('id', f"sample_{len(valid_samples)}"),
                "audio_path": abs_path,
                "emotion": emotion,
                "original_emotions": emotions
            })
        
        logger.info(f"有效样本: {len(valid_samples)}, 跳过样本: {skipped_samples}, 路径失败: {path_failures}")
        
        if len(valid_samples) == 0:
            logger.error("没有找到有效样本! 请检查IEMOCAP数据集路径和JSON格式。")
        
        return valid_samples
    
    def _split_data(self, train_ratio: float):
        """划分训练集和验证集"""
        if not self.all_samples:
            self.samples = []
            return
            
        # 打乱数据
        random.shuffle(self.all_samples)
        
        # 划分数据集
        train_size = int(len(self.all_samples) * train_ratio)
        
        if self.split == "train":
            self.samples = self.all_samples[:train_size]
        else:  # validation
            self.samples = self.all_samples[train_size:]
    
    def _convert_path(self, rel_path: str) -> str:
        """将相对路径转换为绝对路径"""
        # 如果已经是绝对路径，直接返回
        if os.path.isabs(rel_path) and os.path.exists(rel_path):
            return rel_path
            
        # 规范化路径分隔符
        rel_path = os.path.normpath(rel_path.replace('/', os.sep).replace('\\', os.sep))
        
        # 构建基础路径
        base_path = self.iemocap_root
        
        # 直接拼接路径并检查
        abs_path = os.path.join(base_path, rel_path)
        if os.path.exists(abs_path):
            return abs_path
        
        # 从错误信息看到的JSON示例，路径格式是"IEMOCAP_full_release/Session1/sentences/wav/Ses01F_impro01/Ses01F_impro01_F000.wav"
        # 这意味着路径中已经包含了"IEMOCAP_full_release"，所以不需要再次添加
        
        # 尝试直接使用相对路径的最后几部分
        # 查找文件名
        basename = os.path.basename(rel_path)
        
        # 尝试找到包含这个文件的目录
        if basename.endswith('.wav'):
            file_prefix = basename.split('.')[0]  # 获取不带扩展名的文件名
            
            # 从路径中提取会话号和对话类型
            path_parts = rel_path.split(os.sep)
            session_dir = None
            dialog_type = None
            
            for part in path_parts:
                if part.startswith("Session"):
                    session_dir = part
                elif part.startswith("Ses") and "_" in part:
                    dialog_type = part
            
            # 如果找到会话号，尝试构建路径
            if session_dir and dialog_type:
                candidate = os.path.join(base_path, "IEMOCAP_full_release", session_dir, "sentences", "wav", dialog_type, basename)
                if os.path.exists(candidate):
                    return candidate
            
            # 如果只能从文件名确定会话号
            if file_prefix.startswith("Ses") and len(file_prefix) >= 5:
                session_num = file_prefix[3:4]  # 提取会话号
                session_dir = f"Session{session_num}"
                
                # 从文件名提取对话类型 (Ses01F_impro01_F000 -> Ses01F_impro01)
                if "_" in file_prefix:
                    parts = file_prefix.split("_")
                    if len(parts) >= 2:
                        dialog_type = f"{parts[0]}_{parts[1]}"
                        
                        candidate = os.path.join(base_path, "IEMOCAP_full_release", session_dir, "sentences", "wav", dialog_type, basename)
                        if os.path.exists(candidate):
                            return candidate
        
        # 如果以上方法都失败，尝试递归查找
        for session_num in range(1, 6):
            session_dir = os.path.join(base_path, "IEMOCAP_full_release", f"Session{session_num}")
            if os.path.exists(session_dir):
                for root, _, files in os.walk(session_dir):
                    if basename in files:
                        return os.path.join(root, basename)
        
        # 失败后返回原始路径
        return abs_path
    
    def _map_emotion(self, emotion: str) -> str:
        """映射情感标签"""
        if emotion in EMOTION_MAPPING:
            return EMOTION_MAPPING[emotion]
        
        # 尝试模糊匹配
        emotion_lower = emotion.lower()
        for key, value in EMOTION_MAPPING.items():
            if key.lower() in emotion_lower or emotion_lower in key.lower():
                return value
        
        return "neutral"  # 默认返回中性情感
    
    def _get_majority_emotion(self, emotions: List[str]) -> str:
        """获取多数投票的情感标签"""
        if not emotions:
            return "neutral"
        
        # 映射所有情感标签
        mapped_emotions = [self._map_emotion(e) for e in emotions]
        
        # 计数
        emotion_counts = {}
        for emotion in mapped_emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # 返回出现次数最多的情感
        return max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    def _log_emotion_distribution(self):
        """记录情感分布"""
        if not self.samples:
            logger.info(f"{self.split}集情感分布: 无样本")
            return
            
        emotion_counts = {}
        for sample in self.samples:
            emotion = sample["emotion"]
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        logger.info(f"{self.split}集情感分布:")
        for emotion, count in emotion_counts.items():
            logger.info(f"  {emotion}: {count} 样本 ({count/len(self.samples)*100:.1f}%)")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """获取样本并处理"""
        sample = self.samples[idx]
        
        try:
            # 加载音频文件
            audio, sr = librosa.load(sample["audio_path"], sr=self.sample_rate)
            
            # 限制最大长度
            max_length = self.max_audio_length * self.sample_rate
            if len(audio) > max_length:
                audio = audio[:max_length]
            
            # 构建输入提示
            prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>Identify the emotion in this audio:"
            
            # 处理输入
            inputs = self.processor(
                text=prompt, 
                audio=audio,
                sampling_rate=self.sample_rate,
                return_tensors="pt"
            )
            
            # 处理标签
            labels = self.processor(text=sample["emotion"], return_tensors="pt").input_ids[0]
            
            # 移除批次维度
            processed = {
                "input_ids": inputs.input_ids[0],
                "attention_mask": inputs.attention_mask[0],
                "labels": labels,
                "id": sample["id"],
                "emotion": sample["emotion"]
            }
            
            # 添加音频特定的属性
            if hasattr(inputs, "audio_input_values"):
                processed["audio_input_values"] = inputs.audio_input_values[0]
            if hasattr(inputs, "audio_attention_mask"):
                processed["audio_attention_mask"] = inputs.audio_attention_mask[0]
            
            return processed
            
        except Exception as e:
            logger.error(f"处理样本 {sample['id']} 时出错: {e}")
            # 返回一个空白样本以避免批处理错误
            if len(self.samples) > 1:
                return self.__getitem__(random.randint(0, len(self.samples)-1))
            else:
                # 如果只有一个样本且处理失败，返回一个空结构
                return {"id": sample["id"], "error": str(e)}

# 简化版测试函数
def test_json_file(json_file, iemocap_root):
    """测试JSON文件中的路径是否可以正确解析，关注audio字段"""
    print(f"测试JSON文件: {json_file}")
    print(f"IEMOCAP根目录: {iemocap_root}")
    
    # 加载JSON数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"JSON文件中共有 {len(data)} 个样本")
    
    success_count = 0
    failure_count = 0
    
    for i, item in enumerate(data[:10]):  # 只测试前10个样本
        # 获取音频路径 - 重点检查audio字段
        rel_path = item.get('audio')
        
        if not rel_path:
            print(f"样本 {i} ({item.get('id', 'unknown')}) 没有audio字段")
            # 检查其他可能的字段
            for field in ['audio_path', 'path', 'wav_path']:
                if field in item:
                    print(f"  但有 {field} 字段: {item[field]}")
            failure_count += 1
            continue
        
        # 构建绝对路径
        abs_path = os.path.join(iemocap_root, rel_path)
        
        if os.path.exists(abs_path):
            success_count += 1
            print(f"样本 {i} 路径正确: {abs_path}")
        else:
            failure_count += 1
            print(f"样本 {i} 路径不存在: {abs_path}")
            
            # 尝试在IEMOCAP_full_release外寻找
            basename = os.path.basename(rel_path)
            for session_num in range(1, 6):
                session_dir = os.path.join(iemocap_root, f"Session{session_num}")
                if os.path.exists(session_dir):
                    for root, _, files in os.walk(session_dir):
                        if basename in files:
                            print(f"  但在此处找到: {os.path.join(root, basename)}")
    
    print(f"\n总结: 成功 {success_count}, 失败 {failure_count}")
    
    # 如果所有样本都失败，提供更详细的建议
    if success_count == 0:
        print("\n所有样本路径转换失败! 可能的解决方案:")
        print("1. 确保IEMOCAP_full_release文件夹在正确的位置")
        print("2. JSON中的路径格式应为 'IEMOCAP_full_release/Session1/...'")
        print("3. 尝试修改_convert_path函数以适应您的路径格式")
        
        # 打印第一个样本的详细信息作为参考
        if len(data) > 0:
            item = data[0]
            rel_path = item.get('audio')
            if rel_path:
                print(f"\n样本路径示例: {rel_path}")
                parts = rel_path.split('/')
                print(f"路径组成部分: {parts}")
                
                # 构建不同的尝试路径
                paths_to_try = [
                    os.path.join(iemocap_root, rel_path),
                    os.path.join(iemocap_root, *parts[1:]) if len(parts) > 1 else ""
                ]
                
                print(f"\n可能的绝对路径:")
                for path in paths_to_try:
                    exists = os.path.exists(path)
                    print(f"  {path} - {'存在' if exists else '不存在'}")
    
    return success_count > 0

# 移到外部作为全局函数
def collate_fn(batch):
    # 过滤无效的样本
    valid_batch = [b for b in batch if b is not None and "input_ids" in b]
    if not valid_batch:
        return None
        
    # 提取keys
    keys = valid_batch[0].keys()
    
    # 整理数据
    batch_dict = {}
    for key in keys:
        if key in ["id", "emotion"]:
            batch_dict[key] = [sample[key] for sample in valid_batch]
        else:
            # 对于张量，进行padding
            if torch.is_tensor(valid_batch[0][key]):
                batch_dict[key] = torch.nn.utils.rnn.pad_sequence(
                    [sample[key] for sample in valid_batch],
                    batch_first=True,
                    padding_value=0
                )
    
    return batch_dict

def create_iemocap_dataloaders(
    json_file: str,
    iemocap_root: str,
    processor: AutoProcessor,
    batch_size: int = 4,
    train_ratio: float = 0.8,
    max_audio_length: int = 30,
    num_workers: int = 2,
    seed: int = 42,
    debug: bool = False
) -> Tuple[DataLoader, DataLoader]:
    """创建IEMOCAP数据加载器"""
    # 数据集
    train_dataset = IEMOCAPDataset(
        json_file=json_file,
        iemocap_root=iemocap_root,
        processor=processor,
        split="train",
        max_audio_length=max_audio_length,
        train_ratio=train_ratio,
        seed=seed,
        debug=debug
    )
    
    val_dataset = IEMOCAPDataset(
        json_file=json_file,
        iemocap_root=iemocap_root,
        processor=processor,
        split="validation",
        max_audio_length=max_audio_length,
        train_ratio=train_ratio,
        seed=seed,
        debug=debug
    )
    
    # 修正空数据集情况
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        logger.error("数据集为空，无法创建数据加载器!")
        return None, None
    
    # 数据加载器 - 使用全局的collate_fn
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers if num_workers > 0 else 0,  # 根据系统情况设置
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers if num_workers > 0 else 0,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader

# 在使用示例中设置num_workers=0可避免多进程问题
def example_usage():
    from transformers import AutoProcessor
    
    # 设置参数
    json_file = "iemocap_ambiguous.json"  # JSON文件路径
    iemocap_root = r"C:\Users\luoya\Desktop\unimelb-research\dataset\IEMOCAP"  # IEMOCAP根目录
    
    # 首先测试JSON文件中的路径
    test_json_file(json_file, iemocap_root)
    
    # 加载处理器
    processor_path = "./qwen2_audio_processor"  # 处理器路径
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    
    # 创建数据加载器，注意这里num_workers设为0以避免Windows多进程问题
    train_loader, val_loader = create_iemocap_dataloaders(
        json_file=json_file,
        iemocap_root=iemocap_root,
        processor=processor,
        batch_size=2,
        train_ratio=0.8,
        debug=True,
        num_workers=0  # 在Windows上设置为0避免多进程问题
    )
    
    if train_loader is None or val_loader is None:
        print("数据加载器创建失败")
        return
    
    # 打印数据集统计信息
    print(f"训练集样本数: {len(train_loader.dataset)}")
    print(f"验证集样本数: {len(val_loader.dataset)}")
    
    # 测试一个批次
    for batch in train_loader:
        if batch is None:
            continue
        print(f"批次大小: {len(batch['id'])}")
        print(f"情感标签: {batch['emotion']}")
        break

if __name__ == "__main__":
    example_usage()

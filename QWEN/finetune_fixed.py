import os
import torch
import numpy as np
import librosa
import random
import json
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from datasets import Dataset, DatasetDict
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel,
    PeftConfig
)
from torch.utils.data import Dataset as TorchDataset, DataLoader

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子
set_seed(42)

# 设置环境变量
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# 禁用 bitsandbytes 报错
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# 情感标签映射 - 与dataloader中保持一致
EMOTION_MAPPING = {
    "Neutral state": "neutral",
    "Frustration": "frustrated", 
    "Anger": "angry",
    "Happiness": "happy",
    "Excited": "happy",
    "Sadness": "sad",
    "Fear": "anxious",
    "Surprise": "surprised",
    "Disgust": "angry",
    "Other": "neutral",
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
    "exc": "happy",
    "sad": "sad",
    "fea": "fearful",
    "sur": "surprised",
    "dis": "disgusted",
    "oth": "neutral"
}

# 数据加载器部分
class IEMOCAPDataset(TorchDataset):
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
            # 获取音频路径 - 检查所有可能的字段名
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
        
        # 尝试直接使用相对路径的最后几部分
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
            # 检查文件是否存在
            if not os.path.exists(sample["audio_path"]):
                logger.error(f"音频文件不存在: {sample['audio_path']}")
                raise FileNotFoundError(f"音频文件不存在: {sample['audio_path']}")
                
            # 尝试获取文件大小
            try:
                file_size = os.path.getsize(sample["audio_path"])
                if file_size == 0:
                    logger.error(f"音频文件为空: {sample['audio_path']}")
                    raise ValueError(f"音频文件大小为0: {sample['audio_path']}")
            except Exception as e:
                logger.error(f"检查文件大小时出错: {sample['audio_path']}, 错误: {e}")
            
            # 加载音频文件，使用错误捕获
            try:
                audio, sr = librosa.load(sample["audio_path"], sr=self.sample_rate)
            except Exception as e:
                logger.error(f"加载音频文件失败: {sample['audio_path']}, 错误: {e}")
                raise
                
            # 检查音频数据
            if len(audio) == 0:
                logger.error(f"加载的音频数据为空: {sample['audio_path']}")
                raise ValueError(f"加载的音频数据为空: {sample['audio_path']}")
            
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
            logger.error(f"处理样本 {sample['id']} 时出错: {str(e)}", exc_info=True)
            # 返回一个空白样本以避免批处理错误
            if len(self.samples) > 1:
                alternate_idx = random.randint(0, len(self.samples)-1)
                # 避免重复选择同一个可能有问题的样本
                while alternate_idx == idx and len(self.samples) > 1:
                    alternate_idx = random.randint(0, len(self.samples)-1)
                logger.info(f"使用替代样本 {self.samples[alternate_idx]['id']}")
                return self.__getitem__(alternate_idx)
            else:
                # 如果只有一个样本且处理失败，返回一个空结构
                return {"id": sample["id"], "error": str(e)}

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
        num_workers=num_workers if num_workers > 0 else 0,
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

def print_trainable_parameters(model):
    """打印模型可训练参数的数量和比例"""
    trainable_params = 0
    all_params = 0
    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    logger.info(
        f"可训练参数: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%), "
        f"总参数: {all_params:,}"
    )

def get_peft_config(model):
    """获取适合模型的PEFT配置"""
    # 查找模型结构
    target_modules = []
    
    # 查找模型中的关键投影矩阵
    pattern_found = False
    for name, _ in model.named_modules():
        if any(substr in name for substr in ["q_proj", "k_proj", "v_proj", "o_proj"]):
            parts = name.split(".")
            for part in parts:
                if any(substr in part for substr in ["q_proj", "k_proj", "v_proj", "o_proj"]):
                    if part not in target_modules:
                        target_modules.append(part)
                        pattern_found = True
    
    # 如果没有找到标准命名模式，使用通用模式
    if not pattern_found:
        target_modules = ["query_key_value"]  # 通用备选
    
    logger.info(f"目标模块: {target_modules}")
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",  # 不要为偏置项使用LoRA
    )
    
    return peft_config

def main():
    logger.info("开始微调 Qwen2-Audio 模型用于情绪识别...")
    
    # 打印 PyTorch 和 CUDA 信息
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU name: {torch.cuda.get_device_name(0)}")
    
    # 加载本地模型和处理器
    local_model_path = "./qwen2_audio_model"
    local_processor_path = "./qwen2_audio_processor"
    
    processor = AutoProcessor.from_pretrained(
        local_processor_path,
        trust_remote_code=True
    )
    
    # 加载基础模型
    logger.info("加载模型中...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,  # 使用半精度
        device_map="auto"
    )
    
    # 打印模型结构
    logger.info("模型结构:")
    for name, _ in model.named_children():
        logger.info(f"- {name}")
    
    # 获取PEFT配置
    peft_config = get_peft_config(model)
    
    # 应用LoRA
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)
    
    # 加载数据集
    json_file = "iemocap_ambiguous.json"
    iemocap_root = r"C:\Users\luoya\Desktop\unimelb-research\dataset\IEMOCAP"

    # 创建数据加载器
    train_loader, val_loader = create_iemocap_dataloaders(
        json_file=json_file,
        iemocap_root=iemocap_root,
        processor=processor,
        batch_size=1,  # Qwen2-Audio模型很大，需要小batch_size
        train_ratio=0.8,
        max_audio_length=30,  # 最大音频长度(秒)
        num_workers=0,  # 避免Windows多进程问题
        debug=True  # 启用调试信息
    )

    # 转换为HuggingFace的Dataset格式
    from datasets import Dataset

    def loader_to_dataset(dataloader):
        all_data = []
        for batch in tqdm(dataloader, desc="转换数据集"):
            if batch is None:
                continue
            # 移除批次维度
            for i in range(len(batch["id"])):
                sample = {
                    "input_ids": batch["input_ids"][i],
                    "attention_mask": batch["attention_mask"][i],
                    "labels": batch["labels"][i]
                }
                if "audio_input_values" in batch:
                    sample["audio_input_values"] = batch["audio_input_values"][i]
                if "audio_attention_mask" in batch:
                    sample["audio_attention_mask"] = batch["audio_attention_mask"][i]
                all_data.append(sample)
        return Dataset.from_list(all_data)

    # 转换为Dataset格式
    train_dataset = loader_to_dataset(train_loader)
    val_dataset = loader_to_dataset(val_loader)

    # 创建数据整理器
    data_collator = DataCollatorForSeq2Seq(
        processor.tokenizer,
        model=model,
        pad_to_multiple_of=8,
    )

    # 定义训练参数
    training_args = Seq2SeqTrainingArguments(
        output_dir="./qwen2_audio_emotion_model",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # 减少以节省内存
        learning_rate=2e-5,
        num_train_epochs=3,
        warmup_ratio=0.1,  # 使用比例而不是步数
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        # evaluation_strategy="steps",  # 参数名称可能在不同版本中不同
        save_strategy="steps",
        load_best_model_at_end=False,
        push_to_hub=False,
        fp16=True,
        generation_max_length=50,
        report_to="tensorboard",
        save_total_limit=3,  # 只保存最好的3个检查点
        logging_first_step=True
    )
    
    # 创建训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    
    # 保存最终模型
    logger.info("保存微调后的模型...")
    model.save_pretrained("./qwen2_audio_emotion_final")
    
    logger.info("微调完成!")

if __name__ == "__main__":
    main()
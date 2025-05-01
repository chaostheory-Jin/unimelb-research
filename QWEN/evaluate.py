import os
import torch
import numpy as np
import librosa
import random
import json
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import re

from datasets import Dataset
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子
set_seed(42)
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

# 只保留五个主要情感类别
VALID_EMOTIONS = ["angry", "happy", "sad", "neutral"]

# 情感标签映射 - 只保留明确的映射
EMOTION_MAPPING = {
    "Neutral state": "neutral",
    "Frustration": "sad",
    "Anger": "angry",
    "Happiness": "happy",
    "Excited": "happy",
    "Sadness": "sad",
    "neutral": "neutral",
    "frustrated": "sad",
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    # IEMOCAP原始标签
    "neu": "neutral",
    "fru": "sad",
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
}

# 多样化的指令模板
INSTRUCTION_TEMPLATES = [
    "Identify the emotion in this audio. Choose one from: angry, happy, sad, neutral.",
    "What is the emotional state expressed in this audio? Select from: angry, happy, sad, neutral.",
    "Listen to this audio and determine the emotion. Options: angry, happy, sad, neutral.",
    "Detect the emotion in the following audio clip. Choose from: angry, happy, sad, neutral.",
    "Classify the emotion in this audio sample. Options are: angry, happy, sad, neutral.",
    "What emotion is being expressed in this audio? Select one: angry, happy, sad, neutral.",
    "Analyze this audio and tell me the emotion. Choose from: angry, happy, sad, neutral.",
    "Based on this audio, what is the speaker's emotional state? Select from: angry, happy, sad, neutral.",
    "Determine the emotional tone of this audio. Options: angry, happy, sad, neutral.",
    "Identify the predominant emotion in this audio clip. Choose one: angry, happy, sad, neutral.",
]

# 多样化的回答模板
ANSWER_TEMPLATES = [
    "{emotion}",
    "The emotion is {emotion}.",
    "This audio expresses {emotion}.",
    "I detect {emotion} in this audio.",
    "The speaker sounds {emotion}.",
    "The emotional state is {emotion}.",
    "This is a {emotion} emotion.",
    "{emotion} is the emotion expressed in this audio.",
    "The audio conveys a {emotion} emotional state.",
    "Based on my analysis, the emotion is {emotion}."
]

# 数据集加载类
class IEMOCAPDataset(torch.utils.data.Dataset):
    def __init__(self, json_file: str, iemocap_root: str, processor, split: str = "train",
                 max_audio_length: int = 30, sample_rate: Optional[int] = None, seed: int = 42,
                 train_ratio: float = 0.8, debug: bool = False):
        self.iemocap_root = iemocap_root
        self.processor = processor
        self.split = split
        self.max_audio_length = max_audio_length
        self.sample_rate = sample_rate or processor.feature_extractor.sampling_rate
        self.debug = debug

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.all_samples = self._load_data(json_file)
        self._split_data(train_ratio)
        logger.info(f"加载了 {len(self.samples)} 个样本用于 {split} 集")
        self._log_emotion_distribution()

    def _load_data(self, json_file: str) -> List[Dict]:
        logger.info(f"从 {json_file} 加载数据...")
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"JSON文件中共有 {len(data)} 个样本")

        valid_samples = []
        skipped_samples = 0
        path_failures = 0
        emotion_filtered = 0
        need_prediction_skipped = 0

        for item in tqdm(data, desc="处理数据集"):
            # 跳过需要预测的样本
            if item.get('need_prediction') == "yes":
                need_prediction_skipped += 1
                continue
                
            rel_path = item.get('audio_path') or item.get('path') or item.get('wav_path') or item.get('audio')
            if not rel_path:
                if self.debug:
                    logger.warning(f"样本 {item.get('id', 'unknown')} 没有音频路径")
                skipped_samples += 1
                continue

            if 'emotion' not in item:
                skipped_samples += 1
                continue

            emotions = item['emotion']
            if isinstance(emotions, str):
                emotions = [emotions]
            if not emotions:
                skipped_samples += 1
                continue

            # 直接检查原始情感是否在我们要保留的类别中
            valid_emotion = False
            mapped_emotions = []
            
            for emotion in emotions:
                # 尝试使用映射获取标准化情感
                if emotion in EMOTION_MAPPING:
                    mapped_emotion = EMOTION_MAPPING[emotion]
                    if mapped_emotion in VALID_EMOTIONS:
                        mapped_emotions.append(mapped_emotion)
                        valid_emotion = True
                # 检查原始情感是否直接匹配有效情感
                elif emotion in VALID_EMOTIONS:
                    mapped_emotions.append(emotion)
                    valid_emotion = True
                    
            # 如果没有有效的情感标签，跳过此样本
            if not valid_emotion:
                emotion_filtered += 1
                continue
                
            # 获取多数情感
            counts = {}
            for emo in mapped_emotions:
                counts[emo] = counts.get(emo, 0) + 1
            emotion = max(counts.items(), key=lambda x: x[1])[0]
                
            abs_path = self._convert_path(rel_path)
            if not os.path.exists(abs_path):
                if self.debug:
                    logger.warning(f"找不到音频文件: {abs_path} (原始路径: {rel_path})")
                path_failures += 1
                continue

            valid_samples.append({
                "id": item.get('id', f"sample_{len(valid_samples)}"),
                "audio_path": abs_path,
                "emotion": emotion,
                "original_emotions": emotions
            })
        
        logger.info(f"有效样本: {len(valid_samples)}, 需预测跳过: {need_prediction_skipped}, "
                    f"跳过无路径/标签样本: {skipped_samples}, 路径失败: {path_failures}, 情感过滤: {emotion_filtered}")
        if len(valid_samples) == 0:
            logger.error("没有找到有效样本! 请检查IEMOCAP数据集路径和JSON格式。")
        return valid_samples

    def _split_data(self, train_ratio: float):
        if not self.all_samples:
            self.samples = []
            return
        
        # 按情感类别分组
        samples_by_emotion = {}
        for sample in self.all_samples:
            emotion = sample["emotion"]
            if emotion not in samples_by_emotion:
                samples_by_emotion[emotion] = []
            samples_by_emotion[emotion].append(sample)
        
        # 查找最小类别的样本数
        min_samples = min(len(samples) for samples in samples_by_emotion.values())
        logger.info(f"最小类别样本数: {min_samples}")
        
        # 对每个类别进行平衡采样（训练集）
        balanced_samples = []
        for emotion, samples in samples_by_emotion.items():
            # 随机打乱每个类别的样本
            random.shuffle(samples)
            # 确保每个类别的样本数量一致
            balanced_count = min(len(samples), min_samples * 2)  # 保留更多样本但保持相对平衡
            balanced_samples.extend(samples[:balanced_count])
        
        # 随机打乱平衡后的样本
        random.shuffle(balanced_samples)
        
        # 计算训练/验证分割点
        train_size = int(len(balanced_samples) * train_ratio)
        
        # 分割数据
        if self.split == "train":
            self.samples = balanced_samples[:train_size]
        else:
            self.samples = balanced_samples[train_size:]
        
        logger.info(f"平衡后{self.split}集样本数: {len(self.samples)}")

    def _convert_path(self, rel_path: str) -> str:
        if os.path.isabs(rel_path) and os.path.exists(rel_path):
            if rel_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                return rel_path
            else:
                wav_path = os.path.splitext(rel_path)[0] + '.wav'
                if os.path.exists(wav_path):
                    return wav_path
        
        rel_path = os.path.normpath(rel_path.replace('/', os.sep).replace('\\', os.sep))
        base_path = self.iemocap_root
        
        abs_path = os.path.join(base_path, rel_path)
        if os.path.exists(abs_path) and abs_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            return abs_path
        
        wav_path = os.path.splitext(abs_path)[0] + '.wav'
        if os.path.exists(wav_path):
            return wav_path
        
        basename = os.path.basename(rel_path)
        basename_noext = os.path.splitext(basename)[0]
        
        for session_num in range(1, 6):
            session_dir = os.path.join(base_path, "IEMOCAP_full_release", f"Session{session_num}")
            if os.path.exists(session_dir):
                for root, _, files in os.walk(session_dir):
                    for file in files:
                        if file.startswith(basename_noext) and file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                            return os.path.join(root, file)
        
        return os.path.splitext(abs_path)[0] + '.wav'

    def _map_emotion(self, emotion: str) -> str:
        """仅映射已定义的情感类别，不对其他类别进行处理"""
        if emotion in EMOTION_MAPPING:
            return EMOTION_MAPPING[emotion]
        if emotion in VALID_EMOTIONS:
            return emotion
        return None  # 返回None表示不是我们要的情感类别

    def _get_majority_emotion(self, emotions: List[str]) -> str:
        """获取多数情感，只考虑有效的五类情感"""
        mapped = []
        for e in emotions:
            mapped_emotion = self._map_emotion(e)
            if mapped_emotion in VALID_EMOTIONS:  # 只添加有效的情感
                mapped.append(mapped_emotion)
        
        if not mapped:  # 如果没有有效的情感，返回None
            return None
            
        counts = {}
        for emo in mapped:
            counts[emo] = counts.get(emo, 0) + 1
        return max(counts.items(), key=lambda x: x[1])[0]

    def _log_emotion_distribution(self):
        if not self.samples:
            logger.info(f"{self.split}集情感分布: 无样本")
            return
        counts = {}
        for sample in self.samples:
            counts[sample["emotion"]] = counts.get(sample["emotion"], 0) + 1
        logger.info(f"{self.split}集情感分布:")
        for emo, cnt in counts.items():
            logger.info(f"  {emo}: {cnt} 样本 ({cnt/len(self.samples)*100:.1f}%)")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]
        try:
            if not os.path.exists(sample["audio_path"]):
                logger.error(f"音频文件不存在: {sample['audio_path']}")
                raise FileNotFoundError(f"音频文件不存在: {sample['audio_path']}")
            if os.path.getsize(sample["audio_path"]) == 0:
                logger.error(f"音频文件为空: {sample['audio_path']}")
                raise ValueError(f"音频文件大小为0: {sample['audio_path']}")

            try:
                audio, sr = librosa.load(sample["audio_path"], sr=self.sample_rate)
            except Exception as e:
                logger.error(f"加载音频文件失败: {sample['audio_path']}, 错误: {e}")
                raise

            if len(audio) == 0:
                logger.error(f"加载的音频数据为空: {sample['audio_path']}")
                raise ValueError(f"加载的音频数据为空: {sample['audio_path']}")
            max_length = self.max_audio_length * self.sample_rate
            if len(audio) > max_length:
                audio = audio[:max_length]

            # 使用更清晰的提示模板
            if self.split == "train":
                # 训练时使用多样的提示
                prompt_template = random.choice([
                    "Tell me the emotion in this audio. Options: angry, happy, sad, neutral.",
                    "What emotion does this person express? Choose from: angry, happy, sad, neutral.",
                    "Identify the emotion: angry, happy, sad, neutral."
                ])
            else:
                # 验证时使用固定提示
                prompt_template = "What is the emotion in this audio? angry, happy, sad, neutral."
            
            prompt = f"<|audio_bos|><|AUDIO|><|audio_eos|>{prompt_template}"
            
            # 使用完整句子作为目标，例如 "The emotion is happy."
            target_prefix = random.choice(["The emotion is ", "This audio expresses ", "I detect "]) if self.split == "train" else "The emotion is "
            target_text = f"{target_prefix}{sample['emotion']}."
            
            # 处理输入
            inputs = self.processor(text=prompt, audio=audio, sampling_rate=self.sample_rate, return_tensors="pt")
            
            # 编码目标
            target_encoding = self.processor.tokenizer(target_text, return_tensors="pt")
            labels = target_encoding.input_ids[0]
            
            # 确保标签是合适的长度，避免维度不匹配
            if len(labels) > len(inputs.input_ids[0]):
                labels = labels[:len(inputs.input_ids[0])]
            else:
                # 用 -100 填充标签
                padding = torch.full((len(inputs.input_ids[0]) - len(labels),), -100, dtype=torch.long)
                labels = torch.cat([labels, padding])
            
            return {
                "input_ids": inputs.input_ids[0],
                "attention_mask": inputs.attention_mask[0],
                "labels": labels
            }
        except Exception as e:
            logger.error(f"处理样本出错: {str(e)}")
            # 返回随机样本代码保持不变...

class CustomDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        if not features:
            return None
            
        max_input_length = max(len(f["input_ids"]) for f in features)
        
        batch = {
            "input_ids": torch.zeros((len(features), max_input_length), dtype=torch.long),
            "attention_mask": torch.zeros((len(features), max_input_length), dtype=torch.long),
            "labels": [],
        }
        
        for i, feature in enumerate(features):
            input_ids = feature["input_ids"]
            attn_mask = feature["attention_mask"]
            
            if not isinstance(input_ids, torch.Tensor):
                input_ids = torch.tensor(input_ids, dtype=torch.long)
            if not isinstance(attn_mask, torch.Tensor):
                attn_mask = torch.tensor(attn_mask, dtype=torch.long)
                
            batch["input_ids"][i, :len(input_ids)] = input_ids
            batch["attention_mask"][i, :len(attn_mask)] = attn_mask
            
            if not isinstance(feature["labels"], torch.Tensor):
                batch["labels"].append(torch.tensor(feature["labels"], dtype=torch.long))
            else:
                batch["labels"].append(feature["labels"])
            
            if "audio_features" in feature:
                if "audio_features" not in batch:
                    batch["audio_features"] = []
                batch["audio_features"].append(feature["audio_features"])
                
        if batch["labels"]:
            try:
                batch["labels"] = torch.nn.utils.rnn.pad_sequence(
                    batch["labels"], batch_first=True, padding_value=-100
                )
            except TypeError as e:
                print(f"标签类型: {[type(l) for l in batch['labels']]}")
                print(f"标签形状: {[l.shape if isinstance(l, torch.Tensor) else None for l in batch['labels']]}")
                raise e
            
        if "audio_features" in batch and batch["audio_features"]:
            batch["audio_features"] = torch.stack(batch["audio_features"])
            
        return batch

def loader_to_dataset(dataloader) -> Dataset:
    all_samples = []
    for batch in tqdm(dataloader, desc="转换数据集"):
        if batch is None:
            continue
        bs = batch["input_ids"].size(0)
        for i in range(bs):
            sample = {
                "input_ids": batch["input_ids"][i],
                "attention_mask": batch["attention_mask"][i],
                "labels": batch["labels"][i]
            }
            if "audio_features" in batch:
                sample["audio_features"] = batch["audio_features"][i]
            all_samples.append(sample)
    return Dataset.from_list(all_samples)

def get_peft_config() -> LoraConfig:
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    logger.info(f"使用目标模块: {target_modules}")
    return LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=target_modules,
        bias="none",
    )

def print_trainable_parameters(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"可训练参数: {trainable:,} ({100 * trainable / total:.2f}%), 总参数: {total:,}")

def collate_fn(batch: List[Dict]) -> Dict:
    valid_batch = [x for x in batch if x and len(x.get("input_ids", [])) > 0]
    if not valid_batch:
        return None
    
    keys = valid_batch[0].keys()
    collated = {}
    
    for key in keys:
        if key in ["input_ids", "attention_mask", "labels", "audio_features"]:
            collated[key] = torch.nn.utils.rnn.pad_sequence(
                [sample[key] for sample in valid_batch],
                batch_first=True, padding_value=0
            )
        else:
            collated[key] = [sample[key] for sample in valid_batch]
    
    return collated

def dataset_collate_fn(batch, processor=None, model=None):
    # 过滤掉空张量
    valid_batch = [b for b in batch if len(b.get("input_ids", [])) > 0]
    if not valid_batch:
        return None
    
    # 确保所有字段都是张量格式
    for i, item in enumerate(valid_batch):
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in item and not isinstance(item[key], torch.Tensor):
                valid_batch[i][key] = torch.tensor(item[key], dtype=torch.long)
    
    # 手动进行填充以确保尺寸一致
    max_length = max(len(item["input_ids"]) for item in valid_batch)
    
    # 初始化批次字典
    batch_dict = {
        "input_ids": torch.zeros((len(valid_batch), max_length), dtype=torch.long),
        "attention_mask": torch.zeros((len(valid_batch), max_length), dtype=torch.long),
        "labels": torch.full((len(valid_batch), max_length), -100, dtype=torch.long),
    }
    
    # 手动填充
    for i, item in enumerate(valid_batch):
        input_length = len(item["input_ids"])
        batch_dict["input_ids"][i, :input_length] = item["input_ids"]
        batch_dict["attention_mask"][i, :input_length] = item["attention_mask"]
        
        # 确保标签长度一致
        label_length = min(len(item["labels"]), max_length)
        batch_dict["labels"][i, :label_length] = item["labels"][:label_length]
    
    # 处理音频特征
    if "audio_features" in valid_batch[0]:
        # 检查所有样本是否都有音频特征
        if all("audio_features" in item for item in valid_batch):
            # 堆叠所有音频特征
            batch_dict["audio_features"] = torch.stack([item["audio_features"] for item in valid_batch])
    
    return batch_dict

def create_collator(proc, mdl):
    def collate_fn_wrapper(batch):
        return dataset_collate_fn(batch, proc, mdl)
    return collate_fn_wrapper

# 修改情感提取函数，移除frustrated相关代码并添加到sad中
def extract_emotion_from_text(text):
    text = text.lower()
    
    # 处理标准emotions和它们的变体
    emotion_mapping = {
        "angry": ["angry", "anger", "mad", "furious", "rage", "irritated", "annoyed"],
        "happy": ["happy", "happiness", "joy", "joyful", "pleased", "delight", "cheerful", "glad", "excited"],
        "sad": ["sad", "sadness", "unhappy", "depressed", "sorrow", "gloomy", "miserable", "disappointed", "upset"],
        "neutral": ["neutral", "calm", "normal", "indifferent", "flat", "no emotion"]
    }
    
    # 处理模型常见的其他情感输出(映射到我们的四种标准情感)
    alternative_emotions = {
        "surprise": "happy",
        "surprised": "happy", 
        "astonishment": "happy",
        "disbelief": "neutral",
        "agreement": "neutral",
        "frustration": "sad",
        "confused": "sad",
        "fear": "sad",
        "anxious": "sad"
    }
    
    # 1. 首先尝试找"the emotion is X"这样的模式
    emotion_pattern = r"the emotion is\s+(\w+)"
    matches = re.findall(emotion_pattern, text)
    
    if matches:
        for match in matches:
            clean_match = match.strip(".,;:!? ")
            
            # 检查是否直接匹配标准情感
            for emotion, variants in emotion_mapping.items():
                if clean_match in variants:
                    return emotion
            
            # 检查是否匹配替代情感
            if clean_match in alternative_emotions:
                return alternative_emotions[clean_match]
    
    # 2. 检查更广泛的模式，例如"X emotion"
    for emotion, variants in emotion_mapping.items():
        for variant in variants:
            if variant in text:
                return emotion
    
    # 3. 检查替代情感词
    for alt_emotion, mapped_emotion in alternative_emotions.items():
        if alt_emotion in text:
            return mapped_emotion
    
    # 4. 处理特殊情况
    if "surprise" in text or "surprised" in text or "astonishment" in text:
        return "happy"
    if "agree" in text or "nodding" in text or "approval" in text:
        return "neutral"
    if "disbelief" in text:
        return "neutral"
    
    # 5. 从文本中提取可能的情感词
    potential_emotion_words = ["happiness", "happy", "joy", "angry", "anger", "sad", "sadness", "neutral", "surprise", 
                              "approval", "disbelief", "agreement", "confused", "fear", "anxious"]
    
    for word in potential_emotion_words:
        if word in text:
            # 映射到标准情感
            for emotion, variants in emotion_mapping.items():
                if word in variants:
                    return emotion
            
            # 检查替代情感
            if word in alternative_emotions:
                return alternative_emotions[word]
    
    # 如果没有找到有效的情感，检查是否包含表示积极或消极的词汇
    positive_indicators = ["positive", "pleasant", "good", "joyful", "cheerful", "upbeat"]
    negative_indicators = ["negative", "unpleasant", "bad", "distressed", "downbeat", "displeasure"]
    
    for word in positive_indicators:
        if word in text:
            return "happy"
    
    for word in negative_indicators:
        if word in text:
            return "sad"
    
    # 如果什么都没找到，默认为neutral
    return "neutral"

# 评估模型函数
def evaluate_model(model, processor, test_dataset, iemocap_root, save_results=True):
    logger.info("开始评估模型...")
    model.eval()
    
    # 确保模型在正确的设备上
    device = model.device
    logger.info(f"评估使用设备: {device}")
    
    results = []
    true_labels = []
    pred_labels = []
    
    for idx, item in enumerate(tqdm(test_dataset.samples, desc="评估进度")):
        try:
            abs_path = item["audio_path"]
            true_emotion = item["emotion"]
            true_labels.append(true_emotion)
            
            audio, sr = librosa.load(abs_path, sr=16000)
            if len(audio) == 0:
                logger.warning(f"音频数据为空: {abs_path}")
                continue
                
            max_length = 30 * 16000  # 30秒
            if len(audio) > max_length:
                audio = audio[:max_length]
                
            # 使用不同的提示来评估每个样本，这样可以减少特定提示的偏差
            prompts = [
                "<|audio_bos|><|AUDIO|><|audio_eos|>What is the emotion in this audio? Please select one: angry, happy, sad, or neutral. The emotion is",
                "<|audio_bos|><|AUDIO|><|audio_eos|>Tell me the emotion expressed in this audio. Choose exactly one from: angry, happy, sad, or neutral. The emotion is",
                "<|audio_bos|><|AUDIO|><|audio_eos|>Identify the emotion in this audio as either angry, happy, sad, or neutral. The emotion is"
            ]
            
            # 对每个样本使用多个提示，统计所有结果
            all_predictions = []
            for prompt in prompts:
                inputs = processor(text=prompt, audio=audio, sampling_rate=16000, return_tensors="pt")
                
                for key in inputs:
                    inputs[key] = inputs[key].to(device)
                    
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=100,  # 增加长度
                        min_new_tokens=5,    # 强制生成至少5个新token
                        num_beams=5,
                        no_repeat_ngram_size=2,
                        temperature=0.9,     # 稍微提高温度增加多样性
                        do_sample=True,      # 启用采样
                        forced_decoder_ids=None,  # 确保没有强制生成特定token
                        early_stopping=False  # 禁止早停，确保生成足够长度
                    )
                    
                prediction_text = processor.batch_decode(output, skip_special_tokens=True)[0].strip().lower()
                logger.info(f"原始预测: {prediction_text}")
                
                # 更复杂的情感提取逻辑
                detected_emotion = extract_emotion_from_text(prediction_text)
                logger.info(f"原始预测文本: {prediction_text}")
                logger.info(f"提取到的情感: {detected_emotion if detected_emotion else 'None'}")
                
                if detected_emotion:
                    all_predictions.append(detected_emotion)
            
            # 如果至少有一个有效预测，使用最频繁出现的
            if all_predictions:
                # 统计每个预测的频率
                emotion_counts = {}
                for emotion in all_predictions:
                    emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
                
                # 选择最频繁的情感
                pred_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
            else:
                # 默认为中性
                pred_emotion = "neutral"
                
            pred_labels.append(pred_emotion)
            
            result = {
                "id": item.get('id', f"sample_{idx}"),
                "audio_path": abs_path,
                "true_emotion": true_emotion,
                "predicted_emotion": pred_emotion,
                "predicted_text": prediction_text,
                "correct": pred_emotion == true_emotion
            }
            results.append(result)
            
            # 在评估函数中添加
            raw_output_ids = output[0].tolist()
            input_length = inputs.input_ids.size(1)
            new_tokens = raw_output_ids[input_length:]
            logger.info(f"新生成的token IDs: {new_tokens}")
            logger.info(f"解码后: {processor.tokenizer.decode(new_tokens)}")
            
        except Exception as e:
            logger.error(f"处理样本时出错: {abs_path}, 错误: {str(e)}")
            continue
    
    # 计算评估指标
    accuracy = accuracy_score(true_labels, pred_labels)
    report = classification_report(true_labels, pred_labels, zero_division=0, output_dict=True)
    report_text = classification_report(true_labels, pred_labels, zero_division=0)
    conf_matrix = confusion_matrix(true_labels, pred_labels, labels=VALID_EMOTIONS)
    
    # 计算每个类别的样本数量
    class_counts = {}
    for emotion in VALID_EMOTIONS:
        class_counts[emotion] = true_labels.count(emotion)
    
    # 为每个类别计算精确率、召回率和F1分数
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    support_per_class = {}
    
    for emotion in VALID_EMOTIONS:
        if emotion in report:
            precision_per_class[emotion] = report[emotion]['precision']
            recall_per_class[emotion] = report[emotion]['recall']
            f1_per_class[emotion] = report[emotion]['f1-score']
            support_per_class[emotion] = report[emotion]['support']
    
    logger.info(f"准确率: {accuracy:.4f}")
    logger.info(f"详细分类报告:\n{report_text}")
    logger.info("每个类别的样本数量:")
    for emotion, count in class_counts.items():
        logger.info(f"  {emotion}: {count} 样本")
    
    # 可视化混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=VALID_EMOTIONS,
                yticklabels=VALID_EMOTIONS)
    plt.title('情感识别混淆矩阵')
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.tight_layout()
    
    # 保存结果
    if save_results:
        os.makedirs("evaluation_results", exist_ok=True)
        plt.savefig("evaluation_results/confusion_matrix.png")
        
        # 保存详细指标到文本文件
        with open("evaluation_results/metrics.txt", "w") as f:
            f.write(f"准确率: {accuracy:.4f}\n\n")
            f.write(f"详细分类报告:\n{report_text}\n\n")
            f.write("每个类别的样本数量:\n")
            for emotion, count in class_counts.items():
                f.write(f"  {emotion}: {count} 样本\n")
        
        # 保存详细预测结果到CSV
        df = pd.DataFrame(results)
        df.to_csv("evaluation_results/prediction_results.csv", index=False)
        
        # 保存指标到JSON以便后续分析
        metrics = {
            "accuracy": accuracy,
            "class_counts": class_counts,
            "precision_per_class": precision_per_class,
            "recall_per_class": recall_per_class,
            "f1_per_class": f1_per_class,
            "support_per_class": support_per_class,
            "confusion_matrix": conf_matrix.tolist(),
            "report": report
        }
        with open("evaluation_results/detailed_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4)
        
        # 保存所有结果为pickle文件，方便后续加载和分析
        import pickle
        full_results = {
            "true_labels": true_labels,
            "pred_labels": pred_labels,
            "results": results,
            "metrics": metrics
        }
        with open("evaluation_results/full_results.pkl", "wb") as f:
            pickle.dump(full_results, f)
        
        # 计算每个情感类别的准确率
        emotion_accuracies = {}
        for emotion in VALID_EMOTIONS:
            indices = [i for i, label in enumerate(true_labels) if label == emotion]
            if indices:
                correct = sum(pred_labels[i] == true_labels[i] for i in indices)
                emotion_accuracies[emotion] = correct / len(indices)
            else:
                emotion_accuracies[emotion] = 0
        
        # 绘制情感准确率条形图
        plt.figure(figsize=(12, 6))
        emotions = list(emotion_accuracies.keys())
        accuracies = list(emotion_accuracies.values())
        plt.bar(emotions, accuracies)
        plt.title('各情感类别的识别准确率')
        plt.xlabel('情感')
        plt.ylabel('准确率')
        plt.ylim(0, 1)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("evaluation_results/emotion_accuracies.png")
        
        # 绘制精确率、召回率和F1分数对比图
        plt.figure(figsize=(15, 6))
        x = np.arange(len(VALID_EMOTIONS))
        width = 0.25
        
        plt.bar(x - width, [precision_per_class.get(e, 0) for e in VALID_EMOTIONS], width, label='Precision')
        plt.bar(x, [recall_per_class.get(e, 0) for e in VALID_EMOTIONS], width, label='Recall')
        plt.bar(x + width, [f1_per_class.get(e, 0) for e in VALID_EMOTIONS], width, label='F1-Score')
        
        plt.xlabel('情感类别')
        plt.ylabel('分数')
        plt.title('各情感类别的精确率、召回率和F1分数')
        plt.xticks(x, VALID_EMOTIONS, rotation=45)
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        plt.savefig("evaluation_results/precision_recall_f1.png")
    
    return accuracy, report, results, metrics

# 5. 添加回调来监控训练进度和类别分布
class EmotionMetricsCallback(TrainerCallback):
    def __init__(self, eval_dataset, processor):
        self.eval_dataset = eval_dataset
        self.processor = processor
    
    def on_evaluate(self, args, state, control, model, metrics=None, **kwargs):
        # 在每次评估时检查类别分布
        logger.info("分析当前类别预测分布...")
        
        # 创建一个小批量样本进行评估
        eval_samples = random.sample(self.eval_dataset.samples, min(10, len(self.eval_dataset.samples)))
        
        predictions = []
        for sample in eval_samples:
            try:
                audio, sr = librosa.load(sample["audio_path"], sr=16000)
                if len(audio) == 0 or len(audio) > 30 * 16000:
                    continue
                    
                prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>What is the emotion in this audio, choose from: angry, happy, sad, neutral?"
                inputs = self.processor(text=prompt, audio=audio, sampling_rate=16000, return_tensors="pt")
                
                # 移动到模型设备
                device = next(model.parameters()).device
                for key in inputs:
                    inputs[key] = inputs[key].to(device)
                
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        do_sample=False
                    )
                
                prediction = self.processor.batch_decode(output, skip_special_tokens=True)[0].strip().lower()
                
                # 提取预测的情感
                pred_emotion = None
                for emotion in VALID_EMOTIONS:
                    if emotion in prediction.lower():
                        pred_emotion = emotion
                        break
                
                if pred_emotion:
                    predictions.append(pred_emotion)
                else:
                    # 默认为中性
                    predictions.append("neutral")
                    
            except Exception as e:
                continue
        
        # 统计预测分布
        if predictions:
            emotion_counts = {}
            for emotion in predictions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            logger.info("当前预测分布:")
            for emotion, count in emotion_counts.items():
                logger.info(f"  {emotion}: {count} ({count/len(predictions)*100:.1f}%)")
        
        # 重要：不要返回任何值，这会替换控制对象
        # 错误的做法: return metrics

def main():
    logger.info("开始微调 Qwen2-Audio 模型用于五分类情绪识别...")

    logger.info(f"PyTorch版本: {torch.__version__}")
    logger.info(f"CUDA是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"CUDA版本: {torch.version.cuda}")
        logger.info(f"GPU名称: {torch.cuda.get_device_name(0)}")

    # 模型和处理器路径
    local_model_path = "./qwen2_audio_model"
    local_processor_path = "./qwen2_audio_processor"
    processor = AutoProcessor.from_pretrained(local_processor_path, trust_remote_code=True)

    # 数据集路径
    json_file = "iemocap_ambiguous.json"
    iemocap_root = r"C:\Users\luoya\Desktop\unimelb-research\dataset\IEMOCAP"

    from torch.utils.data import DataLoader
    train_dataset_obj = IEMOCAPDataset(
        json_file=json_file, iemocap_root=iemocap_root, processor=processor,
        split="train", max_audio_length=30, train_ratio=0.8, debug=True
    )
    val_dataset_obj = IEMOCAPDataset(
        json_file=json_file, iemocap_root=iemocap_root, processor=processor,
        split="validation", max_audio_length=30, train_ratio=0.8, debug=True
    )
    train_loader = DataLoader(train_dataset_obj, batch_size=1, shuffle=True, collate_fn=collate_fn, num_workers=0)
    val_loader = DataLoader(val_dataset_obj, batch_size=1, shuffle=False, collate_fn=collate_fn, num_workers=0)

    logger.info("开始转换训练集...")
    train_dataset = loader_to_dataset(train_loader)
    logger.info("开始转换验证集...")
    val_dataset = loader_to_dataset(val_loader)
    logger.info(f"训练集样本数: {len(train_dataset)}")
    logger.info(f"验证集样本数: {len(val_dataset)}")

    training_args = Seq2SeqTrainingArguments(
        output_dir="./qwen2_audio_emotion_model",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-6,
        num_train_epochs=10,
        warmup_ratio=0.2,
        weight_decay=0.01,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        save_strategy="steps",
        eval_strategy="steps",
        fp16=True,
        generation_max_length=30,
        report_to="tensorboard",
        save_total_limit=3,
        logging_first_step=True,
        predict_with_generate=False,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        remove_unused_columns=False,
    )
    
    
    # 加载微调后的模型
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "./qwen2_audio_model",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    eval_model = PeftModel.from_pretrained(base_model, "./qwen2_audio_emotion_final")
    eval_model.eval()
    
    # 检查模型设备
    device = next(eval_model.parameters()).device
    logger.info(f"评估模型位于设备: {device}")
    
    # 评估模型
    accuracy, report, results, metrics = evaluate_model(
        model=eval_model,
        processor=processor,
        test_dataset=val_dataset_obj,
        iemocap_root=iemocap_root,
        save_results=True
    )
    
    logger.info(f"最终评估准确率: {accuracy:.4f}")
    logger.info(f"宏平均F1: {metrics['report']['macro avg']['f1-score']:.4f}")
    logger.info(f"加权平均F1: {metrics['report']['weighted avg']['f1-score']:.4f}")
    logger.info("评估指标已保存到 evaluation_results 目录")
    logger.info("所有流程完成!")

    # 添加调试代码
    for emotion in VALID_EMOTIONS:
        emotion_ids = processor.tokenizer(emotion, return_tensors="pt").input_ids
        decoded = processor.tokenizer.decode(emotion_ids[0])
        logger.info(f"编码-解码 '{emotion}' -> '{decoded}'")

if __name__ == "__main__":
    main()

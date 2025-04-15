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

from datasets import Dataset
from transformers import (
    AutoProcessor,
    Qwen2AudioForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed
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
VALID_EMOTIONS = ["angry", "happy", "frustrated", "sad", "neutral"]

# 情感标签映射
EMOTION_MAPPING = {
    "Neutral state": "neutral",
    "Frustration": "frustrated", 
    "Anger": "angry",
    "Happiness": "happy",
    "Excited": "happy",
    "Sadness": "sad",
    "Fear": "neutral",  # 将少数类映射到neutral
    "Surprise": "neutral",  # 将少数类映射到neutral
    "Disgust": "angry",
    "Other": "neutral",
    "neutral": "neutral",
    "frustrated": "frustrated",
    "angry": "angry",
    "happy": "happy",
    "sad": "sad",
    "anxious": "neutral",  # 将少数类映射到neutral
    "surprised": "neutral",  # 将少数类映射到neutral
    "fearful": "neutral",  # 将少数类映射到neutral
    # IEMOCAP原始标签
    "neu": "neutral",
    "fru": "frustrated", 
    "ang": "angry",
    "hap": "happy",
    "exc": "happy",
    "sad": "sad",
    "fea": "neutral",  # 将少数类映射到neutral
    "sur": "neutral",  # 将少数类映射到neutral
    "dis": "angry",
    "oth": "neutral"
}

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

        for item in tqdm(data, desc="处理数据集"):
            # # 跳过需要预测的样本
            # if item.get('need_prediction') == "yes":
            #     skipped_samples += 1
            #     continue
                
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

            emotion = self._get_majority_emotion(emotions)
            
            # 过滤掉不在五分类范围内的情感
            if emotion not in VALID_EMOTIONS:
                emotion_filtered += 1
                continue
                
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
        logger.info(f"有效样本: {len(valid_samples)}, 跳过样本: {skipped_samples}, 路径失败: {path_failures}, 情感过滤: {emotion_filtered}")
        if len(valid_samples) == 0:
            logger.error("没有找到有效样本! 请检查IEMOCAP数据集路径和JSON格式。")
        return valid_samples

    def _split_data(self, train_ratio: float):
        if not self.all_samples:
            self.samples = []
            return
        random.shuffle(self.all_samples)
        train_size = int(len(self.all_samples) * train_ratio)
        if self.split == "train":
            self.samples = self.all_samples[:train_size]
        else:
            self.samples = self.all_samples[train_size:]

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
        if emotion in EMOTION_MAPPING:
            return EMOTION_MAPPING[emotion]
        emotion_lower = emotion.lower()
        for key, value in EMOTION_MAPPING.items():
            if key.lower() in emotion_lower or emotion_lower in key.lower():
                return value
        return "neutral"

    def _get_majority_emotion(self, emotions: List[str]) -> str:
        mapped = [self._map_emotion(e) for e in emotions]
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

            prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Identify the emotion in this audio. Choose one from: angry, happy, frustrated, sad, neutral."
            inputs = self.processor(text=prompt, audio=audio, sampling_rate=self.sample_rate, return_tensors="pt")
            
            label_text = sample["emotion"]
            labels = torch.full_like(inputs.input_ids[0], -100)
            emo_tokens = self.processor.tokenizer.encode(sample["emotion"], add_special_tokens=False)
            labels[:len(emo_tokens)] = torch.tensor(emo_tokens)
            
            ret = {
                "input_ids": inputs.input_ids[0],
                "attention_mask": inputs.attention_mask[0],
                "labels": labels
            }
            if hasattr(inputs, "audio_features"):
                ret["audio_features"] = inputs.audio_input_values[0]
            return ret
        except Exception as e:
            logger.error(f"处理样本 {sample['id']} 时出错: {str(e)}", exc_info=True)
            if len(self.samples) > 1:
                alt_idx = random.randint(0, len(self.samples) - 1)
                return self.__getitem__(alt_idx)
            else:
                return {"input_ids": torch.tensor([]), "attention_mask": torch.tensor([]), "labels": torch.tensor([])}

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
    for i, item in enumerate(batch):
        for key in ["input_ids", "attention_mask", "labels"]:
            if key in item and not isinstance(item[key], torch.Tensor):
                batch[i][key] = torch.tensor(item[key], dtype=torch.long)
        if "audio_features" in item and not isinstance(item["audio_features"], torch.Tensor):
            batch[i]["audio_features"] = torch.tensor(item["audio_features"], dtype=torch.float)
    
    data_collator = DataCollatorForSeq2Seq(
        processor.tokenizer if processor else None,
        model=model,
        padding=True,
        pad_to_multiple_of=8,
        label_pad_token_id=-100
    )
    return data_collator(batch)

def create_collator(proc, mdl):
    def collate_fn_wrapper(batch):
        return dataset_collate_fn(batch, proc, mdl)
    return collate_fn_wrapper

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
                
            prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Identify the emotion in this audio. Choose one from: angry, happy, frustrated, sad, neutral."
            inputs = processor(text=prompt, audio=audio, sampling_rate=16000, return_tensors="pt")
            
            # 确保所有输入都移到正确的设备上
            for key in inputs:
                inputs[key] = inputs[key].to(device)
                
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False
                )
                
            prediction = processor.batch_decode(output, skip_special_tokens=True)[0].strip().lower()
            
            # 提取预测的情感
            pred_emotion = "neutral"  # 默认值
            for emotion in VALID_EMOTIONS:
                if emotion in prediction.lower():
                    pred_emotion = emotion
                    break
                    
            pred_labels.append(pred_emotion)
            
            result = {
                "id": item.get('id', f"sample_{idx}"),
                "audio_path": abs_path,
                "true_emotion": true_emotion,
                "predicted_emotion": pred_emotion,
                "predicted_text": prediction,
                "correct": pred_emotion == true_emotion
            }
            results.append(result)
            
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

    logger.info("加载模型中...")
    model = Qwen2AudioForConditionalGeneration.from_pretrained(
        local_model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    # 应用 LoRA 微调
    peft_config = get_peft_config()
    model = get_peft_model(model, peft_config)
    print_trainable_parameters(model)

    # 对内部 base_model.forward 打猴子补丁，确保移除 decoder_* 参数
    old_base_forward = model.base_model.forward
    def new_base_forward(*args, **kwargs):
        kwargs.pop("decoder_input_ids", None)
        kwargs.pop("decoder_attention_mask", None)
        kwargs.pop("decoder_inputs_embeds", None)
        # 确保labels的形状与attention_mask匹配
        if "labels" in kwargs and "attention_mask" in kwargs:
            if kwargs["labels"].shape[1] != kwargs["attention_mask"].shape[1]:
                min_len = min(kwargs["labels"].shape[1], kwargs["attention_mask"].shape[1])
                kwargs["labels"] = kwargs["labels"][:, :min_len]
                kwargs["attention_mask"] = kwargs["attention_mask"][:, :min_len]
        return old_base_forward(*args, **kwargs)
    model.base_model.forward = new_base_forward

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
        learning_rate=2e-5,
        num_train_epochs=10,
        warmup_ratio=0.1,
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        save_strategy="steps",
        fp16=True,
        generation_max_length=50,
        report_to="tensorboard",
        save_total_limit=3,
        logging_first_step=True,
        predict_with_generate=False,
        include_inputs_for_metrics=True,
        remove_unused_columns=False
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=create_collator(processor, model)
    )

    logger.info("开始训练...")
    trainer.train()

    logger.info("保存微调后的模型...")
    model.save_pretrained("./qwen2_audio_emotion_final")
    logger.info("微调完成!")
    
    # 训练完成后直接评估
    logger.info("开始评估...")
    
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

if __name__ == "__main__":
    main()

# evaluate.py
import os
import torch
import numpy as np
import librosa
import json
import logging
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration
from peft import PeftModel
import pandas as pd
import pickle

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 五分类标签
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

# 评估函数
def evaluate_model(model_path, processor_path, json_file, iemocap_root, save_results=True, sample_limit=None):
    logger.info("加载模型和处理器...")
    processor = AutoProcessor.from_pretrained(processor_path, trust_remote_code=True)
    
    # 加载基础模型
    base_model = Qwen2AudioForConditionalGeneration.from_pretrained(
        "./qwen2_audio_model",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 加载微调的PEFT模型
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # 检查模型设备
    device = next(model.parameters()).device
    logger.info(f"评估模型位于设备: {device}")
    
    # 加载测试数据
    logger.info(f"从 {json_file} 加载测试数据...")
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 过滤掉标记为need_prediction的样本和不在五类中的情感
    valid_data = []
    skipped_samples = 0
    need_prediction_samples = 0
    emotion_filtered = 0
    
    for item in data:
        # # 跳过需要预测的样本
        # if item.get('need_prediction') == "yes":
        #     need_prediction_samples += 1
        #     continue
            
        # 获取情感标签
        if 'emotion' not in item:
            skipped_samples += 1
            continue
            
        emotions = item['emotion']
        if isinstance(emotions, str):
            emotions = [emotions]
        if not emotions:
            skipped_samples += 1
            continue
            
        # 获取主要情感
        mapped_emotion = get_majority_emotion(emotions)
        
        # 过滤掉不在五分类范围内的情感
        if mapped_emotion not in VALID_EMOTIONS:
            emotion_filtered += 1
            continue
            
        valid_data.append(item)
    
    logger.info(f"有效样本: {len(valid_data)}, 跳过需预测样本: {need_prediction_samples}, "
                f"跳过无情感样本: {skipped_samples}, 情感过滤: {emotion_filtered}")
    
    # 将数据分成训练集和测试集
    random_state = np.random.RandomState(42)
    indices = list(range(len(valid_data)))
    random_state.shuffle(indices)
    
    # 使用20%的数据作为测试
    test_size = int(len(valid_data) * 0.2)
    test_indices = indices[:test_size]
    
    if sample_limit is not None and sample_limit < len(test_indices):
        logger.info(f"限制测试样本数量为 {sample_limit}")
        test_indices = test_indices[:sample_limit]
    
    results = []
    true_labels = []
    pred_labels = []
    
    for idx in tqdm(test_indices, desc="评估进度"):
        item = valid_data[idx]
        
        # 获取音频路径
        rel_path = item.get('audio_path') or item.get('path') or item.get('wav_path') or item.get('audio')
        if not rel_path:
            continue
            
        # 获取真实情感标签
        if 'emotion' not in item:
            continue
            
        emotions = item['emotion']
        if isinstance(emotions, str):
            emotions = [emotions]
        
        # 获取主要情感
        true_emotion = get_majority_emotion(emotions)
        true_labels.append(true_emotion)
        
        # 加载和处理音频
        abs_path = convert_path(rel_path, iemocap_root)
        if not os.path.exists(abs_path):
            logger.warning(f"找不到音频文件: {abs_path}")
            continue
            
        try:
            # 加载音频
            audio, sr = librosa.load(abs_path, sr=16000)
            if len(audio) == 0:
                logger.warning(f"音频数据为空: {abs_path}")
                continue
                
            # 限制音频长度
            max_length = 30 * 16000  # 30秒
            if len(audio) > max_length:
                audio = audio[:max_length]
                
            # 准备输入
            prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Identify the emotion in this audio. Choose one from: angry, happy, frustrated, sad, neutral."
            inputs = processor(text=prompt, audio=audio, sampling_rate=16000, return_tensors="pt")
            
            # 将输入移到正确的设备上
            for key in inputs:
                inputs[key] = inputs[key].to(device)
                
            # 生成预测
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=20,
                    do_sample=False
                )
                
            # 解码预测结果
            prediction = processor.batch_decode(output, skip_special_tokens=True)[0].strip().lower()
            
            # 提取预测的情感
            pred_emotion = "neutral"  # 默认值
            for emotion in VALID_EMOTIONS:
                if emotion in prediction.lower():
                    pred_emotion = emotion
                    break
            
            pred_labels.append(pred_emotion)
            
            # 保存结果
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
    logger.info(f"宏平均F1: {report['macro avg']['f1-score']:.4f}")
    logger.info(f"加权平均F1: {report['weighted avg']['f1-score']:.4f}")
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
            f.write(f"准确率: {accuracy:.4f}\n")
            f.write(f"宏平均F1: {report['macro avg']['f1-score']:.4f}\n")
            f.write(f"加权平均F1: {report['weighted avg']['f1-score']:.4f}\n\n")
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

# 辅助函数
def map_emotion(emotion: str) -> str:
    # 如果是单词中包含情感，提取情感名词
    emotion = emotion.lower().strip()
    
    # 直接匹配标准情感
    for standard_emotion in VALID_EMOTIONS:
        if standard_emotion in emotion:
            return standard_emotion
    
    # 通过映射表匹配
    for key, value in EMOTION_MAPPING.items():
        if key.lower() in emotion:
            return value
    
    # 默认返回中性
    return "neutral"

def get_majority_emotion(emotions: List[str]) -> str:
    mapped = [map_emotion(e) for e in emotions]
    counts = {}
    for emo in mapped:
        counts[emo] = counts.get(emo, 0) + 1
    return max(counts.items(), key=lambda x: x[1])[0]

def convert_path(rel_path: str, iemocap_root: str) -> str:
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        if rel_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
            return rel_path
        else:
            wav_path = os.path.splitext(rel_path)[0] + '.wav'
            if os.path.exists(wav_path):
                return wav_path
    
    # 规范化路径
    rel_path = os.path.normpath(rel_path.replace('/', os.sep).replace('\\', os.sep))
    base_path = iemocap_root
    
    # 首先尝试直接组合路径
    abs_path = os.path.join(base_path, rel_path)
    if os.path.exists(abs_path) and abs_path.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
        return abs_path
    
    # 尝试查找相同名称但带有.wav扩展名的文件
    wav_path = os.path.splitext(abs_path)[0] + '.wav'
    if os.path.exists(wav_path):
        return wav_path
    
    # 尝试在会话目录中查找文件
    basename = os.path.basename(rel_path)
    basename_noext = os.path.splitext(basename)[0]
    
    for session_num in range(1, 6):
        session_dir = os.path.join(base_path, "IEMOCAP_full_release", f"Session{session_num}")
        if os.path.exists(session_dir):
            for root, _, files in os.walk(session_dir):
                for file in files:
                    if file.startswith(basename_noext) and file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                        return os.path.join(root, file)
    
    # 如果找不到，返回加了.wav的原始路径
    return os.path.splitext(abs_path)[0] + '.wav'

if __name__ == "__main__":
    model_path = "./qwen2_audio_emotion_final"  # 微调后的模型路径
    processor_path = "./qwen2_audio_processor"  # 处理器路径
    json_file = "iemocap_ambiguous.json"        # 数据集JSON文件
    iemocap_root = r"C:\Users\luoya\Desktop\unimelb-research\dataset\IEMOCAP"  # IEMOCAP数据集根目录
    
    # 评估模型并保存结果
    accuracy, report, results, metrics = evaluate_model(
        model_path=model_path,
        processor_path=processor_path,
        json_file=json_file,
        iemocap_root=iemocap_root,
        save_results=True,
        sample_limit=None  # 设置为None以评估所有测试样本
    )
    
    logger.info(f"评估完成! 准确率: {accuracy:.4f}")
    logger.info(f"宏平均F1: {metrics['report']['macro avg']['f1-score']:.4f}")
    logger.info(f"加权平均F1: {metrics['report']['weighted avg']['f1-score']:.4f}")
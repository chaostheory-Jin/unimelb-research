import os
import json
import torch
import logging
import librosa
from pathlib import Path
from tqdm import tqdm

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置IEMOCAP数据集的基础路径
IEMOCAP_BASE_PATH = r"C:\Users\luoya\Desktop\unimelb-research\dataset\IEMOCAP"

# 将相对路径转换为绝对路径的函数
def convert_path(rel_path, base_path=IEMOCAP_BASE_PATH):
    """将相对路径转换为绝对路径"""
    # 如果已经是绝对路径，直接返回
    if os.path.isabs(rel_path):
        return rel_path
    
    # 修正基础路径，确保包含IEMOCAP_full_release子目录
    full_base_path = base_path
    if "IEMOCAP_full_release" not in base_path:
        full_base_path = os.path.join(base_path, "IEMOCAP_full_release")
    
    # 尝试不同的组合方式
    potential_paths = [
        # 直接连接
        os.path.join(full_base_path, rel_path),
        # 连接到基础路径（不含IEMOCAP_full_release）
        os.path.join(base_path, rel_path),
        # 去掉可能的前导斜杠
        os.path.join(full_base_path, rel_path.lstrip('/')),
        # 仅使用文件名
        os.path.join(full_base_path, os.path.basename(rel_path))
    ]
    
    # 处理Session格式路径
    if "Session" in rel_path:
        parts = rel_path.split(os.sep)
        session_found = False
        
        # 查找Session部分
        for i, part in enumerate(parts):
            if part.startswith("Session"):
                session_found = True
                
                # 1. 尝试从Session开始的路径
                session_path = os.path.join(full_base_path, *parts[i:])
                potential_paths.append(session_path)
                
                # 2. 尝试直接用会话文件夹加后续路径
                session_direct_path = os.path.join(full_base_path, part, *parts[i+1:])
                potential_paths.append(session_direct_path)
                
                break
        
        # 如果路径中没有Session但包含Ses前缀（如Ses01F_impro01）
        if not session_found:
            for i, part in enumerate(parts):
                if part.startswith("Ses"):
                    # 提取会话编号 (例如，从"Ses01F_impro01"中提取"1")
                    try:
                        sess_num = int(part[3:5])
                        session_folder = f"Session{sess_num}"
                        
                        # 构建修正路径
                        if i > 0:
                            # 如果Ses不是路径的第一部分
                            corrected_path = os.path.join(full_base_path, session_folder, *parts[:i], *parts[i:])
                            potential_paths.append(corrected_path)
                        
                        # 另一种常见的格式: Session{n}/sentences/wav/Ses{n}X_xxx
                        wav_path = os.path.join(full_base_path, session_folder, "sentences", "wav", part)
                        potential_paths.append(wav_path)
                        
                        # 处理形如 Ses01F_impro01/Ses01F_impro01_F001.wav 的情况
                        if len(parts) > i+1 and parts[i+1].startswith(part):
                            wav_file_path = os.path.join(full_base_path, session_folder, "sentences", "wav", part, parts[i+1])
                            potential_paths.append(wav_file_path)
                    except:
                        pass
                    break
    
    # 处理特定的结构，如果路径中包含特定目录
    if "sentences/wav" in rel_path or "sentences\\wav" in rel_path:
        # 先尝试提取会话标识符 (Ses01F 等)
        ses_parts = rel_path.split(os.sep)
        for part in ses_parts:
            if part.startswith("Ses") and len(part) >= 5:
                try:
                    sess_num = int(part[3:5])
                    session_folder = f"Session{sess_num}"
                    
                    # 构建 Session{n}/sentences/wav/... 路径
                    idx = rel_path.find(part)
                    if idx >= 0:
                        rel_suffix = rel_path[idx:]
                        corrected_path = os.path.join(full_base_path, session_folder, "sentences", "wav", rel_suffix)
                        potential_paths.append(corrected_path)
                except:
                    pass
    
    # 检查每个可能的路径
    for path in potential_paths:
        if os.path.exists(path):
            return path
    
    # 如果找不到，进行更深入的搜索
    basename = os.path.basename(rel_path)
    if basename.endswith('.wav'):
        # 递归搜索wav文件
        for root, _, files in os.walk(full_base_path):
            if basename in files:
                return os.path.join(root, basename)
    
    # 如果无法找到匹配的文件，记录错误并返回原始路径
    logger.warning(f"无法找到文件: {rel_path}")
    return rel_path

# 设置情绪标签映射
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

def map_emotion(emotion):
    """更灵活的情绪标签映射"""
    if emotion in EMOTION_MAPPING:
        return EMOTION_MAPPING[emotion]
    
    # 尝试模糊匹配
    emotion_lower = emotion.lower()
    for key, value in EMOTION_MAPPING.items():
        if key.lower() in emotion_lower or emotion_lower in key.lower():
            return value
    
    logger.warning(f"未知情感标签: {emotion}")
    return "neutral"  # 默认返回

def get_majority_emotion(emotions):
    """获取多数投票的情绪标签"""
    if not emotions:
        return "neutral"
    
    emotion_counts = {}
    for emotion in emotions:
        mapped_emotion = map_emotion(emotion)
        emotion_counts[mapped_emotion] = emotion_counts.get(mapped_emotion, 0) + 1
    
    # 获取出现次数最多的情绪
    majority_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
    return majority_emotion

def test_json_dataset(json_file_path):
    """测试从JSON文件加载数据集"""
    logger.info(f"从JSON文件加载数据集: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"成功加载JSON文件，包含 {len(data)} 个样本")
    except Exception as e:
        logger.error(f"加载JSON文件失败: {e}")
        return False
    
    # 检查数据格式
    if not isinstance(data, list):
        logger.error(f"JSON数据格式错误: 应为列表，实际为 {type(data)}")
        return False
    
    valid_samples = []
    invalid_samples = []
    
    # 检查每个样本
    for i, item in enumerate(data):
        if i < 5:
            logger.info(f"样本 {i+1}: {item}")
        
        # 检查样本格式
        if not isinstance(item, dict):
            logger.warning(f"样本 {i+1} 不是字典类型: {type(item)}")
            invalid_samples.append({"index": i, "reason": "非字典类型"})
            continue
        
        # 检查ID字段
        if 'id' not in item:
            logger.warning(f"样本 {i+1} 缺少ID字段")
            invalid_samples.append({"index": i, "reason": "缺少ID字段"})
            continue
        
        # 检查情绪字段
        if 'emotion' not in item:
            logger.warning(f"样本 {i+1} (ID: {item.get('id')}) 缺少emotion字段")
            invalid_samples.append({"index": i, "reason": "缺少emotion字段", "id": item.get('id')})
            continue
        
        # 检查音频路径 (可能存在于不同字段)
        rel_audio_path = None
        if 'audio_path' in item:
            rel_audio_path = item['audio_path']
        elif 'path' in item:
            rel_audio_path = item['path']
        elif 'wav_path' in item:
            rel_audio_path = item['wav_path']
        
        if not rel_audio_path:
            logger.warning(f"样本 {i+1} (ID: {item.get('id')}) 缺少音频路径字段")
            invalid_samples.append({"index": i, "reason": "缺少音频路径字段", "id": item.get('id')})
            continue
        
        # 转换为绝对路径
        audio_path = convert_path(rel_audio_path)
        
        # 检查音频文件是否存在
        if not os.path.exists(audio_path):
            logger.warning(f"样本 {i+1} (ID: {item.get('id')}) 的音频文件不存在: {rel_audio_path} -> {audio_path}")
            invalid_samples.append({"index": i, "reason": "音频文件不存在", "id": item.get('id'), "path": audio_path})
            continue
        
        # 获取情绪标签
        emotions = item['emotion']
        if isinstance(emotions, str):
            emotions = [emotions]
        
        emotion = get_majority_emotion(emotions)
        
        # 到这里，样本有效
        valid_samples.append({
            "id": item['id'],
            "audio_path": audio_path,  # 使用转换后的绝对路径
            "rel_audio_path": rel_audio_path,  # 保留原始相对路径
            "emotion": emotion,
            "original_emotion": emotions
        })
    
    # 统计情绪分布
    emotion_counts = {}
    for sample in valid_samples:
        emotion = sample["emotion"]
        emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
    
    logger.info("\n情绪分布:")
    for emotion, count in emotion_counts.items():
        logger.info(f"  {emotion}: {count} 样本 ({count/len(valid_samples)*100:.1f}%)")
    
    # 检查音频加载
    logger.info("\n测试音频加载:")
    audio_test_samples = valid_samples[:3]  # 只测试前3个样本
    
    for sample in audio_test_samples:
        audio_path = sample["audio_path"]
        logger.info(f"加载音频文件: {audio_path}")
        
        try:
            # 尝试加载音频
            audio, sr = librosa.load(audio_path, sr=16000)
            logger.info(f"  成功加载，样本率: {sr}, 长度: {len(audio)}, 持续时间: {len(audio)/sr:.2f}秒")
        except Exception as e:
            logger.error(f"  加载失败: {e}")
    
    # 结果摘要
    logger.info("\n结果摘要:")
    logger.info(f"总样本数: {len(data)}")
    logger.info(f"有效样本数: {len(valid_samples)}")
    logger.info(f"无效样本数: {len(invalid_samples)}")
    
    if invalid_samples:
        logger.info("\n部分无效样本:")
        for i, sample in enumerate(invalid_samples[:5]):
            logger.info(f"  无效样本 {i+1}: {sample}")
    
    return len(valid_samples) > 0

def test_finetune_loading(json_file_path):
    """测试用于微调的数据集加载"""
    logger.info(f"\n测试微调数据加载流程，使用JSON文件: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 预处理：转换路径
        processed_data = []
        path_conversion_failures = 0
        
        for item in data:
            rel_audio_path = item.get('audio_path') or item.get('path') or item.get('wav_path')
            if not rel_audio_path:
                continue
            
            # 转换为绝对路径
            audio_path = convert_path(rel_audio_path)
            
            # 检查转换是否成功
            if os.path.exists(audio_path):
                # 创建新的条目
                new_item = item.copy()
                new_item['audio_path'] = audio_path
                new_item['rel_audio_path'] = rel_audio_path
                processed_data.append(new_item)
                
                # 打印前几个成功的路径转换，用于验证
                if len(processed_data) <= 5:
                    logger.info(f"路径转换成功: {rel_audio_path} -> {audio_path}")
            else:
                path_conversion_failures += 1
                if path_conversion_failures <= 5:
                    logger.warning(f"路径转换失败: {rel_audio_path} -> {audio_path}")
        
        logger.info(f"路径转换结果: 成功={len(processed_data)}, 失败={path_conversion_failures}")
        
        # 划分训练集和验证集
        from sklearn.model_selection import train_test_split
        train_data, val_data = train_test_split(processed_data, test_size=0.2, random_state=42)
        
        logger.info(f"总样本数: {len(processed_data)}")
        logger.info(f"训练集样本数: {len(train_data)}")
        logger.info(f"验证集样本数: {len(val_data)}")
        
        # 模拟数据处理
        class MockProcessor:
            def __init__(self):
                self.feature_extractor = type('obj', (object,), {'sampling_rate': 16000})
            
            def __call__(self, text=None, audio=None, sampling_rate=None, return_tensors=None):
                # 模拟处理器行为
                return type('obj', (object,), {
                    'input_ids': torch.ones((1, 10)),
                    'attention_mask': torch.ones((1, 10)),
                    'audio_input_values': torch.randn((1, 1000)),
                    'audio_attention_mask': torch.ones((1, 1000))
                })
        
        processor = MockProcessor()
        
        # 测试数据处理
        processed_train = []
        processed_val = []
        
        logger.info("测试训练数据处理...")
        for item in tqdm(train_data[:5]):  # 只处理前5个
            try:
                audio_path = item['audio_path']
                if not audio_path or not os.path.exists(audio_path):
                    logger.warning(f"音频文件不存在: {audio_path}")
                    continue
                
                # 加载音频
                audio, sr = librosa.load(audio_path, sr=processor.feature_extractor.sampling_rate)
                
                # 处理输入
                emotions = item['emotion']
                if isinstance(emotions, str):
                    emotions = [emotions]
                
                emotion = get_majority_emotion(emotions)
                
                # 模拟处理
                processed_sample = {
                    "id": item['id'],
                    "audio_path": audio_path,
                    "emotion": emotion,
                    "input_processed": True
                }
                
                processed_train.append(processed_sample)
                
            except Exception as e:
                logger.error(f"处理样本失败: {e}")
        
        logger.info(f"成功处理 {len(processed_train)} 个训练样本")
        
        # 显示结果
        return len(processed_train) > 0
        
    except Exception as e:
        logger.error(f"测试微调数据加载失败: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        # 默认路径，请根据需要修改
        json_file_path = "iemocap_ambiguous.json"
    
    logger.info(f"开始测试JSON数据集: {json_file_path}")
    
    if os.path.exists(json_file_path):
        if test_json_dataset(json_file_path):
            logger.info("JSON数据集测试通过")
            
            # 测试微调数据加载
            if test_finetune_loading(json_file_path):
                logger.info("微调数据加载测试通过")
            else:
                logger.error("微调数据加载测试失败")
        else:
            logger.error("JSON数据集测试失败")
    else:
        logger.error(f"JSON文件不存在: {json_file_path}")
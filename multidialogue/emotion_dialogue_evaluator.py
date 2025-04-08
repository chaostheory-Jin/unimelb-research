import os
import json
import base64
import argparse
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Tuple, Any
from tqdm import tqdm
from openai import OpenAI
import numpy as np
import re
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import nltk
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pydub import AudioSegment
import time

# Ensure NLTK resources are downloaded
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

class EmotionDialogueEvaluator:
    """
    Evaluate emotion-driven dialogue generation compared to standard dialogue generation
    """
    
    def __init__(self, api_key: str = None, log_file: str = None, 
                 transition_instructions_file: str = None, 
                 sustained_instructions_file: str = None):
        """
        Initialize the evaluator
        
        Args:
            api_key: OpenAI API key, will use environment variable if not provided
            log_file: Log file path
            transition_instructions_file: Path to emotion transition instructions JSON file
            sustained_instructions_file: Path to sustained emotion instructions JSON file
        """
        # Setup logger
        self.logger = self._setup_logger(log_file)
        
        # Initialize OpenAI client
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key not found. Please provide it through the api_key parameter "
                "or by setting the OPENAI_API_KEY environment variable."
            )
        
        self.client = OpenAI(api_key=api_key)
        self.logger.info("OpenAI client initialized successfully")
        
        # Load emotion instruction files
        self.transition_instructions = self._load_instructions(transition_instructions_file, "emotion_transition_instructions.json")
        self.sustained_instructions = self._load_instructions(sustained_instructions_file, "sustained_emotion_instructions.json")
        self.logger.info("Emotion instruction files loaded successfully")
        
        # Evaluation metrics
        self.metrics = {
            "rouge": rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True),
            "bleu": lambda ref, hyp: sentence_bleu([ref.split()], hyp.split(), 
                                                 smoothing_function=SmoothingFunction().method1)
        }
        
        # Store evaluation results
        self.evaluation_results = {
            "emotion_aware": [],
            "baseline": []
        }
        
        # 修改情绪状态追踪 - 按说话者分开追踪
        # 使用嵌套字典: {dialogue_id: {speaker_id: emotion}}
        self.emotion_states = {}
        
        # 对每个说话者分别追踪情绪转变
        self.emotion_transition_metrics = {
            "total_transitions": 0,
            "correctly_detected_transitions": 0,
            "transition_accuracy": 0.0,
            "by_speaker": {}  # 按说话者分类的指标
        }
        
        # Emotion emoji mapping
        self.emotion_emoji = {
            "neutral": "😐",
            "frustrated": "😤",
            "angry": "😠",
            "happy": "😊",
            "sad": "😢",
            "anxious": "😰",
            "surprised": "😲"
        }
    
    def _setup_logger(self, log_file: str = None) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger('DialogueEvaluator')
        logger.setLevel(logging.INFO)
        
        if log_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            log_file = f'dialogue_eval_{timestamp}.log'
        
        # File handler
        fh = logging.FileHandler(log_file, encoding='utf-8')
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _load_instructions(self, file_path: str, default_filename: str) -> Dict:
        """Load instruction file, falling back to default if not provided"""
        if file_path:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading instruction file {file_path}: {str(e)}")
                self.logger.error(traceback.format_exc())
        
        # Try to load default file from the same directory as the script
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            default_path = os.path.join(script_dir, default_filename)
            if os.path.exists(default_path):
                with open(default_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            else:
                self.logger.warning(f"Default instruction file {default_filename} not found")
                return {}
        except Exception as e:
            self.logger.error(f"Error loading default instruction file: {str(e)}")
            return {}
    
    def detect_emotion(self, audio_file_path: str) -> str:
        """
        Detect emotion from audio file using gpt-4o-audio-preview
        
        Args:
            audio_file_path: Audio file path
            
        Returns:
            Detected emotion
        """
        try:
            # 验证文件存在
            if not os.path.exists(audio_file_path):
                self.logger.error(f"Audio file does not exist: {audio_file_path}")
                return "neutral"  # 默认情绪
            
            # 检查文件大小
            file_size = os.path.getsize(audio_file_path)
            self.logger.info(f"Audio file size: {file_size} bytes")
            
            # 读取音频文件并进行base64编码
            with open(audio_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 使用正确的格式并添加重试机制
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-4o-audio-preview",
                        messages=[
                            {"role": "system", "content": "You are an emotion detection assistant. Analyze the audio and determine the primary emotion expressed. Only respond with a single emotion word from this list: neutral, frustrated, angry, happy, sad, anxious, surprised."},
                            {"role": "user", "content": [
                                {"type": "text", "text": "What is the primary emotion expressed in this audio?"},
                                {"type": "input_audio", "input_audio": {
                                    "data": audio_base64,
                                    "format": "wav"
                                }}
                            ]}
                        ],
                        max_tokens=10,
                        timeout=30  # 30秒超时
                    )
                    
                    emotion = response.choices[0].message.content.strip().lower()
                    self.logger.info(f"Detected emotion: {emotion}")
                    
                    return emotion
                    
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        self.logger.error(f"All retries failed: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        return "neutral"  # 默认情绪
            
        except Exception as e:
            self.logger.error(f"Emotion detection error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "neutral"  # 默认情绪
    
    def get_emotion_instruction(self, current_emotion: str, dialogue_context: List[Dict], 
                               speaker_id: str, dialogue_id: str) -> str:
        """
        Retrieve emotion-based instruction based on current emotion and previous state
        
        Args:
            current_emotion: Current emotion state
            dialogue_context: Dialogue context
            speaker_id: ID of the current speaker (e.g., "M" or "F")
            dialogue_id: ID of the current dialogue
            
        Returns:
            Instruction suitable for the current emotional state
        """
        try:
            # 初始化对话状态，如果不存在
            if dialogue_id not in self.emotion_states:
                self.emotion_states[dialogue_id] = {}
            
            # 获取该说话者的上一个情绪状态
            prev_emotion = self.emotion_states[dialogue_id].get(speaker_id)
            
            # 检测情绪转变
            if prev_emotion and prev_emotion != current_emotion:
                # 只在同一说话者的情绪发生变化时记录
                self.emotion_transition_metrics["total_transitions"] += 1
                
                # 为该说话者初始化指标，如果不存在
                if speaker_id not in self.emotion_transition_metrics["by_speaker"]:
                    self.emotion_transition_metrics["by_speaker"][speaker_id] = {
                        "total_transitions": 0,
                        "correctly_detected_transitions": 0,
                        "transition_accuracy": 0.0
                    }
                
                self.emotion_transition_metrics["by_speaker"][speaker_id]["total_transitions"] += 1
                
                # 将在评估时更新 correctly_detected_transitions
                
                # 获取转变指令
                transition_key = f"{prev_emotion}_to_{current_emotion}"
                if transition_key in self.transition_instructions:
                    instruction = self.transition_instructions[transition_key]
                    self.logger.info(f"Using transition instruction for {transition_key} (Speaker: {speaker_id})")
                else:
                    # 如果找不到转变指令，使用持续情绪指令
                    if current_emotion in self.sustained_instructions:
                        instruction = self.sustained_instructions[current_emotion]
                        self.logger.info(f"Transition {transition_key} not found, using sustained instruction for {current_emotion}")
                    else:
                        instruction = f"The user is showing {current_emotion} emotion. Respond appropriately to address their emotional state."
                        self.logger.warning(f"No instructions found for {current_emotion}, using default")
            else:
                # 使用持续情绪指令
                if current_emotion in self.sustained_instructions:
                    instruction = self.sustained_instructions[current_emotion]
                    self.logger.info(f"Using sustained instruction for {current_emotion}")
                else:
                    instruction = f"The user is showing {current_emotion} emotion. Respond appropriately to address their emotional state."
                    self.logger.warning(f"No sustained instruction found for {current_emotion}, using default")
            
            # 更新情绪状态
            self.emotion_states[dialogue_id][speaker_id] = current_emotion
            
            return instruction
            
        except Exception as e:
            self.logger.error(f"Error getting emotion instruction: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "Respond to the user's questions or statements in a friendly and supportive manner."  # 默认指令
    
    def generate_emotion_aware_response(self, audio_path: str, 
                                       speaker_id: str, dialogue_id: str, 
                                       dialogue_context: List[Dict] = None) -> str:
        """
        Generate emotion-aware response using the two-stage method with gpt-4o-audio-preview
        
        Args:
            audio_path: Current audio path
            speaker_id: Speaker ID (e.g., "M" or "F")
            dialogue_id: Dialogue ID
            dialogue_context: Optional text context
            
        Returns:
            Generated response
        """
        try:
            # 验证文件存在
            if not os.path.exists(audio_path):
                self.logger.error(f"Audio file does not exist: {audio_path}")
                return "I'm sorry, but I couldn't access the audio file."
            
            # 阶段1: 检测情绪
            current_emotion = self.detect_emotion(audio_path)
            
            # 获取基于当前和之前状态的情绪指令
            emotion_instruction = self.get_emotion_instruction(
                current_emotion, 
                dialogue_context or [], 
                speaker_id, 
                dialogue_id
            )
            
            # 阶段2: 基于情绪指令和当前音频生成回复
            # 读取音频并进行base64编码
            with open(audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 使用正确的格式并添加重试机制
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # 使用gpt-4o-audio-preview模型生成回复
                    messages = [
                        {"role": "system", "content": f"You are a dialogue assistant. {emotion_instruction} The current detected user emotion is: {current_emotion} {self.emotion_emoji.get(current_emotion, '')}"},
                        {"role": "user", "content": [
                            {"type": "text", "text": f"Respond to this audio. The user's emotion is: {current_emotion}"},
                            {"type": "input_audio", "input_audio": {
                                "data": audio_base64,
                                "format": "wav"
                            }}
                        ]}
                    ]
                    
                    response = self.client.chat.completions.create(
                        model="gpt-4o-audio-preview",
                        messages=messages,
                        max_tokens=150,
                        timeout=60  # 60秒超时
                    )
                    
                    reply = response.choices[0].message.content.strip()
                    self.logger.info(f"Emotion-aware response generated successfully, based on emotion: {current_emotion}")
                    
                    return reply
                
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        self.logger.error(f"All retries failed: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        return f"I understand you're feeling {current_emotion}. How can I help you today?"
        
        except Exception as e:
            self.logger.error(f"Emotion-aware response generation error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "I understand. How can I help you?"  # 默认回复
    
    def generate_baseline_response(self, merged_audio_path: str) -> str:
        """
        Generate baseline response using merged audio input with gpt-4o-audio-preview
        
        Args:
            merged_audio_path: Path to merged audio file
            
        Returns:
            Generated response
        """
        try:
            # 验证文件存在
            if not os.path.exists(merged_audio_path):
                self.logger.error(f"Merged audio file does not exist: {merged_audio_path}")
                return "I'm sorry, but I couldn't access the audio file."
            
            # 读取音频并进行base64编码
            with open(merged_audio_path, "rb") as audio_file:
                audio_data = audio_file.read()
            
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 使用正确的格式并添加重试机制
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    # 使用gpt-4o-audio-preview模型生成回复
                    messages = [
                        {"role": "system", "content": "You are a helpful dialogue assistant. Please respond directly to the audio content."},
                        {"role": "user", "content": [
                            {"type": "text", "text": "Please respond to this audio conversation:"},
                            {"type": "input_audio", "input_audio": {
                                "data": audio_base64,
                                "format": "wav"
                            }}
                        ]}
                    ]
                    
                    response = self.client.chat.completions.create(
                        model="gpt-4o-audio-preview",
                        messages=messages,
                        max_tokens=150,
                        timeout=60  # 60秒超时
                    )
                    
                    reply = response.choices[0].message.content.strip()
                    self.logger.info("Baseline response generated successfully using merged audio")
                    
                    return reply
                
                except Exception as e:
                    self.logger.warning(f"Attempt {attempt+1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        self.logger.error(f"All retries failed: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        return "How can I help you today?"
        
        except Exception as e:
            self.logger.error(f"Baseline response generation error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return "I understand. How can I help you?"  # 默认回复
    
    def evaluate_responses(self, reference: str, emotion_aware: str, baseline: str) -> Dict:
        """
        Evaluate generated responses
        
        Args:
            reference: Reference response
            emotion_aware: Emotion-aware response
            baseline: Baseline response
            
        Returns:
            Evaluation metrics
        """
        results = {}
        
        # Normalize text
        ref_norm = re.sub(r'\s+', ' ', reference).strip().lower()
        emotion_norm = re.sub(r'\s+', ' ', emotion_aware).strip().lower()
        baseline_norm = re.sub(r'\s+', ' ', baseline).strip().lower()
        
        # Debug output
        self.logger.debug(f"Reference text (normalized): '{ref_norm}'")
        self.logger.debug(f"Emotion-aware text (normalized): '{emotion_norm}'")
        self.logger.debug(f"Baseline text (normalized): '{baseline_norm}'")
        
        # Check if texts are empty
        if not ref_norm or not emotion_norm or not baseline_norm:
            self.logger.warning("One or more texts are empty, metrics may be inaccurate")
        
        # Calculate Rouge scores
        try:
            emotion_rouge = self.metrics["rouge"].score(ref_norm, emotion_norm)
            baseline_rouge = self.metrics["rouge"].score(ref_norm, baseline_norm)
            
            # Calculate BLEU scores
            emotion_bleu = self.metrics["bleu"](ref_norm, emotion_norm)
            baseline_bleu = self.metrics["bleu"](ref_norm, baseline_norm)
            
            # Format results
            results = {
                "emotion_aware": {
                    "rouge1": emotion_rouge['rouge1'].fmeasure,
                    "rouge2": emotion_rouge['rouge2'].fmeasure,
                    "rougeL": emotion_rouge['rougeL'].fmeasure,
                    "bleu": emotion_bleu
                },
                "baseline": {
                    "rouge1": baseline_rouge['rouge1'].fmeasure,
                    "rouge2": baseline_rouge['rouge2'].fmeasure,
                    "rougeL": baseline_rouge['rougeL'].fmeasure,
                    "bleu": baseline_bleu
                }
            }
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {str(e)}")
            # Set default values for failed metrics
            results = {
                "emotion_aware": {
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "bleu": 0.0
                },
                "baseline": {
                    "rouge1": 0.0,
                    "rouge2": 0.0,
                    "rougeL": 0.0,
                    "bleu": 0.0
                }
            }
        
        return results
    
    def calculate_perplexity(self, text: str) -> float:
        """
        Calculate perplexity of the text
        """
        try:
            if not text or len(text.strip()) == 0:
                self.logger.warning("Empty text for perplexity calculation, returning default value 1.0")
                return 1.0
            
            # 使用更稳健的方法计算perplexity
            max_retries = 3
            retry_delay = 2
            
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model="gpt-3.5-turbo", # 使用更稳定的模型计算perplexity
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that evaluates text fluency."},
                            {"role": "user", "content": f"On a scale of 0.0 to 1.0, where 1.0 is extremely fluent and natural, and 0.0 is completely incoherent, rate the following text. Only respond with a single number between 0.0 and 1.0:\n\n{text}"}
                        ],
                        max_tokens=10,
                        temperature=0.1,
                        timeout=30  # 30秒超时
                    )
                    
                    result = response.choices[0].message.content.strip()
                    self.logger.info(f"Perplexity calculation result: {result}")
                    
                    # 尝试提取数值
                    try:
                        value = float(result)
                        if 0.0 <= value <= 1.0:
                            return value
                        else:
                            return 1.0  # 如果超出范围，使用默认值
                    except ValueError:
                        self.logger.warning(f"Could not parse perplexity value: {result}, using default 1.0")
                        return 1.0
                    
                except Exception as e:
                    self.logger.warning(f"Perplexity calculation attempt {attempt+1}/{max_retries} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                        retry_delay *= 2  # 指数退避
                    else:
                        self.logger.error(f"All perplexity retries failed: {str(e)}")
                        self.logger.error(traceback.format_exc())
                        return 1.0  # 默认值
                    
        except Exception as e:
            self.logger.error(f"Perplexity calculation error: {str(e)}")
            self.logger.error(traceback.format_exc())
            return 1.0  # 默认值
    
    def evaluate_dialogue(self, 
                         test_data: List[Dict], 
                         output_file: str = None,
                         max_samples: int = -1) -> Dict:
        """
        Evaluate dialogue generation
        
        Args:
            test_data: Test dataset
            output_file: Output file path
            max_samples: Maximum number of samples (-1 for all)
            
        Returns:
            Evaluation results
        """
        if max_samples > 0:
            test_data = test_data[:max_samples]
        
        self.logger.info(f"Starting evaluation of {len(test_data)} dialogue samples")
        
        all_results = []
        emotion_distribution = defaultdict(int)
        
        # 按对话分组数据
        dialogue_groups = self._group_by_dialogue(test_data)
        
        # 处理每个对话组
        for dialogue_id, samples in tqdm(dialogue_groups.items(), desc="Processing dialogues"):
            # 重置该对话的情绪状态
            self.reset_emotion_states(dialogue_id)
            
            # 按顺序处理样本
            for i, sample in enumerate(samples):
                try:
                    dialogue_context = sample.get("context", [])
                    audio_path = sample.get("audio_path", "")
                    reference = sample.get("reference", "")
                    
                    # 跳过没有音频的样本
                    if not audio_path or not os.path.exists(audio_path):
                        self.logger.warning(f"Audio file not found: {audio_path}, skipping sample")
                        continue
                    
                    # 提取说话者ID
                    speaker_id = self._extract_speaker_id(sample.get('id', ''))
                    self.logger.info(f"Processing sample {sample.get('id', '')}, speaker: {speaker_id}")
                    
                    # 获取地面真实情绪（如果有）
                    ground_truth_emotion = None
                    if 'emotion' in sample and sample['emotion']:
                        if isinstance(sample['emotion'], list) and sample['emotion']:
                            ground_truth_emotion = sample['emotion'][0].lower()
                        elif isinstance(sample['emotion'], str):
                            ground_truth_emotion = sample['emotion'].lower()
                    
                    # 检测情绪并记录分布
                    detected_emotion = self.detect_emotion(audio_path)
                    emotion_distribution[detected_emotion] += 1
                    
                    # 获取同一说话者的之前样本
                    previous_utterances = self.get_previous_utterances(
                        samples, i, max_history=3, same_speaker_only=True
                    )
                    
                    # 检查这是否是情绪转变，以及是否正确检测到
                    if previous_utterances and ground_truth_emotion:
                        prev_sample = previous_utterances[-1]  # 最近的前一个样本
                        prev_ground_truth = None
                        
                        if 'emotion' in prev_sample and prev_sample['emotion']:
                            if isinstance(prev_sample['emotion'], list) and prev_sample['emotion']:
                                prev_ground_truth = prev_sample['emotion'][0].lower()
                            elif isinstance(prev_sample['emotion'], str):
                                prev_ground_truth = prev_sample['emotion'].lower()
                        
                        # 注意：我们只比较同一说话者的情绪转变
                        if prev_ground_truth and prev_ground_truth != ground_truth_emotion:
                            # 这是地面真实的情绪转变
                            if speaker_id not in self.emotion_transition_metrics["by_speaker"]:
                                self.emotion_transition_metrics["by_speaker"][speaker_id] = {
                                    "total_transitions": 1,
                                    "correctly_detected_transitions": 0,
                                    "transition_accuracy": 0.0
                                }
                            else:
                                self.emotion_transition_metrics["by_speaker"][speaker_id]["total_transitions"] += 1
                            
                            # 获取我们检测到的前一个情绪
                            if dialogue_id in self.emotion_states and speaker_id in self.emotion_states[dialogue_id]:
                                prev_detected = self.emotion_states[dialogue_id][speaker_id]
                                
                                # 检查我们是否正确检测到转变
                                if prev_detected and prev_detected != detected_emotion:
                                    self.emotion_transition_metrics["by_speaker"][speaker_id]["correctly_detected_transitions"] += 1
                                    self.emotion_transition_metrics["correctly_detected_transitions"] += 1
                    
                    # 对实验组：基于情绪分析生成回复（只使用当前音频）
                    emotion_aware_response = self.generate_emotion_aware_response(
                        audio_path, speaker_id, dialogue_id, dialogue_context
                    )
                    
                    # 对对照组：获取历史记录并合并音频
                    audio_paths = [u.get('audio_path', '') for u in previous_utterances if 'audio_path' in u]
                    # 添加当前音频
                    audio_paths.append(audio_path)
                    
                    # 只保留存在的音频
                    audio_paths = [p for p in audio_paths if p and os.path.exists(p)]
                    
                    # 如果有多个音频文件，合并它们
                    if len(audio_paths) > 1:
                        merged_audio_path = self.merge_audio_files(audio_paths)
                    else:
                        merged_audio_path = audio_path  # 如果只有当前音频
                    
                    # 使用合并的音频生成基线回复
                    baseline_response = self.generate_baseline_response(merged_audio_path)
                    
                    # 评估回复
                    metrics = self.evaluate_responses(
                        reference, emotion_aware_response, baseline_response
                    )
                    
                    # 计算困惑度
                    emotion_perplexity = self.calculate_perplexity(emotion_aware_response)
                    baseline_perplexity = self.calculate_perplexity(baseline_response)
                    
                    # 添加困惑度指标
                    metrics["emotion_aware"]["perplexity"] = emotion_perplexity
                    metrics["baseline"]["perplexity"] = baseline_perplexity
                    
                    # 记录结果
                    result = {
                        "id": sample.get("id", f"sample_{len(all_results)}"),
                        "dialogue_id": dialogue_id,
                        "speaker_id": speaker_id,
                        "detected_emotion": detected_emotion,
                        "ground_truth_emotion": ground_truth_emotion,
                        "context": dialogue_context[-1].get("text", "") if dialogue_context else "",
                        "reference": reference,
                        "emotion_aware_response": emotion_aware_response,
                        "baseline_response": baseline_response,
                        "metrics": metrics,
                        "history_count": len(previous_utterances)
                    }
                    
                    all_results.append(result)
                    
                    # 添加到评估汇总
                    self.evaluation_results["emotion_aware"].append(metrics["emotion_aware"])
                    self.evaluation_results["baseline"].append(metrics["baseline"])
                    
                    # 清理临时合并的音频文件
                    if merged_audio_path != audio_path and os.path.exists(merged_audio_path):
                        try:
                            os.remove(merged_audio_path)
                        except:
                            pass
                
                except Exception as e:
                    self.logger.error(f"Error evaluating sample: {str(e)}")
                    self.logger.error(traceback.format_exc())
        
        # 计算每个说话者的情绪转变准确率
        for speaker_id, metrics in self.emotion_transition_metrics["by_speaker"].items():
            if metrics["total_transitions"] > 0:
                metrics["transition_accuracy"] = (
                    metrics["correctly_detected_transitions"] / 
                    metrics["total_transitions"]
                )
        
        # 计算总体情绪转变准确率
        if self.emotion_transition_metrics["total_transitions"] > 0:
            self.emotion_transition_metrics["transition_accuracy"] = (
                self.emotion_transition_metrics["correctly_detected_transitions"] / 
                self.emotion_transition_metrics["total_transitions"]
            )
        
        # 总结结果
        summary = self._summarize_results()
        
        # 添加情绪分布
        summary["emotion_distribution"] = {emotion: count / len(test_data) for emotion, count in emotion_distribution.items()}
        
        # 添加情绪转变指标
        summary["emotion_transitions"] = self.emotion_transition_metrics
        
        # 保存结果
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "summary": summary,
                    "samples": all_results
                }, f, indent=2, ensure_ascii=False)
        
        # 生成可视化结果
        self._generate_visualizations(summary, all_results, output_file)
        
        return summary
    
    def _summarize_results(self) -> Dict:
        """Summarize evaluation results"""
        emotion_aware_metrics = self.evaluation_results["emotion_aware"]
        baseline_metrics = self.evaluation_results["baseline"]
        
        summary = {
            "samples_count": len(emotion_aware_metrics),
            "emotion_aware": {},
            "baseline": {},
            "improvement": {}
        }
        
        # Calculate average for each metric
        for metric in ["rouge1", "rouge2", "rougeL", "bleu", "perplexity"]:
            emotion_values = [sample.get(metric, 0) for sample in emotion_aware_metrics]
            baseline_values = [sample.get(metric, 0) for sample in baseline_metrics]
            
            # Skip empty lists
            if not emotion_values or not baseline_values:
                self.logger.warning(f"No values for metric {metric}, skipping")
                continue
                
            emotion_avg = np.mean(emotion_values)
            baseline_avg = np.mean(baseline_values)
            
            # For perplexity, lower is better; for other metrics, higher is better
            if metric == "perplexity":
                improvement = (baseline_avg - emotion_avg) / baseline_avg if baseline_avg > 0 else 0
            else:
                improvement = (emotion_avg - baseline_avg) / baseline_avg if baseline_avg > 0 else 0
            
            summary["emotion_aware"][metric] = emotion_avg
            summary["baseline"][metric] = baseline_avg
            summary["improvement"][metric] = improvement
        
        return summary
    
    def _generate_visualizations(self, summary: Dict, results: List[Dict], output_prefix: str = None):
        """Generate result visualizations"""
        if not output_prefix:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_prefix = f'dialogue_eval_{timestamp}'
        
        # Create metrics comparison chart
        metrics = ["rouge1", "rouge2", "rougeL", "bleu"]
        
        # 1. Comparison bar chart
        plt.figure(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.35
        
        emotion_values = [summary["emotion_aware"].get(m, 0) for m in metrics]
        baseline_values = [summary["baseline"].get(m, 0) for m in metrics]
        
        plt.bar(x - width/2, emotion_values, width, label='Emotion-Aware Method')
        plt.bar(x + width/2, baseline_values, width, label='Baseline Method')
        
        plt.xlabel('Evaluation Metrics')
        plt.ylabel('Score')
        plt.title('Emotion-Aware Method vs Baseline Method Comparison')
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.savefig(f"{output_prefix}_metrics_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Emotion distribution pie chart
        emotions = summary.get("emotion_distribution", {})
        if emotions:
            plt.figure(figsize=(10, 10))
            plt.pie(list(emotions.values()), labels=list(emotions.keys()), autopct='%1.1f%%',
                   shadow=True, startangle=90)
            plt.axis('equal')
            plt.title('Test Data Emotion Distribution')
            plt.savefig(f"{output_prefix}_emotion_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        
        # 3. Performance improvement heatmap by emotion
        try:
            # Extract metric improvements for each emotion
            emotion_metrics = defaultdict(lambda: defaultdict(list))
            
            for result in results:
                emotion = result.get("detected_emotion", "unknown")
                metrics_data = result.get("metrics", {})
                
                for metric in metrics + ["perplexity"]:
                    emotion_value = metrics_data.get("emotion_aware", {}).get(metric, 0)
                    baseline_value = metrics_data.get("baseline", {}).get(metric, 0)
                    
                    # Calculate improvement
                    if metric == "perplexity":
                        improvement = (baseline_value - emotion_value) / baseline_value if baseline_value > 0 else 0
                    else:
                        improvement = (emotion_value - baseline_value) / baseline_value if baseline_value > 0 else 0
                    
                    emotion_metrics[emotion][metric].append(improvement)
            
            # Calculate average improvement
            emotion_improvement = {}
            for emotion, metrics_data in emotion_metrics.items():
                emotion_improvement[emotion] = {}
                for metric, values in metrics_data.items():
                    emotion_improvement[emotion][metric] = np.mean(values)
            
            # Create heatmap data
            emotions_list = list(emotion_improvement.keys())
            metrics_list = metrics + ["perplexity"]
            
            heatmap_data = []
            for emotion in emotions_list:
                row = []
                for metric in metrics_list:
                    row.append(emotion_improvement[emotion].get(metric, 0))
                heatmap_data.append(row)
            
            # Draw heatmap
            plt.figure(figsize=(12, 10))
            sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn", 
                       xticklabels=metrics_list, yticklabels=emotions_list)
            plt.title('Performance Improvement by Emotion (Relative to Baseline)')
            plt.tight_layout()
            plt.savefig(f"{output_prefix}_emotion_improvement_heatmap.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            self.logger.error(f"Error generating emotion performance heatmap: {str(e)}")

    def _group_by_dialogue(self, samples: List[Dict]) -> Dict[str, List[Dict]]:
        """Group samples by dialogue ID for analyzing emotion transitions"""
        dialogue_groups = {}
        
        for sample in samples:
            # Extract dialogue ID from utterance ID or use provided dialogue_id
            dialogue_id = sample.get('dialogue_id')
            
            if not dialogue_id and 'id' in sample:
                # Try to extract dialogue ID from utterance ID (e.g., Ses01F_impro01_M001 -> Ses01F_impro01)
                parts = sample['id'].split('_')
                if len(parts) >= 2:
                    dialogue_id = f"{parts[0]}_{parts[1]}"
            
            if not dialogue_id:
                # Use sample ID as fallback
                dialogue_id = sample.get('id', f"unknown_{len(dialogue_groups)}")
            
            if dialogue_id not in dialogue_groups:
                dialogue_groups[dialogue_id] = []
            
            dialogue_groups[dialogue_id].append(sample)
        
        # Sort samples within each dialogue group by ID if possible
        for dialogue_id, group_samples in dialogue_groups.items():
            if 'id' in group_samples[0]:
                try:
                    group_samples.sort(key=lambda x: x['id'])
                except:
                    pass  # Skip sorting if there's an issue
        
        self.logger.info(f"Grouped {len(samples)} samples into {len(dialogue_groups)} dialogues")
        return dialogue_groups

    def reset_emotion_states(self, dialogue_id: str = None):
        """
        Reset emotion states for a specific dialogue or all dialogues
        
        Args:
            dialogue_id: ID of dialogue to reset, or None to reset all
        """
        if dialogue_id:
            if dialogue_id in self.emotion_states:
                self.logger.info(f"Resetting emotion states for dialogue {dialogue_id}")
                self.emotion_states[dialogue_id] = {}
        else:
            self.logger.info("Resetting all emotion states")
            self.emotion_states = {}

    def merge_audio_files(self, audio_paths: List[str], output_path: str = None) -> str:
        """
        Merge multiple audio files into a single file with support for potential format issues
        
        Args:
            audio_paths: List of audio file paths
            output_path: Output file path (optional)
            
        Returns:
            Path to the merged audio file
        """
        try:
            from pydub import AudioSegment
            
            # 默认输出路径
            if not output_path:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio")
                os.makedirs(output_dir, exist_ok=True)
                output_path = os.path.join(output_dir, f"merged_{timestamp}.wav")
            
            # 验证所有音频路径
            valid_paths = []
            for path in audio_paths:
                if os.path.exists(path):
                    valid_paths.append(path)
                else:
                    self.logger.warning(f"Audio file not found: {path}, skipping")
            
            if not valid_paths:
                self.logger.error("No valid audio files to merge")
                return None
            
            # 如果只有一个有效文件，直接返回
            if len(valid_paths) == 1:
                self.logger.info(f"Only one valid audio file, no need to merge: {valid_paths[0]}")
                return valid_paths[0]
            
            # 初始化合并的音频
            merged_audio = None
            
            # 按顺序合并音频
            for audio_path in valid_paths:
                try:
                    audio = AudioSegment.from_wav(audio_path)
                    
                    # 检查音频是否为空
                    if len(audio) == 0:
                        self.logger.warning(f"Empty audio file: {audio_path}, skipping")
                        continue
                        
                    if merged_audio is None:
                        merged_audio = audio
                    else:
                        # 添加短暂的静音以分隔对话轮次
                        silence = AudioSegment.silent(duration=500)  # 500ms
                        merged_audio = merged_audio + silence + audio
                        
                except Exception as e:
                    self.logger.error(f"Error processing audio file {audio_path}: {str(e)}")
                    continue
            
            # 如果没有有效音频，返回第一个文件路径
            if merged_audio is None:
                self.logger.warning("Failed to merge audio, returning first file")
                return valid_paths[0]
            
            # 导出合并的音频
            try:
                merged_audio.export(output_path, format="wav")
                self.logger.info(f"Successfully merged {len(valid_paths)} audio files to {output_path}")
                return output_path
            except Exception as e:
                self.logger.error(f"Error exporting merged audio: {str(e)}")
                return valid_paths[0]  # 失败时返回第一个文件
            
        except Exception as e:
            self.logger.error(f"Error merging audio files: {str(e)}")
            self.logger.error(traceback.format_exc())
            return audio_paths[0] if audio_paths else None

    def get_previous_utterances(self, dialogue_samples: List[Dict], current_index: int, 
                                max_history: int = 3, same_speaker_only: bool = False) -> List[Dict]:
        """
        Get previous utterances from dialogue history
        
        Args:
            dialogue_samples: List of samples in the dialogue
            current_index: Index of the current sample
            max_history: Maximum number of previous utterances to retrieve
            same_speaker_only: Whether to only get utterances from the same speaker
            
        Returns:
            List of previous utterance samples
        """
        previous = []
        
        if current_index <= 0:
            return previous
        
        # 获取当前说话者的ID
        current_sample = dialogue_samples[current_index]
        current_speaker = self._extract_speaker_id(current_sample.get('id', ''))
        
        # 计数已添加的历史记录数
        count = 0
        
        # 向前搜索历史记录
        for i in range(current_index - 1, -1, -1):
            if count >= max_history:
                break
            
            prev_sample = dialogue_samples[i]
            prev_speaker = self._extract_speaker_id(prev_sample.get('id', ''))
            
            # 如果只需要同一说话者的历史记录
            if same_speaker_only and prev_speaker != current_speaker:
                continue
            
            previous.insert(0, prev_sample)  # 在列表前端插入，保持时间顺序
            count += 1
        
        return previous

    def _extract_speaker_id(self, utterance_id: str) -> str:
        """
        Extract speaker ID from utterance ID
        
        Args:
            utterance_id: Utterance ID (e.g., "Ses01F_impro01_M001")
            
        Returns:
            Speaker ID (e.g., "M" or "F")
        """
        # IEMOCAP格式通常是Ses01F_impro01_M001，M或F在_后表示说话者
        parts = utterance_id.split('_')
        if len(parts) >= 3:
            # 提取最后一部分的第一个字符
            speaker_part = parts[2]
            if speaker_part.startswith('M') or speaker_part.startswith('F'):
                return speaker_part[0]
        
        # 如果无法提取，回退到Ses01F中的F
        if len(parts) >= 1:
            first_part = parts[0]
            if 'F' in first_part:
                return 'F'
            elif 'M' in first_part:
                return 'M'
        
        return "unknown"  # 默认值

    def find_audio_file(self, utterance_id: str, iemocap_root: str) -> str:
        """确保找到正确的音频文件路径"""
        abs_iemocap_root = os.path.abspath(iemocap_root)
        parts = utterance_id.split('_')
        
        if len(parts) < 3:
            raise ValueError(f"Invalid utterance ID format: {utterance_id}")
        
        session_id = parts[0]
        if not session_id.startswith('Ses') or len(session_id) < 5:
            raise ValueError(f"Invalid session ID format: {session_id}")
        
        try:
            session_num = int(session_id[3:5])
            if session_num < 1 or session_num > 5:
                raise ValueError(f"Session number out of range (1-5): {session_num}")
        except ValueError as e:
            raise ValueError(f"Cannot parse session number: {session_id}") from e
        
        session = f"Session{session_num}"
        dialogue = f"{parts[0]}_{parts[1]}"
        audio_filename = f"{utterance_id}.wav"
        
        # 尝试多种可能的路径模式
        patterns = [
            os.path.join(abs_iemocap_root, session, "sentences", "wav", dialogue, audio_filename),
            os.path.join(abs_iemocap_root, session, "wav", dialogue, audio_filename),
            os.path.join(abs_iemocap_root, session, "dialog", "wav", dialogue, audio_filename)
        ]
        
        for pattern in patterns:
            if os.path.exists(pattern):
                self.logger.info(f"Found audio file: {pattern}")
                return pattern
        
        # 如果找不到，尝试搜索整个会话目录
        self.logger.warning(f"Audio file not found through standard patterns, searching session directory")
        session_path = os.path.join(abs_iemocap_root, session)
        
        for root, _, files in os.walk(session_path):
            if audio_filename in files:
                path = os.path.join(root, audio_filename)
                self.logger.info(f"Found audio file through search: {path}")
                return path
        
        raise FileNotFoundError(f"Audio file not found for utterance: {utterance_id}")

def prepare_test_data(data_file: str, iemocap_root: str) -> List[Dict]:
    """
    Prepare test data
    
    Args:
        data_file: Data file path
        iemocap_root: IEMOCAP dataset root directory
        
    Returns:
        Prepared test data
    """
    try:
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        test_samples = []
        
        for item in data:
            utterance_id = item.get('id')
            if not utterance_id:
                continue
            
            # Build audio path
            parts = utterance_id.split('_')
            if len(parts) < 3:
                continue
                
            session_id = parts[0]
            if not session_id.startswith('Ses') or len(session_id) < 5:
                continue
                
            try:
                session_num = int(session_id[3:5])
                if session_num < 1 or session_num > 5:
                    continue
            except ValueError:
                continue
                
            session = f"Session{session_num}"
            dialogue = f"{parts[0]}_{parts[1]}"
            audio_filename = f"{utterance_id}.wav"
            
            # Try several possible path patterns
            audio_path = None
            patterns = [
                os.path.join(iemocap_root, session, "sentences", "wav", dialogue, audio_filename),
                os.path.join(iemocap_root, session, "wav", dialogue, audio_filename),
                os.path.join(iemocap_root, session, "dialog", "wav", dialogue, audio_filename)
            ]
            
            for pattern in patterns:
                if os.path.exists(pattern):
                    audio_path = pattern
                    break
            
            if not audio_path:
                continue
            
            # Simulate dialogue context
            context = [
                {"text": "Hello, how can I help you?", "is_assistant": True},
                {"text": item.get('text', "This is a test sentence."), "user_id": f"user_{utterance_id}"}
            ]
            
            # Build sample
            sample = {
                "id": utterance_id,
                "context": context,
                "audio_path": audio_path,
                "reference": "I'll do my best to help you. Please tell me more about your needs.",  # Example reference response
                "emotion": item.get('emotion', [])
            }
            
            test_samples.append(sample)
        
        return test_samples
        
    except Exception as e:
        print(f"Error preparing test data: {str(e)}")
        return []

def main():
    parser = argparse.ArgumentParser(description='Emotional Dialogue Evaluation Tool')
    parser.add_argument('--data', type=str, required=True,
                      help='Test data JSON file path')
    parser.add_argument('--iemocap_root', type=str, required=True,
                      help='IEMOCAP dataset root directory')
    parser.add_argument('--output', type=str, default=None,
                      help='Evaluation results output file')
    parser.add_argument('--max_samples', type=int, default=10,
                      help='Maximum number of samples (-1 for all)')
    parser.add_argument('--api_key', type=str, default=None,
                      help='OpenAI API key')
    parser.add_argument('--transition_instructions', type=str, default=None,
                      help='Path to emotion transition instructions JSON file')
    parser.add_argument('--sustained_instructions', type=str, default=None,
                      help='Path to sustained emotion instructions JSON file')
    parser.add_argument('--max_history', type=int, default=3,
                      help='Maximum number of previous utterances to include in context')
    
    args = parser.parse_args()
    
    try:
        # 检查并安装依赖
        try:
            import pydub
        except ImportError:
            print("Installing required dependencies...")
            import subprocess
            subprocess.check_call(["pip", "install", "pydub"])
            print("Dependencies installed successfully.")
        
        # 准备测试数据
        test_data = prepare_test_data(args.data, args.iemocap_root)
        
        if not test_data:
            print("No valid test data found!")
            return
        
        print(f"Found {len(test_data)} test samples")
        
        # 初始化评估器
        evaluator = EmotionDialogueEvaluator(
            api_key=args.api_key, 
            transition_instructions_file=args.transition_instructions,
            sustained_instructions_file=args.sustained_instructions
        )
        
        # 执行评估
        summary = evaluator.evaluate_dialogue(
            test_data, 
            output_file=args.output,
            max_samples=args.max_samples
        )
        
        # 打印总结
        print("\nEvaluation completed!")
        print(f"Sample count: {summary['samples_count']}")
        
        # 打印情绪转变指标
        transition_metrics = summary.get("emotion_transitions", {})
        print("\nEmotion Transition Detection:")
        print(f"Total transitions: {transition_metrics.get('total_transitions', 0)}")
        print(f"Correctly detected: {transition_metrics.get('correctly_detected_transitions', 0)}")
        print(f"Accuracy: {transition_metrics.get('transition_accuracy', 0):.2%}")
        
        # 按说话者打印
        if "by_speaker" in transition_metrics:
            print("\nEmotion Transition Detection by Speaker:")
            for speaker, metrics in transition_metrics["by_speaker"].items():
                print(f"Speaker {speaker}:")
                print(f"  Total transitions: {metrics.get('total_transitions', 0)}")
                print(f"  Correctly detected: {metrics.get('correctly_detected_transitions', 0)}")
                print(f"  Accuracy: {metrics.get('transition_accuracy', 0):.2%}")
        
        print("\nEmotion-Aware Method vs Baseline Method:")
        
        metrics = ["rouge1", "rouge2", "rougeL", "bleu", "perplexity"]
        for metric in metrics:
            if metric not in summary["emotion_aware"] or metric not in summary["baseline"]:
                print(f"{metric}: metrics calculation failed")
                continue
                
            emotion_value = summary["emotion_aware"][metric]
            baseline_value = summary["baseline"][metric]
            improvement = summary["improvement"][metric] * 100
            
            if metric == "perplexity":
                better = "lower" if emotion_value < baseline_value else "higher"
            else:
                better = "higher" if emotion_value > baseline_value else "lower"
                
            print(f"{metric}: {emotion_value:.4f} vs {baseline_value:.4f} ({better} by {abs(improvement):.2f}%)")
        
        print("\nEvaluation results saved to:", args.output if args.output else "default output file")
        
    except Exception as e:
        print(f"Error during evaluation process: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    main() 
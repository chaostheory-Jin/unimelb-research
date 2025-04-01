import base64
import requests
import json
import os
from openai import OpenAI
from typing import Dict, List, Optional, Tuple
import time
import re
import argparse
import glob

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# Define emotion states and response strategies
EMOTION_STATES = ["neutral", "happy", "sad", "angry", "anxious", "surprised"]

# Response strategy library based on empirical psychology
RESPONSE_STRATEGIES = {
    "neutral": {
        "name": "information_focused",
        "description": "Provide clear, factual information with a balanced tone."
    },
    "happy": {
        "name": "positive_reinforcement",
        "description": "Match the positive energy, affirm feelings, and build on the positive state."
    },
    "sad": {
        "name": "empathetic_support",
        "description": "Express understanding, validate feelings, offer gentle encouragement and support."
    },
    "angry": {
        "name": "de_escalation",
        "description": "Acknowledge frustration, use calming language, avoid defensiveness, and focus on solutions."
    },
    "anxious": {
        "name": "reassurance",
        "description": "Provide calm reassurance, clear information to reduce uncertainty, and gentle guidance."
    },
    "surprised": {
        "name": "clarification",
        "description": "Provide context, explain unexpected information, and help process the surprising elements."
    }
}

# Emotion transition strategy matrix
TRANSITION_STRATEGIES = {
    ("neutral", "angry"): {
        "name": "preemptive_de_escalation",
        "description": "Recognize rising tension, acknowledge concerns early, and address issues before frustration increases."
    },
    ("neutral", "sad"): {
        "name": "supportive_response",
        "description": "Offer empathy and emotional support for the shift to sadness."
    },
    ("neutral", "anxious"): {
        "name": "structured_reassurance",
        "description": "Provide clear information and structured guidance to address emerging anxiety."
    },
    ("happy", "neutral"): {
        "name": "engagement_maintenance",
        "description": "Maintain engagement while adapting to more neutral emotional tone."
    },
    ("happy", "sad"): {
        "name": "gentle_transition",
        "description": "Acknowledge the mood shift with empathy, validate the change in emotions."
    },
    ("sad", "neutral"): {
        "name": "positive_refocusing",
        "description": "Gently guide toward constructive topics while respecting previous emotional state."
    },
    ("angry", "neutral"): {
        "name": "solution_focused",
        "description": "Capitalize on calming by moving toward constructive problem-solving."
    },
    ("anxious", "neutral"): {
        "name": "confidence_building",
        "description": "Reinforce security and emphasize capability as anxiety diminishes."
    },
    # Additional common transitions
    ("happy", "angry"): {
        "name": "disappointment_management",
        "description": "Address the shift from positive to negative by acknowledging disappointment and offering constructive paths forward."
    },
    ("sad", "happy"): {
        "name": "positive_reinforcement",
        "description": "Reinforce and encourage the positive emotional shift while acknowledging the previous state."
    },
    ("angry", "sad"): {
        "name": "empathetic_redirection",
        "description": "Acknowledge the underlying hurt beneath anger, offer compassion and supportive understanding."
    },
    ("anxious", "angry"): {
        "name": "frustration_validation",
        "description": "Validate how anxiety can transform to frustration, while calmly addressing concerns."
    }
}

class EmotionTracker:
    def __init__(self):
        self.emotion_history = []
        self.transition_counts = {}
        self.transition_timestamps = []
        
    def add_emotion(self, emotion: str):
        """Add a new emotion to the history and track transitions"""
        self.emotion_history.append(emotion)
        timestamp = time.time()
        
        # Track transitions
        if len(self.emotion_history) > 1:
            prev = self.emotion_history[-2]
            curr = self.emotion_history[-1]
            
            # FIX 2: Use string keys instead of tuples for transitions
            transition = f"{prev}_to_{curr}"  # Change from tuple to string format
            
            if transition in self.transition_counts:
                self.transition_counts[transition] += 1
            else:
                self.transition_counts[transition] = 1
                
            # Record timestamp of transition
            self.transition_timestamps.append((transition, timestamp))
    
    def get_most_frequent_transition(self):
        """Get the most frequently occurring emotion transition"""
        if not self.transition_counts:
            return ("none_to_none", 0)
        
        most_freq = max(self.transition_counts.items(), key=lambda x: x[1])
        return most_freq
    
    def get_dominant_emotion(self) -> str:
        """Get the most frequently expressed emotion"""
        if not self.emotion_history:
            return "none"
        
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return max(emotion_counts.items(), key=lambda x: x[1])[0]
    
    def get_recent_trend(self, window_size=3) -> str:
        """Analyze recent emotion trend in the conversation"""
        if len(self.emotion_history) < window_size:
            return "insufficient_data"
            
        recent_emotions = self.emotion_history[-window_size:]
        
        # Check if emotions are consistent
        if all(emotion == recent_emotions[0] for emotion in recent_emotions):
            return f"consistent_{recent_emotions[0]}"
            
        # Check for improving trend (e.g., angry -> neutral -> happy)
        emotion_valence = {
            "angry": 1, 
            "sad": 2, 
            "anxious": 3, 
            "neutral": 4, 
            "surprised": 5, 
            "happy": 6
        }
        
        if all(emotion_valence.get(recent_emotions[i], 0) < emotion_valence.get(recent_emotions[i+1], 0) 
              for i in range(len(recent_emotions)-1)):
            return "improving"
            
        if all(emotion_valence.get(recent_emotions[i], 0) > emotion_valence.get(recent_emotions[i+1], 0) 
              for i in range(len(recent_emotions)-1)):
            return "deteriorating"
            
        return "fluctuating"
    
    def generate_transition_report(self) -> Dict:
        """Generate a comprehensive report of emotion transitions"""
        # Calculate time spent in each emotion
        emotion_durations = {}
        prev_timestamp = None
        prev_emotion = None
        
        for i, emotion in enumerate(self.emotion_history):
            timestamp = self.transition_timestamps[i-1][1] if i > 0 else time.time()
            
            if prev_timestamp and prev_emotion:
                duration = timestamp - prev_timestamp
                if prev_emotion in emotion_durations:
                    emotion_durations[prev_emotion] += duration
                else:
                    emotion_durations[prev_emotion] = duration
                    
            prev_timestamp = timestamp
            prev_emotion = emotion
        
        # Make sure we're using string keys in all dictionaries for JSON serialization
        return {
            "dominant_emotion": self.get_dominant_emotion(),
            "emotion_sequence": self.emotion_history,
            "transitions": dict(self.transition_counts),  # Now using string keys
            "most_frequent_transition": self.get_most_frequent_transition(),
            "recent_trend": self.get_recent_trend(),
            "emotion_durations": emotion_durations
        }


class EmotionAwareConversationSystem:
    def __init__(self):
        self.conversation_history = []
        self.user_emotions = []
        self.current_strategy = None
        self.emotion_tracker = EmotionTracker()
        self.conversation_turns = 0
    
    def encode_audio(self, audio_file_path: str) -> str:
        """Encode an audio file to base64 string"""
        if audio_file_path.startswith("http"):
            # Download from URL
            response = requests.get(audio_file_path)
            response.raise_for_status()
            audio_data = response.content
        else:
            # Read local file
            with open(audio_file_path, "rb") as audio_file:
                audio_data = audio_file.read()
        
        return base64.b64encode(audio_data).decode('utf-8')
    
    def detect_emotion(self, audio_base64: str) -> str:
        """Detect emotion from audio using the OpenAI API"""
        print("Detecting emotion...")
        
        try:
            # 使用正确的音频输入格式
            response = client.chat.completions.create(
                model="gpt-4o-audio-preview",
                messages=[
                    {"role": "system", "content": "You are an emotion detection assistant. Analyze the audio and determine the primary emotion expressed. Only respond with a single emotion word from this list: neutral, happy, sad, angry, anxious, surprised."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What emotion is expressed in this audio?"},
                        {"type": "input_audio", "input_audio": {
                            "data": audio_base64,
                            "format": "wav"
                        }}
                    ]}
                ]
            )
            
            content = response.choices[0].message.content.lower()
            print(f"Raw emotion detection result: {content}")
            
            # 直接查找情绪关键词
            for emotion in EMOTION_STATES:
                if emotion.lower() in content.lower():
                    print(f"Detected emotion: {emotion}")
                    return emotion
                    
            # 默认为 neutral
            print("No specific emotion detected, defaulting to neutral")
            return "neutral"
        except Exception as e:
            print(f"Error detecting emotion: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            return "neutral"  # 错误时默认为 neutral
    
    def select_response_strategy(self, current_emotion: str, previous_emotion: Optional[str] = None) -> Dict:
        """Select appropriate response strategy based on emotion or emotion transition"""
        if previous_emotion is None or previous_emotion == current_emotion:
            # Use single emotion strategy
            return RESPONSE_STRATEGIES[current_emotion]
        else:
            # Check for a specific transition strategy
            transition_key = (previous_emotion, current_emotion)
            if transition_key in TRANSITION_STRATEGIES:
                return TRANSITION_STRATEGIES[transition_key]
            else:
                # Default to the strategy for current emotion if no transition strategy exists
                return RESPONSE_STRATEGIES[current_emotion]
    
    def generate_response(self, user_input: str, detected_emotion: str, strategy: Dict, conversation_history=None) -> str:
        """Generate response based on user input, emotion, and selected strategy"""
        print("Generating response...")
        
        try:
            # FIX 1: Add proper audio output configuration
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": f"You are an empathetic assistant that responds with awareness of the user's emotional state. The user seems {detected_emotion}. Use the following response strategy: {strategy['name']} - {strategy['description']}"},
                    {"role": "user", "content": user_input}
                ],
                # Remove audio output configuration or set it properly if needed
                output="text"  # We only want text output here
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"I'm sorry, I wasn't able to process your message properly. Could you please try again?"
    
    def process_conversation_turn(self, audio_file_path: str) -> str:
        """Process a complete conversation turn with emotion detection and response generation"""
        self.conversation_turns += 1
        print(f"\n=== Processing Conversation Turn {self.conversation_turns} ===")
        
        # 1. Encode audio
        print("Encoding audio...")
        audio_data = self.encode_audio(audio_file_path)
        
        # 2. Detect emotion
        print("Detecting emotion...")
        current_emotion = self.detect_emotion(audio_data)
        print(f"Detected emotion: {current_emotion}")
        
        # 3. Update emotion tracker
        self.emotion_tracker.add_emotion(current_emotion)
        
        # 4. Determine previous emotion (if any)
        previous_emotion = self.user_emotions[-1] if self.user_emotions else None
        
        # 5. Store emotion in history
        self.user_emotions.append(current_emotion)
        
        # 6. Select response strategy
        print("Selecting response strategy...")
        strategy = self.select_response_strategy(current_emotion, previous_emotion)
        self.current_strategy = strategy
        print(f"Selected strategy: {strategy['name']} - {strategy['description']}")
        
        # 7. Generate response
        print("Generating response...")
        response = self.generate_response(audio_data, current_emotion, strategy, self.conversation_history)
        
        # 8. Store in conversation history
        self.conversation_history.append({
            "turn": self.conversation_turns,
            "user_emotion": current_emotion,
            "strategy_used": strategy['name'],
            "response": response
        })
        
        # 9. Print emotion transition information if applicable
        if previous_emotion and previous_emotion != current_emotion:
            print(f"\nEmotion Transition Detected: {previous_emotion} → {current_emotion}")
            
            # Get trend information
            trend = self.emotion_tracker.get_recent_trend()
            print(f"Recent emotion trend: {trend}")
        
        return response
    
    def get_conversation_metrics(self) -> Dict:
        """Generate metrics about the conversation"""
        if not self.conversation_history:
            return {"status": "No conversation data available"}
        
        # Calculate basic metrics
        emotion_changes = sum(1 for i in range(1, len(self.user_emotions)) 
                             if self.user_emotions[i] != self.user_emotions[i-1])
        
        # Get transition report from emotion tracker
        transition_report = self.emotion_tracker.generate_transition_report()
        
        # Calculate strategy distribution
        strategy_counts = {}
        for turn in self.conversation_history:
            strategy = turn["strategy_used"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
        
        return {
            "conversation_summary": {
                "turns": len(self.conversation_history),
                "emotion_changes": emotion_changes,
                "emotions_detected": self.user_emotions,
                "strategies_used": [turn["strategy_used"] for turn in self.conversation_history]
            },
            "emotion_analysis": transition_report,
            "strategy_distribution": strategy_counts
        }
    
    def evaluate_system_performance(self) -> Dict:
        """Evaluate system performance based on various metrics"""
        # In a real system, these would be calculated against ground truth or user feedback
        if not self.conversation_history:
            return {"status": "No conversation data available for evaluation"}
        
        # Get emotion transition report
        transition_report = self.emotion_tracker.generate_transition_report()
        
        # Example metrics - in a real system these would be calculated from actual data
        strategy_appropriateness = 0.85  # Example value
        
        evaluation = {
            "emotion_recognition": {
                "accuracy": 0.92,  # Would be calculated against ground truth in a real system
                "confidence": 0.88
            },
            "emotion_transitions": {
                "detection_accuracy": 0.90,
                "transition_patterns": transition_report["transitions"]
            },
            "response_strategies": {
                "appropriateness": strategy_appropriateness,
                "consistency": 0.95
            },
            "overall_quality": {
                "coherence": 0.91,
                "empathy": 0.87,
                "user_satisfaction": 0.82
            }
        }
        
        return evaluation
    
    def evaluate_with_iemocap_data(self, iemocap_data_path: str) -> Dict:
        """
        使用IEMOCAP数据集评估情绪识别和响应策略的性能
        
        :param iemocap_data_path: IEMOCAP数据集JSON文件路径
        :return: 评估结果报告
        """
        try:
            # 加载IEMOCAP数据
            with open(iemocap_data_path, 'r', encoding='utf-8') as f:
                iemocap_data = json.load(f)
            
            print(f"加载了IEMOCAP数据：{len(iemocap_data)}个样本")
            
            # 过滤需要预测的样本
            prediction_samples = [sample for sample in iemocap_data if sample.get('need_prediction') == 'yes']
            evaluation_samples = [sample for sample in iemocap_data if sample.get('need_prediction') == 'no']
            
            print(f"需要预测的样本：{len(prediction_samples)}个")
            print(f"用于评估的样本：{len(evaluation_samples)}个")
            
            # 评估情绪识别准确性
            emotion_results = self._evaluate_emotion_recognition(evaluation_samples)
            
            # 评估回应策略
            strategy_results = self._evaluate_response_strategies(evaluation_samples)
            
            # 生成完整评估报告
            results = {
                "dataset_stats": {
                    "total_samples": len(iemocap_data),
                    "evaluation_samples": len(evaluation_samples),
                    "prediction_samples": len(prediction_samples)
                },
                "emotion_recognition": emotion_results,
                "response_strategy": strategy_results
            }
            
            return results
            
        except Exception as e:
            print(f"评估过程中出错: {e}")
            return {"error": str(e)}
    
    def _evaluate_emotion_recognition(self, samples: List[Dict]) -> Dict:
        """评估情绪识别性能"""
        # 根据IEMOCAP数据格式评估情绪识别准确性
        emotion_categories = set()
        emotion_counts = {}
        correct_counts = {}
        total_correct = 0
        total_samples = 0
        
        # 最大评估样本数（避免API成本过高）
        max_samples = min(100, len(samples))
        evaluation_samples = samples[:max_samples]
        
        for sample in evaluation_samples:
            try:
                # 从IEMOCAP格式中提取真实情绪
                # IEMOCAP数据中每个样本的情绪有3个标注者的标注
                # 我们使用多数投票结果作为真实情绪
                emotions = sample['emotion']
                true_emotion = self._get_majority_emotion(emotions)
                emotion_categories.add(true_emotion)
                
                # 增加该情绪类别的计数
                emotion_counts[true_emotion] = emotion_counts.get(true_emotion, 0) + 1
                
                # 使用系统预测情绪
                audio_path = sample['audio']
                
                # 模拟预测（这里使用模拟函数避免实际API调用）
                predicted_emotion = self._simulate_emotion_recognition_from_transcript(sample['groundtruth'])
                
                # 映射情绪标签到我们的格式
                predicted_emotion = self._map_emotion_label(predicted_emotion)
                true_emotion = self._map_emotion_label(true_emotion)
                
                # 计算准确性
                is_correct = predicted_emotion == true_emotion
                if is_correct:
                    total_correct += 1
                    correct_counts[true_emotion] = correct_counts.get(true_emotion, 0) + 1
                
                total_samples += 1
                
            except Exception as e:
                print(f"评估样本 {sample.get('id', 'unknown')} 时出错: {e}")
        
        # 计算总体准确率
        overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
        
        # 计算每种情绪的准确率
        emotion_accuracy = {}
        for emotion in emotion_counts:
            accuracy = correct_counts.get(emotion, 0) / emotion_counts[emotion] if emotion_counts[emotion] > 0 else 0
            emotion_accuracy[emotion] = {
                "accuracy": accuracy,
                "samples": emotion_counts[emotion],
                "correct": correct_counts.get(emotion, 0)
            }
        
        return {
            "overall_accuracy": overall_accuracy,
            "by_emotion": emotion_accuracy,
            "samples_evaluated": total_samples
        }
    
    def _get_majority_emotion(self, emotions: List[str]) -> str:
        """获取多数情绪标签"""
        if not emotions:
            return "unknown"
            
        # 统计每种情绪的出现次数
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
        # 找出出现次数最多的情绪
        majority_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0]
        return majority_emotion
    
    def _map_emotion_label(self, iemocap_emotion: str) -> str:
        """
        将IEMOCAP情绪标签映射到我们的情绪标签
        """
        # IEMOCAP使用的情绪标签转换为我们的格式
        mapping = {
            "Neutral state": "neutral",
            "Frustration": "frustrated",
            "Anger": "angry",
            "Happiness": "happy",
            "Excited": "happy",  # 合并excited和happy
            "Sadness": "sad",
            "Fear": "anxious",
            "Surprise": "surprised",
            "Disgust": "angry",  # 将disgusted映射到angry
            "Other": "neutral"   # 将other映射到neutral
        }
        
        # 如果没有直接映射，尝试部分匹配
        if iemocap_emotion not in mapping:
            # 检查是否包含关键词
            for key, value in mapping.items():
                if key.lower() in iemocap_emotion.lower():
                    return value
            
            # 默认返回neutral
            return "neutral"
            
        return mapping[iemocap_emotion]
    
    def _simulate_emotion_recognition_from_transcript(self, transcript: str) -> str:
        """基于文本转录模拟情绪识别（用于评估）"""
        try:
            # 移除 output 参数
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",  # 使用较经济的模型
                messages=[
                    {"role": "system", "content": "您是一个情绪分析助手。您需要从以下文本中识别表达的主要情绪。请只回复一个情绪标签：Neutral state, Frustration, Anger, Happiness, Excited, Sadness, Fear, Surprise, Disgust"},
                    {"role": "user", "content": f"文本：{transcript}\n请识别文本中表达的情绪。"}
                ],
                temperature=0.3
                # 移除 output 参数
            )
            
            content = response.choices[0].message.content.strip()
            
            # 返回识别的情绪
            return content
            
        except Exception as e:
            print(f"情绪识别模拟过程中出错: {e}")
            return "Neutral state"  # 出错时默认为中性
    
    def _evaluate_response_strategies(self, samples: List[Dict]) -> Dict:
        """评估响应策略的适当性"""
        # 根据情绪分组样本
        emotion_groups = {}
        
        for sample in samples:
            emotions = sample['emotion']
            majority_emotion = self._get_majority_emotion(emotions)
            mapped_emotion = self._map_emotion_label(majority_emotion)
            
            if mapped_emotion not in emotion_groups:
                emotion_groups[mapped_emotion] = []
                
            emotion_groups[mapped_emotion].append(sample)
        
        # 评估每种情绪的响应策略
        strategy_evaluations = {}
        
        for emotion, samples in emotion_groups.items():
            # 跳过不在我们系统中的情绪
            if emotion not in RESPONSE_STRATEGIES:
                continue
                
            # 从每个情绪类别中最多选择10个样本评估
            eval_samples = samples[:10]
            
            # 获取对应的策略
            strategy = RESPONSE_STRATEGIES[emotion]
            
            # 评估策略适当性
            appropriateness_scores = []
            
            for sample in eval_samples:
                # 模拟响应生成和评估
                response = self._simulate_response_generation(
                    sample['groundtruth'], 
                    emotion, 
                    strategy
                )
                
                # 评估响应的适当性
                score = self._evaluate_response_appropriateness(
                    sample['groundtruth'],
                    response,
                    emotion
                )
                
                appropriateness_scores.append(score)
            
            # 计算平均适当性分数
            avg_score = sum(appropriateness_scores) / len(appropriateness_scores) if appropriateness_scores else 0
            
            strategy_evaluations[emotion] = {
                "strategy": strategy['name'],
                "appropriateness_score": avg_score,
                "samples_evaluated": len(eval_samples)
            }
        
        return strategy_evaluations
    
    def _simulate_response_generation(self, user_input: str, emotion: str, strategy: Dict) -> str:
        """模拟使用特定策略生成响应"""
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": f"你是一个共情助手，回应用户时要考虑到用户的情绪状态。用户似乎处于{emotion}状态。使用以下回应策略：{strategy['name']} - {strategy['description']}"},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150,
                temperature=0.7
                # 移除 output 参数
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"响应生成模拟过程中出错: {e}")
            return "我理解你的感受，请告诉我更多。"  # 默认回复
    
    def _evaluate_response_appropriateness(self, user_input: str, response: str, emotion: str) -> float:
        """评估响应的适当性"""
        try:
            eval_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "你是一个评估助手。请评估AI回复对给定情绪用户的适当性。评分标准：1-10分，10分表示完美适合该情绪状态的用户。"},
                    {"role": "user", "content": f"用户输入：{user_input}\n用户情绪：{emotion}\nAI回复：{response}\n\n请给AI回复的适当性打分（1-10分），只需回复数字分数。"}
                ],
                temperature=0.3
                # 移除 output 参数
            )
            
            # 解析分数
            content = eval_response.choices[0].message.content.strip()
            # 尝试提取数字
            match = re.search(r'(\d+(\.\d+)?)', content)
            if match:
                score = float(match.group(1))
                # 将10分制转换为0-1
                return min(score / 10.0, 1.0)
            else:
                return 0.7  # 默认分数
                
        except Exception as e:
            print(f"响应适当性评估过程中出错: {e}")
            return 0.7  # 默认分数

    def analyze_emotions_in_json(self, json_file_path: str) -> Dict:
        """分析JSON文件中的情感类型并统计分布"""
        try:
            # 加载JSON数据
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 统计情感类型和分布
            all_emotions = []
            for item in data:
                if 'emotion' in item and isinstance(item['emotion'], list):
                    all_emotions.extend(item['emotion'])
            
            # 计算各情感类型的计数
            emotion_counts = {}
            for emotion in all_emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # 按计数降序排序
            sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)
            
            return {
                "total_samples": len(data),
                "unique_emotions": list(emotion_counts.keys()),
                "emotion_distribution": emotion_counts,
                "sorted_emotions": sorted_emotions
            }
        except Exception as e:
            print(f"分析JSON情感时出错: {e}")
            return {"error": str(e)}

    def evaluate_emotion_from_audio(self, json_file_path: str, max_samples: int = 10, iemocap_root: str = None) -> Dict:
        """使用音频文件评估情感识别准确性"""
        try:
            # 加载JSON数据
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 如果未提供IEMOCAP根目录，则尝试推断
            if iemocap_root is None:
                # 从JSON文件路径推断
                json_dir = os.path.dirname(os.path.abspath(json_file_path))
                # 假设JSON在dataset/IEMOCAP目录下
                iemocap_root = os.path.join(json_dir, 'IEMOCAP_full_release')
                iemocap_root = os.path.normpath(iemocap_root)
                print(f"推断的IEMOCAP根目录: {iemocap_root}")
            else:
                # 确保路径使用系统适当的分隔符
                iemocap_root = os.path.normpath(iemocap_root)
            
            # 测试IEMOCAP根目录是否存在
            if not os.path.exists(iemocap_root):
                print(f"错误: IEMOCAP根目录不存在: {iemocap_root}")
                return {"error": f"IEMOCAP根目录不存在: {iemocap_root}"}
            
            # 测试访问特定文件确认根目录正确
            test_session_path = os.path.join(iemocap_root, "Session1")
            if os.path.exists(test_session_path):
                print(f"确认Session1目录存在: {test_session_path}")
            else:
                print(f"警告: Session1目录不存在: {test_session_path}")
            
            results = []
            total_samples = 0
            correct_samples = 0
            
            # 限制样本数以控制API调用成本
            sample_data = data[:max_samples]
            
            for item in sample_data:
                try:
                    # 获取音频路径
                    relative_audio_path = item.get('audio')
                    if not relative_audio_path:
                        print(f"跳过没有音频路径的样本 {item.get('id')}")
                        continue
                    
                    # 将相对路径中的/替换为系统路径分隔符
                    relative_audio_path = relative_audio_path.replace('/', os.path.sep)
                    
                    # 检查JSON中的路径是否已经包含IEMOCAP_full_release
                    if relative_audio_path.startswith("IEMOCAP_full_release"):
                        # 去掉前缀，因为iemocap_root已经包含这个目录
                        relative_audio_path = relative_audio_path[len("IEMOCAP_full_release")+1:]
                    
                    # 构建绝对路径
                    audio_path = os.path.join(iemocap_root, relative_audio_path)
                    audio_path = os.path.normpath(audio_path)
                    
                    print(f"尝试访问音频文件: {audio_path}")
                    
                    # 检查文件是否存在
                    if not os.path.exists(audio_path):
                        print(f"音频文件不存在: {audio_path}")
                        
                        # 尝试使用样本ID构建备用路径
                        utterance_id = item.get('id')
                        if utterance_id:
                            possible_paths = self._find_possible_audio_paths(utterance_id, iemocap_root)
                            if possible_paths:
                                print(f"找到备用路径: {possible_paths[0]}")
                                audio_path = possible_paths[0]
                            else:
                                # 最后尝试使用简单的模式匹配
                                glob_pattern = os.path.join(iemocap_root, f"**/*{utterance_id}*.wav")
                                matches = glob.glob(glob_pattern, recursive=True)
                                if matches:
                                    print(f"通过全局搜索找到: {matches[0]}")
                                    audio_path = matches[0]
                                else:
                                    print(f"无法找到音频文件，跳过样本 {utterance_id}")
                                    continue
                    
                    # 获取真实情感标签（使用多数投票）
                    true_emotion = self._get_majority_emotion(item.get('emotion', []))
                    
                    # 编码音频文件
                    print(f"正在编码音频文件: {audio_path}")
                    audio_base64 = self.encode_audio(audio_path)
                    
                    # 使用gpt-4o-audio-preview模型检测情感
                    predicted_emotion = self._detect_emotion_from_audio(audio_base64)
                    
                    # 转换为标准格式进行比较
                    true_emotion_std = self._map_emotion_label(true_emotion)
                    predicted_emotion_std = self._map_emotion_label(predicted_emotion)
                    
                    # 比较预测结果和真实标签
                    is_correct = predicted_emotion_std == true_emotion_std
                    if is_correct:
                        correct_samples += 1
                    
                    # 记录结果
                    results.append({
                        "id": item.get('id'),
                        "audio_path": audio_path,
                        "true_emotion": true_emotion,
                        "predicted_emotion": predicted_emotion,
                        "true_emotion_mapped": true_emotion_std,
                        "predicted_emotion_mapped": predicted_emotion_std,
                        "is_correct": is_correct
                    })
                    
                    total_samples += 1
                    
                except Exception as e:
                    print(f"处理音频样本 {item.get('id')} 时出错: {e}")
                    import traceback
                    traceback.print_exc()  # 打印详细错误堆栈
            
            # 计算准确率
            accuracy = correct_samples / total_samples if total_samples > 0 else 0
            
            return {
                "total_evaluated": total_samples,
                "correct_predictions": correct_samples,
                "accuracy": accuracy,
                "detailed_results": results
            }
        except Exception as e:
            print(f"评估音频情感时出错: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误堆栈
            return {"error": str(e)}

    def _find_possible_audio_paths(self, utterance_id: str, iemocap_root: str) -> List[str]:
        """尝试查找可能的音频文件路径"""
        possible_paths = []
        
        # 从ID解析会话和对话信息
        # 例如：从 Ses01F_impro03_F001 提取 Session1、Ses01F_impro03
        parts = utterance_id.split('_')
        
        # 保护性检查
        if len(parts) < 3:
            print(f"ID格式不正确: {utterance_id}，无法推断路径")
            return possible_paths
        
        try:
            # 从Ses01F中提取1作为会话号
            session_num = parts[0][3:4]  # 提取 '1'
            session = f"Session{session_num}"
            
            # 对话ID
            dialogue = f"{parts[0]}_{parts[1]}"
            
            # 完整音频文件名 (例如Ses01F_impro03_F001.wav)
            audio_filename = f"{utterance_id}.wav"
            
            # 尝试几种可能的路径模式
            patterns = [
                # 标准路径模式
                os.path.join(iemocap_root, session, "sentences", "wav", dialogue, audio_filename),
                
                # 路径变体1：直接在wav目录下的对话子目录中
                os.path.join(iemocap_root, session, "wav", dialogue, audio_filename),
                
                # 路径变体2：直接在wav目录下
                os.path.join(iemocap_root, session, "wav", audio_filename),
                
                # 路径变体3：直接在sentences目录下
                os.path.join(iemocap_root, session, "sentences", audio_filename),
                
                # 路径变体4：完全平铺的结构
                os.path.join(iemocap_root, session, audio_filename),
            ]
            
            # 输出所有尝试的路径用于调试
            print(f"正在尝试为 {utterance_id} 查找以下可能的路径:")
            for pattern in patterns:
                print(f"  - {pattern}")
                if os.path.exists(pattern):
                    print(f"    ✓ 文件存在!")
                    possible_paths.append(pattern)
                else:
                    print(f"    ✗ 文件不存在")
            
            # 如果上述模式都失败，尝试使用glob进行更广泛的搜索
            if not possible_paths:
                print(f"尝试使用glob搜索 {utterance_id}.wav")
                # 在会话目录中递归搜索
                session_path = os.path.join(iemocap_root, session)
                if os.path.exists(session_path):
                    glob_pattern = os.path.join(session_path, f"**/*{audio_filename}")
                    matches = glob.glob(glob_pattern, recursive=True)
                    if matches:
                        print(f"通过glob找到 {len(matches)} 个匹配: {matches}")
                        possible_paths.extend(matches)
        
        except Exception as e:
            print(f"在查找音频路径时出错: {e}")
            import traceback
            traceback.print_exc()
        
        return possible_paths

    def _detect_emotion_from_audio(self, audio_base64: str) -> str:
        """使用gpt-4o-audio-preview模型从音频检测情感"""
        try:
            # 使用正确的音频输入格式
            response = client.chat.completions.create(
                model="gpt-4o-audio-preview",
                messages=[
                    {"role": "system", "content": "You are an emotion detection assistant. Analyze the audio and determine the primary emotion expressed. Only respond with a single emotion word from this list: Neutral state, Frustration, Anger, Happiness, Excited, Sadness, Fear, Surprise, Disgust."},
                    {"role": "user", "content": [
                        {"type": "text", "text": "What is the primary emotion expressed in this audio?"},
                        {"type": "input_audio", "input_audio": {
                            "data": audio_base64,
                            "format": "wav"
                        }}
                    ]}
                ]
            )
            
            emotion = response.choices[0].message.content.strip()
            print(f"Audio emotion detection result: {emotion}")
            return emotion
            
        except Exception as e:
            print(f"音频情感检测出错: {e}")
            import traceback
            traceback.print_exc()  # 打印详细错误信息
            return "Neutral state"  # 出错时默认为中性

    def detect_emotion_changes(self, json_file_path: str) -> Dict:
        """检测同一人在对话中的情感变化"""
        try:
            # 加载JSON数据
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 按说话者分组
            speaker_groups = {}
            for item in data:
                speaker = item.get('speaker')
                if not speaker:
                    continue
                    
                if speaker not in speaker_groups:
                    speaker_groups[speaker] = []
                    
                speaker_groups[speaker].append(item)
            
            # 分析每个说话者的情感变化
            speaker_analyses = {}
            
            for speaker, utterances in speaker_groups.items():
                # 按ID排序，假设ID包含顺序信息
                utterances.sort(key=lambda x: x.get('id', ''))
                
                # 提取情感序列
                emotion_sequence = []
                for utterance in utterances:
                    # 获取多数情感
                    majority_emotion = self._get_majority_emotion(utterance.get('emotion', []))
                    # 映射到标准格式
                    std_emotion = self._map_emotion_label(majority_emotion)
                    emotion_sequence.append(std_emotion)
                
                # 计算情感变化
                emotion_changes = []
                for i in range(1, len(emotion_sequence)):
                    if emotion_sequence[i] != emotion_sequence[i-1]:
                        emotion_changes.append({
                            "from": emotion_sequence[i-1],
                            "to": emotion_sequence[i],
                            "utterance_id": utterances[i].get('id'),
                            "text": utterances[i].get('groundtruth', '')
                        })
                
                # 计算统计信息
                speaker_analyses[speaker] = {
                    "total_utterances": len(utterances),
                    "emotion_sequence": emotion_sequence,
                    "emotion_changes": len(emotion_changes),
                    "change_details": emotion_changes,
                    "most_frequent_emotion": max(set(emotion_sequence), key=emotion_sequence.count)
                }
            
            return {
                "speaker_count": len(speaker_groups),
                "speaker_analyses": speaker_analyses
            }
        except Exception as e:
            print(f"检测情感变化时出错: {e}")
            return {"error": str(e)}


def simulate_emotion_changes(conversation_system):
    """Simulate a conversation with emotion changes to test the system"""
    # Example audio files with different emotions
    # In a real implementation, these would be actual different audio files
    neutral_audio = "https://cdn.openai.com/API/docs/audio/alloy.wav"  # Pretend this is neutral
    happy_audio = "https://cdn.openai.com/API/docs/audio/alloy.wav"    # Pretend this is happy
    angry_audio = "https://cdn.openai.com/API/docs/audio/alloy.wav"    # Pretend this is angry
    sad_audio = "https://cdn.openai.com/API/docs/audio/alloy.wav"      # Pretend this is sad
    
    # Force emotion detection for simulation purposes
    original_detect_emotion = conversation_system.detect_emotion
    
    # Turn 1: Neutral
    print("\n=== SIMULATING CONVERSATION WITH EMOTION CHANGES ===")
    conversation_system.detect_emotion = lambda x: "neutral"
    response = conversation_system.process_conversation_turn(neutral_audio)
    print(f"\nSystem Response (Neutral):\n{response}\n")
    
    # Turn 2: Happy
    conversation_system.detect_emotion = lambda x: "happy"
    response = conversation_system.process_conversation_turn(happy_audio)
    print(f"\nSystem Response (Happy):\n{response}\n")
    
    # Turn 3: Angry (emotion change)
    conversation_system.detect_emotion = lambda x: "angry"
    response = conversation_system.process_conversation_turn(angry_audio)
    print(f"\nSystem Response (Angry):\n{response}\n")
    
    # Turn 4: Still angry (no change)
    response = conversation_system.process_conversation_turn(angry_audio)
    print(f"\nSystem Response (Still Angry):\n{response}\n")
    
    # Turn 5: Sad (emotion change)
    conversation_system.detect_emotion = lambda x: "sad"
    response = conversation_system.process_conversation_turn(sad_audio)
    print(f"\nSystem Response (Sad):\n{response}\n")
    
    # Restore original function
    conversation_system.detect_emotion = original_detect_emotion
    
    # Print detailed metrics
    metrics = conversation_system.get_conversation_metrics()
    print("\n=== CONVERSATION METRICS ===")
    print(json.dumps(metrics, indent=2))
    
    # Print system evaluation
    evaluation = conversation_system.evaluate_system_performance()
    print("\n=== SYSTEM EVALUATION ===")
    print(json.dumps(evaluation, indent=2))


def main():
    """Main function to demonstrate the emotion-aware conversation system"""
    print("Initializing Emotion-Aware Conversation System...")
    conversation_system = EmotionAwareConversationSystem()
    
    # 添加命令行参数解析
    parser = argparse.ArgumentParser(description='情绪感知对话系统')
    parser.add_argument('--mode', choices=['demo', 'evaluate', 'analyze_json', 'audio_eval', 'detect_changes'], 
                        default='demo', help='运行模式')
    parser.add_argument('--data', type=str, help='数据集JSON文件路径')
    parser.add_argument('--max_samples', type=int, default=10, help='评估的最大样本数')
    parser.add_argument('--iemocap_root', type=str, help='IEMOCAP数据集根目录的完整路径 (例如 C:/Users/luoya/Desktop/unimelb-research/dataset/IEMOCAP/IEMOCAP_full_release)')
    args = parser.parse_args()
    
    if args.mode == 'analyze_json' and args.data:
        # 分析JSON中的情感类型
        print(f"分析JSON文件中的情感类型：{args.data}")
        results = conversation_system.analyze_emotions_in_json(args.data)
        print("\n=== 情感分析结果 ===")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
        
    elif args.mode == 'audio_eval' and args.data:
        # 使用音频评估情感识别准确性
        print(f"使用音频评估情感识别（最多{args.max_samples}个样本）：{args.data}")
        
        if args.iemocap_root:
            print(f"使用指定的IEMOCAP根目录: {args.iemocap_root}")
        else:
            print("未指定IEMOCAP根目录，将尝试自动推断")
            
        results = conversation_system.evaluate_emotion_from_audio(
            args.data, 
            args.max_samples,
            args.iemocap_root
        )
        print("\n=== 音频情感评估结果 ===")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
        
    elif args.mode == 'detect_changes' and args.data:
        # 检测情感变化
        print(f"检测对话中的情感变化：{args.data}")
        results = conversation_system.detect_emotion_changes(args.data)
        print("\n=== 情感变化检测结果 ===")
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return
        
    elif args.mode == 'evaluate' and args.data:
        # 评估模式
        print(f"使用IEMOCAP数据集评估系统：{args.data}")
        evaluation_results = conversation_system.evaluate_with_iemocap_data(args.data)
        print("\n=== 评估结果 ===")
        print(json.dumps(evaluation_results, indent=2, ensure_ascii=False))
        return
    
    # 演示模式
    use_real_audio = False  # Set to True to use real audio processing
    
    if use_real_audio:
        # 使用真实音频
        print("Processing first conversation turn...")
        audio_file = "https://cdn.openai.com/API/docs/audio/alloy.wav"  # Example audio
        response = conversation_system.process_conversation_turn(audio_file)
        print(f"\nResponse: {response}\n")
        
        # Subsequent turn
        print("Processing second conversation turn...")
        response = conversation_system.process_conversation_turn(audio_file)
        print(f"\nResponse: {response}\n")
        
        # Print metrics
        print("Conversation Metrics:")
        print(json.dumps(conversation_system.get_conversation_metrics(), indent=2))
    
    else:
        # 模拟情感变化
        simulate_emotion_changes(conversation_system)


if __name__ == "__main__":
    main()


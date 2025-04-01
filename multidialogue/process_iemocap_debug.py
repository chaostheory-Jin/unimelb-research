import os
import glob
import json
import re
from typing import List, Dict, Set

class IEMOCAPProcessor:
    def __init__(self, data_path: str):
        """
        初始化处理器
        data_path: IEMOCAP数据集根目录
        """
        self.data_path = data_path
        self.emotion_mapping = {
            'ang': 'angry',
            'hap': 'happy',
            'exc': 'excited',
            'sad': 'sad',
            'neu': 'neutral',
            'fru': 'frustrated',
            'sur': 'surprised',
            'fea': 'fear',
            'dis': 'disgust',
            'oth': 'other'
        }
        # 我们可以根据需要选择要保留的情绪类别
        self.target_emotions = {'angry', 'happy', 'sad', 'neutral', 'excited', 'frustrated'}
        
        # 添加调试标志
        self.debug = True
    
    def debug_print(self, message):
        """调试输出函数"""
        if self.debug:
            print(f"[DEBUG] {message}")
    
    def scan_directory_structure(self):
        """扫描并打印目录结构以帮助调试"""
        print(f"扫描数据集目录结构: {self.data_path}")
        
        # 检查根目录是否存在
        if not os.path.exists(self.data_path):
            print(f"错误: 根目录不存在: {self.data_path}")
            return False
            
        # 检查Session目录
        session_dirs = glob.glob(os.path.join(self.data_path, "Session*"))
        if not session_dirs:
            print(f"错误: 未找到Session目录")
            # 打印根目录的内容
            print(f"根目录内容: {os.listdir(self.data_path)}")
            return False
            
        print(f"找到 {len(session_dirs)} 个Session目录: {session_dirs}")
        
        # 检查第一个Session的结构
        if session_dirs:
            sample_session = session_dirs[0]
            print(f"样本Session目录内容: {os.listdir(sample_session)}")
            
            # 检查dialog目录
            dialog_dir = os.path.join(sample_session, "dialog")
            if os.path.exists(dialog_dir):
                print(f"Dialog目录内容: {os.listdir(dialog_dir)}")
                
                # 检查EmoEvaluation目录
                emo_dir = os.path.join(dialog_dir, "EmoEvaluation")
                if os.path.exists(emo_dir):
                    emo_files = glob.glob(os.path.join(emo_dir, "*.txt"))
                    print(f"找到 {len(emo_files)} 个情绪评估文件")
                    if emo_files:
                        # 显示第一个文件的前几行
                        with open(emo_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()[:10]
                            print(f"情绪评估文件示例 ({emo_files[0]}):")
                            for line in lines:
                                print(f"  {line.strip()}")
                else:
                    print(f"警告: 未找到EmoEvaluation目录: {emo_dir}")
                
                # 检查transcriptions目录
                trans_dir = os.path.join(dialog_dir, "transcriptions")
                if os.path.exists(trans_dir):
                    trans_files = glob.glob(os.path.join(trans_dir, "*.txt"))
                    print(f"找到 {len(trans_files)} 个转录文件")
                    if trans_files:
                        # 显示第一个文件的前几行
                        with open(trans_files[0], 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()[:10]
                            print(f"转录文件示例 ({trans_files[0]}):")
                            for line in lines:
                                print(f"  {line.strip()}")
                else:
                    print(f"警告: 未找到transcriptions目录: {trans_dir}")
            else:
                print(f"警告: 未找到dialog目录: {dialog_dir}")
        
        return True
        
    def read_emotion_labels(self, eval_file: str) -> Dict[str, str]:
        """
        读取情绪标签文件（EmoEvaluation文件）
        返回utterance_id到情绪标签的映射
        """
        emotion_labels = {}
        
        if not os.path.exists(eval_file):
            self.debug_print(f"情绪评估文件不存在: {eval_file}")
            return emotion_labels
        
        try:
            with open(eval_file, 'r', encoding='utf-8', errors='ignore') as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    # 尝试不同的格式模式
                    if '[' in line:
                        # 尝试检测不同的标签格式
                        try:
                            # 格式 1: [start-end] utterance_id emotion [V/A values]
                            matches = re.search(r'\[(.*?)\]\s+(\S+)\s+(\S+)', line)
                            if matches:
                                utterance_id = matches.group(2)
                                emotion = matches.group(3)
                                emotion = self.emotion_mapping.get(emotion.lower(), 'other')
                                emotion_labels[utterance_id] = emotion
                                continue
                                
                            # 格式 2: 使用制表符分隔
                            parts = line.strip().split('\t')
                            if len(parts) >= 3:
                                utterance_id = parts[1]
                                emotion = parts[2]
                                emotion = self.emotion_mapping.get(emotion.lower(), 'other')
                                emotion_labels[utterance_id] = emotion
                        except Exception as e:
                            self.debug_print(f"在第{line_num}行解析情绪标签时出错: {e}, 行内容: {line}")
        except Exception as e:
            self.debug_print(f"读取情绪评估文件时出错: {e}")
                    
        self.debug_print(f"从 {eval_file} 读取了 {len(emotion_labels)} 个情绪标签")
        return emotion_labels
    
    def read_transcript(self, transcript_file: str) -> List[Dict]:
        """
        读取对话转录文件
        transcript_file: 转录文件路径
        返回包含话语信息的列表
        """
        utterances = []
        
        if not os.path.exists(transcript_file):
            self.debug_print(f"转录文件不存在: {transcript_file}")
            return utterances
        
        try:
            with open(transcript_file, 'r', encoding='utf-8', errors='ignore') as f:
                line_num = 0
                for line in f:
                    line_num += 1
                    try:
                        # 尝试多种可能的格式
                        if line.startswith("Ses"):
                            # 格式 1: Ses01F_impro01_F000: text
                            if ":" in line:
                                parts = line.strip().split(":", 1)
                                if len(parts) >= 2:
                                    utterance_id = parts[0].strip()
                                    text = parts[1].strip()
                                    
                                    # 提取说话者信息 (通常是M或F)
                                    speaker = utterance_id[-4] if len(utterance_id) >= 4 else "?"
                                    
                                    utterance = {
                                        'utterance_id': utterance_id,
                                        'speaker': speaker,
                                        'content': text,
                                    }
                                    utterances.append(utterance)
                        elif re.match(r'^\[.*\]', line):
                            # 格式 2: [M/F] text
                            match = re.match(r'^\[(M|F)\](.*)', line)
                            if match:
                                speaker = match.group(1)
                                text = match.group(2).strip()
                                # 生成一个临时ID
                                utterance_id = f"unknown_{len(utterances)}"
                                
                                utterance = {
                                    'utterance_id': utterance_id,
                                    'speaker': speaker,
                                    'content': text,
                                }
                                utterances.append(utterance)
                    except Exception as e:
                        self.debug_print(f"在第{line_num}行解析转录时出错: {e}, 行内容: {line}")
        except Exception as e:
            self.debug_print(f"读取转录文件时出错: {e}")
                    
        self.debug_print(f"从 {transcript_file} 读取了 {len(utterances)} 个话语")
        return utterances
    
    def search_files(self, session_id: str):
        """搜索给定session中的文件"""
        session_path = os.path.join(self.data_path, f'Session{session_id}')
        
        if not os.path.exists(session_path):
            print(f"Session路径不存在: {session_path}")
            return [], []
            
        # 搜索所有可能的目录
        possible_dialog_dirs = [
            os.path.join(session_path, "dialog"),
            os.path.join(session_path, "dialogs"),
            session_path
        ]
        
        eval_files = []
        trans_files = []
        
        for dialog_dir in possible_dialog_dirs:
            if not os.path.exists(dialog_dir):
                continue
                
            # 搜索情绪评估文件
            for root, dirs, files in os.walk(dialog_dir):
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        
                        # 检查文件内容来判断类型
                        try:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                                content = f.read(1000)  # 读取前1000个字符
                                
                                # 情绪评估文件通常包含特定的标记
                                if '[' in content and ']' in content and any(emo in content.lower() for emo in ['ang', 'hap', 'sad', 'neu']):
                                    eval_files.append(file_path)
                                    
                                # 转录文件通常包含对话内容
                                elif "Ses" in content and ":" in content:
                                    trans_files.append(file_path)
                        except:
                            pass
                            
        print(f"在Session{session_id}中找到 {len(eval_files)} 个情绪评估文件和 {len(trans_files)} 个转录文件")
        return eval_files, trans_files
    
    def process_files(self, eval_files: List[str], trans_files: List[str]) -> List[Dict]:
        """
        处理情绪评估文件和转录文件
        """
        all_utterances = []
        all_emotion_labels = {}
        
        # 读取所有情绪标签
        for eval_file in eval_files:
            labels = self.read_emotion_labels(eval_file)
            all_emotion_labels.update(labels)
            
        # 读取所有转录
        for trans_file in trans_files:
            utterances = self.read_transcript(trans_file)
            all_utterances.extend(utterances)
            
        self.debug_print(f"共收集到 {len(all_utterances)} 个话语和 {len(all_emotion_labels)} 个情绪标签")
        
        # 将情绪标签添加到话语中
        for utterance in all_utterances:
            utterance['emotion'] = all_emotion_labels.get(utterance['utterance_id'], 'unknown')
            
        # 按对话ID分组
        dialogue_groups = {}
        for utterance in all_utterances:
            # 从utterance_id中提取对话ID
            parts = utterance['utterance_id'].split('_')
            if len(parts) >= 2:
                dialogue_id = f"{parts[0]}_{parts[1]}"
                
                if dialogue_id not in dialogue_groups:
                    dialogue_groups[dialogue_id] = []
                    
                dialogue_groups[dialogue_id].append(utterance)
                
        # 构建对话
        dialogues = []
        for dialogue_id, turns in dialogue_groups.items():
            # 按utterance_id排序
            turns.sort(key=lambda x: x['utterance_id'])
            
            # 过滤掉情绪为unknown的轮次
            valid_turns = [turn for turn in turns if turn['emotion'] in self.target_emotions]
            
            # 只保留至少有2轮对话的对话
            if len(valid_turns) >= 2:
                dialogue = {
                    'dialogue_id': dialogue_id,
                    'turns': [
                        {
                            'speaker': turn['speaker'],
                            'text': turn['content'],
                            'emotion': turn['emotion'],
                            'utterance_id': turn['utterance_id']
                        } for turn in valid_turns
                    ]
                }
                dialogues.append(dialogue)
                
        return dialogues
    
    def format_for_finetuning(self, dialogues: List[Dict]) -> List[Dict]:
        """
        将对话格式化为适合微调的格式
        """
        formatted_data = []
        
        for dialogue in dialogues:
            turns = dialogue['turns']
            
            # 为每个轮次创建一个样本，包含之前的上下文
            for i in range(1, len(turns)):
                # 当前轮次是要预测情绪的轮次
                current_turn = turns[i]
                
                # 收集历史轮次作为上下文
                history = turns[:i]
                
                # 构建样本
                sample = {
                    'dialogue_id': dialogue['dialogue_id'],
                    'context': [
                        {
                            'speaker': turn['speaker'],
                            'text': turn['text'],
                            'emotion': turn['emotion']
                        } for turn in history
                    ],
                    'response': {
                        'speaker': current_turn['speaker'],
                        'text': current_turn['text'],
                        'emotion': current_turn['emotion']
                    },
                    'utterance_id': current_turn['utterance_id']
                }
                
                formatted_data.append(sample)
        
        return formatted_data
    
    def save_to_json(self, data: List[Dict], output_path: str):
        """
        将处理后的数据保存为JSON格式
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump({
                'data': data,
                'metadata': {
                    'total_samples': len(data),
                    'emotion_types': list(self.target_emotions)
                }
            }, f, indent=2, ensure_ascii=False)
    
    def get_stats(self, data: List[Dict]) -> Dict:
        """
        获取数据集统计信息
        """
        emotion_counts = {}
        dialogue_count = set()
        speakers = set()
        
        for sample in data:
            # 计算每种情绪的样本数量
            emotion = sample['response']['emotion']
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # 计算对话数量
            dialogue_count.add(sample['dialogue_id'])
            
            # 统计说话者
            speakers.add(sample['response']['speaker'])
            for turn in sample['context']:
                speakers.add(turn['speaker'])
                
        return {
            'total_samples': len(data),
            'total_dialogues': len(dialogue_count),
            'emotion_distribution': emotion_counts,
            'speakers': list(speakers)
        }

def main():
    # 设置路径
    iemocap_path = r"C:\Users\luoya\Desktop\unimelb-research\dataset\IEMOCAP\IEMOCAP_full_release"
    output_path = r"C:\Users\luoya\Desktop\unimelb-research\dataset\IEMOCAP\processed_iemocap_dialogue.json"
    
    # 初始化处理器
    processor = IEMOCAPProcessor(iemocap_path)
    
    # 扫描目录结构
    if not processor.scan_directory_structure():
        print("无法正确识别IEMOCAP数据集结构，处理终止")
        return
    
    # 处理所有session
    all_dialogues = []
    for session_id in range(1, 6):  # IEMOCAP有5个session
        print(f"处理Session {session_id}...")
        
        # 搜索文件
        eval_files, trans_files = processor.search_files(str(session_id))
        
        if not eval_files or not trans_files:
            print(f"警告: Session {session_id} 没有找到足够的文件")
            continue
            
        # 处理文件
        session_dialogues = processor.process_files(eval_files, trans_files)
        all_dialogues.extend(session_dialogues)
    
    print(f"共收集到 {len(all_dialogues)} 个对话")
    
    if all_dialogues:
        # 将对话格式化为适合微调的格式
        formatted_data = processor.format_for_finetuning(all_dialogues)
        
        print(f"生成了 {len(formatted_data)} 个训练样本")
        
        # 获取数据集统计信息
        stats = processor.get_stats(formatted_data)
        print("数据集统计信息:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        # 保存处理后的数据
        processor.save_to_json(formatted_data, output_path)
        
        print(f"处理完成！数据已保存到 {output_path}")
    else:
        print("没有处理到任何对话数据，请检查IEMOCAP数据集结构")

if __name__ == "__main__":
    main() 
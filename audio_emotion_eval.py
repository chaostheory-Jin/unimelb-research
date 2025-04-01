import os
import glob

class AudioEmotionEval:
    def find_audio_file(self, utterance_id: str, iemocap_root: str) -> str:
        """查找音频文件的可能路径"""
        parts = utterance_id.split('_')
        if len(parts) < 3:
            raise ValueError(f"无效的话语ID格式: {utterance_id}")

        session_num = parts[0][3:4]  # 从Ses01F中提取1
        session = f"Session{session_num}"
        dialogue = f"{parts[0]}_{parts[1]}"  # 例如：Ses01F_impro01
        audio_filename = f"{utterance_id}.wav"

        # 添加更多详细的调试信息
        print(f"\n调试信息:")
        print(f"话语ID: {utterance_id}")
        print(f"会话号: {session_num}")
        print(f"会话目录: {session}")
        print(f"对话ID: {dialogue}")
        print(f"音频文件名: {audio_filename}")
        print(f"IEMOCAP根目录: {iemocap_root}")

        # 检查根目录是否存在
        if not os.path.exists(iemocap_root):
            print(f"错误: IEMOCAP根目录不存在: {iemocap_root}")
            raise FileNotFoundError(f"IEMOCAP根目录不存在: {iemocap_root}")

        # 定义可能的路径模式
        patterns = [
            # 标准路径模式
            os.path.join(iemocap_root, session, "sentences", "wav", dialogue, audio_filename),
            os.path.join(iemocap_root, session, "wav", dialogue, audio_filename),
            os.path.join(iemocap_root, session, "wav", audio_filename),
            os.path.join(iemocap_root, session, "sentences", audio_filename),
            os.path.join(iemocap_root, session, audio_filename),
            # 添加新的路径模式
            os.path.join(iemocap_root, session, "dialog", "wav", dialogue, audio_filename),
            os.path.join(iemocap_root, "wav", session, dialogue, audio_filename),
            os.path.join(iemocap_root, "audio", session, dialogue, audio_filename)
        ]

        # 检查每个可能的路径
        print("\n尝试查找以下路径:")
        for pattern in patterns:
            normalized_pattern = os.path.normpath(pattern)
            print(f"检查路径: {normalized_pattern}")
            if os.path.exists(normalized_pattern):
                print(f"✓ 找到文件!")
                return normalized_pattern
            else:
                print(f"✗ 文件不存在")

        # 如果上述路径都不存在，使用glob进行递归搜索
        print("\n开始递归搜索...")
        session_path = os.path.join(iemocap_root, session)
        if os.path.exists(session_path):
            # 使用多个glob模式
            glob_patterns = [
                os.path.join(session_path, f"**/{audio_filename}"),
                os.path.join(session_path, f"**/{dialogue}/**/{audio_filename}"),
                os.path.join(session_path, f"**/*{utterance_id}*.wav")
            ]
            
            for glob_pattern in glob_patterns:
                print(f"使用glob模式搜索: {glob_pattern}")
                matches = glob.glob(glob_pattern, recursive=True)
                if matches:
                    print(f"✓ 通过glob找到文件: {matches[0]}")
                    return matches[0]
                else:
                    print(f"✗ 未找到匹配文件")

        # 列出会话目录的内容以帮助调试
        print("\n会话目录内容:")
        try:
            session_dir = os.path.join(iemocap_root, session)
            if os.path.exists(session_dir):
                for root, dirs, files in os.walk(session_dir):
                    print(f"\n目录: {root}")
                    if dirs:
                        print("子目录:", dirs)
                    if files:
                        print("文件:", [f for f in files if f.endswith('.wav')])
        except Exception as e:
            print(f"列出目录内容时出错: {e}")

        raise FileNotFoundError(f"找不到音频文件: {utterance_id}\n请检查IEMOCAP数据集结构是否正确") 
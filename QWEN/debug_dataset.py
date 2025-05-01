import os
import torch
import librosa
import numpy as np
import random
import json
import traceback
from tqdm import tqdm
import logging
from transformers import AutoProcessor, Seq2SeqTrainingArguments

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_training_args():
    """测试训练参数的正确配置"""
    logger.info("测试 Seq2SeqTrainingArguments...")
    
    try:
        # 获取转换器库版本
        import transformers
        logger.info(f"Transformers 版本: {transformers.__version__}")
        
        # 检查参数文档
        logger.info("查看 Seq2SeqTrainingArguments 可用参数:")
        
        # 获取 Seq2SeqTrainingArguments 的签名
        from inspect import signature
        sig = signature(Seq2SeqTrainingArguments)
        params = sig.parameters
        logger.info(f"参数列表: {list(params.keys())}")
        
        # 创建一个最小配置
        args = Seq2SeqTrainingArguments(
            output_dir="./test_output",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
        )
        logger.info("基本配置成功!")
        
        # 尝试添加评估策略
        try:
            args = Seq2SeqTrainingArguments(
                output_dir="./test_output",
                per_device_train_batch_size=1,
                per_device_eval_batch_size=1,
                evaluation_strategy="steps",
            )
            logger.info("evaluation_strategy 参数可用!")
        except TypeError as e:
            logger.error(f"evaluation_strategy 参数错误: {e}")
            logger.info("尝试使用 eval_strategy 替代...")
            
            try:
                args = Seq2SeqTrainingArguments(
                    output_dir="./test_output",
                    per_device_train_batch_size=1,
                    per_device_eval_batch_size=1,
                    eval_strategy="steps",
                )
                logger.info("eval_strategy 参数可用!")
            except TypeError:
                logger.error("eval_strategy 也不可用!")
        
        return True
        
    except Exception as e:
        logger.error(f"测试训练参数时出错: {e}")
        logger.error(traceback.format_exc())
        return False

def test_audio_processing():
    """测试音频处理功能"""
    logger.info("\n测试音频处理功能...")
    
    # 测试音频处理库
    try:
        import librosa
        logger.info(f"Librosa 版本: {librosa.__version__}")
        
        # 检查音频处理依赖
        try:
            import soundfile
            logger.info(f"SoundFile 可用: {soundfile.__version__}")
        except ImportError:
            logger.warning("SoundFile 不可用，将使用 audioread 后备方案")
        
        try:
            import audioread
            logger.info(f"Audioread 可用: {audioread.__version__}")
        except ImportError:
            logger.warning("Audioread 不可用")
        
        # 尝试查找 ffmpeg
        try:
            import subprocess
            result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        except:
            logger.warning("FFMPEG 似乎未安装")
            
        return True
    
    except Exception as e:
        logger.error(f"测试音频处理时出错: {e}")
        logger.error(traceback.format_exc())
        return False

def sample_small_dataset(json_file, output_file, sample_size=100):
    """从JSON文件中抽样生成一个小数据集"""
    logger.info(f"从 {json_file} 抽样 {sample_size} 个样本...")
    
    try:
        # 加载JSON数据
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"原始JSON文件中共有 {len(data)} 个样本")
        
        # 随机抽样
        if len(data) > sample_size:
            small_dataset = random.sample(data, sample_size)
        else:
            small_dataset = data
            logger.warning(f"数据集只有 {len(data)} 个样本，小于请求的 {sample_size}")
        
        # 保存小样本数据集
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(small_dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"已保存 {len(small_dataset)} 个样本到 {output_file}")
        return True
    
    except Exception as e:
        logger.error(f"抽样数据集时出错: {e}")
        logger.error(traceback.format_exc())
        return False

def fix_finetune_script():
    """修复finetune.py脚本中的问题"""
    logger.info("\n修复finetune.py脚本...")
    
    try:
        # 备份原始文件
        import shutil
        shutil.copy("finetune.py", "finetune.py.bak")
        logger.info("已备份 finetune.py 到 finetune.py.bak")
        
        # 读取文件内容
        with open("finetune.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        # 修复 evaluation_strategy 参数
        content = content.replace(
            "evaluation_strategy=\"steps\",", 
            "# evaluation_strategy=\"steps\",  # 参数名称可能在不同版本中不同"
        )
        
        # 修复 __getitem__ 方法中的异常处理
        content = content.replace(
            """        except Exception as e:
            logger.error(f"处理样本 {sample['id']} 时出错: {e}")
            # 返回一个空白样本以避免批处理错误
            if len(self.samples) > 1:
                return self.__getitem__(random.randint(0, len(self.samples)-1))
            else:
                # 如果只有一个样本且处理失败，返回一个空结构
                return {"id": sample["id"], "error": str(e)}""",
            
            """        except Exception as e:
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
                return {"id": sample["id"], "error": str(e)}"""
        )
        
        # 修复加载音频文件的代码
        content = content.replace(
            """            # 加载音频文件
            audio, sr = librosa.load(sample["audio_path"], sr=self.sample_rate)""",
            
            """            # 检查文件是否存在
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
                raise ValueError(f"加载的音频数据为空: {sample['audio_path']}")"""
        )
        
        # 写入修改后的内容
        with open("finetune_fixed.py", "w", encoding="utf-8") as f:
            f.write(content)
        
        logger.info("已生成修复后的脚本 finetune_fixed.py")
        return True
        
    except Exception as e:
        logger.error(f"修复脚本时出错: {e}")
        logger.error(traceback.format_exc())
        return False

def create_minimal_training_script():
    """创建一个最小的训练脚本"""
    logger.info("\n创建最小训练脚本...")
    
    script = """
import os
import torch
import numpy as np
import logging
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
    TaskType
)
from datasets import Dataset

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置随机种子
set_seed(42)

# 创建一个小的测试数据集
def create_dummy_dataset():
    return Dataset.from_dict({
        "input_ids": [torch.randint(0, 1000, (50,)).numpy() for _ in range(5)],
        "attention_mask": [torch.ones(50).numpy() for _ in range(5)],
        "labels": [torch.randint(0, 1000, (20,)).numpy() for _ in range(5)]
    })

def main():
    logger.info("开始测试 Qwen2-Audio 模型训练...")
    
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
        torch_dtype=torch.float16,
        device_map="auto"
    )
    
    # 配置LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        bias="none",
    )
    
    # 应用LoRA
    model = get_peft_model(model, peft_config)
    
    # 创建虚拟数据集
    train_dataset = create_dummy_dataset()
    val_dataset = create_dummy_dataset()
    
    # 创建数据整理器
    data_collator = DataCollatorForSeq2Seq(
        processor.tokenizer,
        model=model,
        pad_to_multiple_of=8,
    )
    
    # 定义训练参数 - 使用简化的参数
    training_args = Seq2SeqTrainingArguments(
        output_dir="./qwen2_test_output",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        num_train_epochs=1,
        save_steps=10,
        logging_steps=5,
        save_strategy="steps",  # 尝试使用save_strategy而不是evaluation_strategy
        report_to="none",
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
    
    logger.info("测试完成!")

if __name__ == "__main__":
    main()
"""
    
    try:
        with open("minimal_train.py", "w", encoding="utf-8") as f:
            f.write(script)
        
        logger.info("已生成最小训练脚本 minimal_train.py")
        return True
    
    except Exception as e:
        logger.error(f"创建最小训练脚本时出错: {e}")
        logger.error(traceback.format_exc())
        return False

def main():
    logger.info("开始调试 Qwen2-Audio 微调问题...")
    
    # 测试训练参数
    test_training_args()
    
    # 测试音频处理
    test_audio_processing()
    
    # 创建小型数据集
    sample_small_dataset("iemocap_ambiguous.json", "iemocap_small.json", 100)
    
    # 修复主脚本
    fix_finetune_script()
    
    # 创建最小训练脚本
    create_minimal_training_script()
    
    logger.info("\n调试完成! 推荐的解决方案:")
    logger.info("1. 安装必要的依赖: pip install soundfile audioread ffmpeg-python")
    logger.info("2. 使用修复后的 finetune_fixed.py 脚本")
    logger.info("3. 或者先运行 minimal_train.py 测试基本功能")
    logger.info("4. 或者使用 iemocap_small.json 作为小型测试数据集")

if __name__ == "__main__":
    main()
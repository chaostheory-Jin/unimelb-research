# 创建一个新文件debug_fix.py

import os
import json
import shutil
from pathlib import Path
import logging
import re

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_json_data():
    """修复JSON数据，过滤掉不正确的文件路径"""
    json_file = "iemocap_ambiguous.json"
    fixed_json = "iemocap_fixed.json"
    
    logger.info(f"修复JSON数据文件: {json_file}")
    
    # 备份原始文件
    if not os.path.exists(f"{json_file}.bak"):
        shutil.copy(json_file, f"{json_file}.bak")
        logger.info(f"已备份原始文件到: {json_file}.bak")
    
    # 加载JSON数据
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"原始数据包含 {len(data)} 个样本")
    
    # 过滤掉问题文件
    problem_patterns = [
        r'\.pk$',  # 以.pk结尾的文件
        r'Ses04F_impro08_F01',  # 具体的问题文件
        r'Ses02F_impro07_F01',
        r'Ses01F_script01_3_F02',
        r'Ses02M_script03_2_F04'
    ]
    
    valid_samples = []
    removed_samples = []
    
    for item in data:
        # 获取音频路径
        audio_path = item.get('audio_path') or item.get('path') or item.get('wav_path') or item.get('audio')
        if not audio_path:
            removed_samples.append(item)
            continue
        
        # 检查是否匹配问题模式
        is_problem = False
        for pattern in problem_patterns:
            if re.search(pattern, audio_path):
                is_problem = True
                removed_samples.append(item)
                break
        
        if not is_problem:
            # 确保路径以.wav结尾
            if not audio_path.lower().endswith('.wav'):
                if '.' not in os.path.basename(audio_path):
                    # 如果没有扩展名，添加.wav
                    item['audio'] = audio_path + '.wav'
            valid_samples.append(item)
    
    logger.info(f"过滤后保留 {len(valid_samples)} 个样本，移除 {len(removed_samples)} 个样本")
    
    # 保存修复后的数据
    with open(fixed_json, 'w', encoding='utf-8') as f:
        json.dump(valid_samples, f, ensure_ascii=False)
    
    logger.info(f"已保存修复后的数据到: {fixed_json}")
    return fixed_json

def fix_training_script():
    """修复训练脚本中的问题"""
    script_file = "finetune_fixed.py"
    new_script = "finetune_debug.py"
    
    logger.info(f"修复训练脚本: {script_file}")
    
    # 备份原始文件
    if not os.path.exists(f"{script_file}.bak"):
        shutil.copy(script_file, f"{script_file}.bak")
        logger.info(f"已备份原始脚本到: {script_file}.bak")
    
    # 读取脚本内容
    with open(script_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 1. 修复训练参数
    content = content.replace(
        """training_args = Seq2SeqTrainingArguments(
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
    evaluation_strategy="steps",
    save_strategy="steps",
    load_best_model_at_end=True,
    push_to_hub=False,
    fp16=True,
    generation_max_length=50,
    report_to="tensorboard",
    save_total_limit=3,  # 只保存最好的3个检查点
    logging_first_step=True
)""",
        """training_args = Seq2SeqTrainingArguments(
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
    # 确保评估策略和保存策略一致
    evaluation_strategy="steps",
    save_strategy="steps",
    # 关闭最佳模型加载以简化训练
    load_best_model_at_end=False,
    push_to_hub=False,
    fp16=True,
    generation_max_length=50,
    report_to="tensorboard",
    save_total_limit=3,  # 只保存最好的3个检查点
    logging_first_step=True
)"""
    )
    
    # 2. 修复数据集转换函数，添加decoder_input_ids的处理
    content = content.replace(
        """def loader_to_dataset(dataloader):
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
    return Dataset.from_list(all_data)""",
        
        """def loader_to_dataset(dataloader):
    all_data = []
    error_count = 0
    
    for batch in tqdm(dataloader, desc="转换数据集"):
        if batch is None:
            continue
            
        try:
            # 移除批次维度
            for i in range(len(batch["id"])):
                try:
                    # 不包含decoder_input_ids, Qwen2Audio模型不接受此参数
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
                except Exception as e:
                    error_count += 1
                    logger.error(f"处理批次中的样本 {batch['id'][i]} 时出错: {e}")
        except Exception as e:
            error_count += 1
            logger.error(f"处理批次时出错: {e}")
    
    logger.info(f"数据集转换完成: {len(all_data)} 个有效样本, {error_count} 个错误")
    return Dataset.from_list(all_data)"""
    )
    
    # 3. 修复音频处理部分，确保只处理.wav文件
    content = content.replace(
        """            # 检查文件是否存在
            if not os.path.exists(sample["audio_path"]):
                logger.error(f"音频文件不存在: {sample['audio_path']}")
                raise FileNotFoundError(f"音频文件不存在: {sample['audio_path']}")""",
                
        """            # 检查文件是否存在且是wav文件
            audio_path = sample["audio_path"]
            if not audio_path.lower().endswith('.wav'):
                logger.error(f"不是WAV文件: {audio_path}")
                raise ValueError(f"只支持WAV文件，但获得了: {audio_path}")
                
            if not os.path.exists(audio_path):
                logger.error(f"音频文件不存在: {audio_path}")
                raise FileNotFoundError(f"音频文件不存在: {audio_path}")"""
    )
    
    # 4. 修复loader_to_dataset调用处，添加更多错误处理
    content = content.replace(
        """    # 转换为Dataset格式
    train_dataset = loader_to_dataset(train_loader)
    val_dataset = loader_to_dataset(val_loader)""",
        
        """    # 转换为Dataset格式
    try:
        logger.info("开始转换训练数据集...")
        train_dataset = loader_to_dataset(train_loader)
        logger.info("开始转换验证数据集...")
        val_dataset = loader_to_dataset(val_loader)
        
        # 检查数据集大小
        if len(train_dataset) == 0 or len(val_dataset) == 0:
            logger.error(f"数据集为空! 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
            raise ValueError("数据集为空，无法继续训练")
            
        logger.info(f"最终数据集大小 - 训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}")
    except Exception as e:
        logger.error(f"转换数据集时出错: {e}")
        raise"""
    )
    
    # 修复main函数中的JSON文件路径
    content = content.replace(
        """    # 加载数据集
    json_file = "iemocap_ambiguous.json"
    iemocap_root = r"C:\Users\luoya\Desktop\unimelb-research\dataset\IEMOCAP\""""
        
        """    # 加载数据集
    json_file = "iemocap_fixed.json"  # 使用修复后的JSON文件
    iemocap_root = r"C:\Users\luoya\Desktop\unimelb-research\dataset\IEMOCAP\""""
    )
    
    # 5. 添加自定义训练器以解决decoder_input_ids的问题
    content = content.replace(
        """    # 创建训练器
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )""",
        
        """    # 创建自定义训练器来处理输入参数
    class CustomSeq2SeqTrainer(Seq2SeqTrainer):
        def compute_loss(self, model, inputs, num_items_in_batch=None, return_outputs=False):
            # 移除不支持的参数
            if 'decoder_input_ids' in inputs:
                del inputs['decoder_input_ids']
                
            return super().compute_loss(model, inputs, num_items_in_batch, return_outputs)
    
    # 使用自定义训练器
    trainer = CustomSeq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )"""
    )
    
    # 保存修改后的脚本
    with open(new_script, 'w', encoding='utf-8') as f:
        f.write(content)
    
    logger.info(f"已保存修复后的脚本到: {new_script}")
    return new_script

def main():
    """主函数，执行所有修复工作"""
    logger.info("开始修复微调脚本和数据...")
    
    # 步骤1: 修复JSON数据
    fixed_json = fix_json_data()
    
    # 步骤2: 修复训练脚本
    fixed_script = fix_training_script()
    
    logger.info(f"\n修复完成！请按以下步骤操作:")
    logger.info(f"1. 检查修复后的JSON文件: {fixed_json}")
    logger.info(f"2. 使用新的训练脚本: python {fixed_script}")
    logger.info(f"3. 如仍有问题，可检查日志详情并进一步调整")

if __name__ == "__main__":
    main()
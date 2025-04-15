
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

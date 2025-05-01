from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import Qwen2AudioForConditionalGeneration, AutoProcessor
import torch

# 设置torch保存模型时使用BF16格式，避免精度损失
torch.set_default_dtype(torch.bfloat16)

# 检查CUDA是否可用
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct", trust_remote_code=True)
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B-Instruct", 
    device_map="auto", 
    trust_remote_code=True
)

conversation = [
    {'role': 'system', 'content': 'You are a helpful assistant.'}, 
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/glass-breaking-151256.mp3"},
        {"type": "text", "text": "What's that sound?"},
    ]},
    {"role": "assistant", "content": "It is the sound of glass shattering."},
    {"role": "user", "content": [
        {"type": "text", "text": "What can you do when you hear that?"},
    ]},
    {"role": "assistant", "content": "Stay alert and cautious, and check if anyone is hurt or if there is any damage to property."},
    {"role": "user", "content": [
        {"type": "audio", "audio_url": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2-Audio/audio/1272-128104-0000.flac"},
        {"type": "text", "text": "What does the person say?"},
    ]},
]
text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
audios = []
for message in conversation:
    if isinstance(message["content"], list):
        for ele in message["content"]:
            if ele["type"] == "audio":
                try:
                    # 指定采样率
                    audio_data, sr = librosa.load(
                        BytesIO(urlopen(ele['audio_url']).read()), 
                        sr=processor.feature_extractor.sampling_rate
                    )
                    audios.append(audio_data)
                    print(f"成功加载音频，长度：{len(audio_data)}，采样率：{sr}")
                except Exception as e:
                    print(f"加载音频时出错: {e}")

# 使用audio参数而不是audios
print("处理输入中...")
inputs = processor(text=text, audio=audios, return_tensors="pt", padding=True)

# 重要：将所有输入移到GPU
for key in inputs:
    if torch.is_tensor(inputs[key]):
        inputs[key] = inputs[key].to(device)
        print(f"将 {key} 移到 {device}，形状: {inputs[key].shape}")

print("开始生成响应...")
# 使用max_new_tokens而不是max_length
generate_ids = model.generate(
    **inputs, 
    max_new_tokens=100,
    do_sample=False,
    num_beams=1  # 简化生成过程
)
generate_ids = generate_ids[:, inputs.input_ids.size(1):]

response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(f"Response: {response}")

# 保存模型
print("保存模型中...")
model.save_pretrained("./qwen2_audio_model")
processor.save_pretrained("./qwen2_audio_processor")
print("模型保存完成!")

# 在evaluate_model函数中
if true_emotion == "frustrated":
    # 添加针对frustrated的特定提示
    prompts.append("<|audio_bos|><|AUDIO|><|audio_eos|>Is this person frustrated or expressing a different emotion?")
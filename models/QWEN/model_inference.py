import os
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# 设置环境变量来禁用 symlinks 警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 设置本地保存路径
local_model_path = "./qwen2_audio_model"
local_processor_path = "./qwen2_audio_processor"

# 首先下载并保存模型
print("正在下载并保存模型...")
model = Qwen2AudioForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2-Audio-7B",
    trust_remote_code=True,
    device_map="auto"
)
model.save_pretrained(local_model_path)

# 保存处理器
print("正在保存处理器...")
processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2-Audio-7B",
    trust_remote_code=True
)
processor.save_pretrained(local_processor_path)

print(f"模型已保存到: {local_model_path}")
print(f"处理器已保存到: {local_processor_path}")

# 准备输入
prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"

# 指定采样率
sampling_rate = processor.feature_extractor.sampling_rate
audio, sr = librosa.load(
    BytesIO(urlopen(url).read()), 
    sr=sampling_rate
)

# 使用 audio 而不是 audios
inputs = processor(
    text=prompt, 
    audio=audio,
    sampling_rate=sampling_rate,
    return_tensors="pt"
)

# 生成输出
generated_ids = model.generate(**inputs, max_length=256)
generated_ids = generated_ids[:, inputs.input_ids.size(1):]
response = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
print(response)
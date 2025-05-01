import os
from io import BytesIO
from urllib.request import urlopen
import librosa
from transformers import AutoProcessor, Qwen2AudioForConditionalGeneration

# 设置环境变量来禁用 symlinks 警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 从本地加载模型和处理器
local_model_path = "./qwen2_audio_model"
local_processor_path = "./qwen2_audio_processor"

model = Qwen2AudioForConditionalGeneration.from_pretrained(
    local_model_path,
    trust_remote_code=True,
    device_map="auto"
)
processor = AutoProcessor.from_pretrained(
    local_processor_path,
    trust_remote_code=True
)

# 测试代码
prompt = "<|audio_bos|><|AUDIO|><|audio_eos|>Generate the caption in English:"
url = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-Audio/glass-breaking-151256.mp3"

# 指定采样率
sampling_rate = processor.feature_extractor.sampling_rate
audio, sr = librosa.load(
    BytesIO(urlopen(url).read()), 
    sr=sampling_rate
)

# 使用处理器处理输入
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
print("输出结果:", response)

#打印模型结构和layer
print(model)
print(model.layers)

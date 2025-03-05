#!/usr/bin/env python3
# install_requirements.py - 为 Apple Silicon M4 芯片安装微调 LLM 所需的依赖

import subprocess
import sys
import platform
import os

def is_apple_silicon():
    """检查是否为 Apple Silicon 芯片"""
    return platform.system() == "Darwin" and platform.machine().startswith(("arm", "aarch"))

def install_requirements():
    """安装所需的 Python 库，针对 M4 芯片优化"""
    # 基础依赖
    base_requirements = [
        "numpy",
        "pandas",
        "tqdm",
        "librosa",
        "soundfile",
        "sentencepiece",
        "protobuf",
        "accelerate",
        "bitsandbytes-apple-silicon",  # 专为 Apple Silicon 优化的版本
        "datasets",
        "peft",
    ]
    
    # 安装基础依赖
    print("Installing base requirements...")
    for package in base_requirements:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")
    
    # 为 Apple Silicon 安装优化的 PyTorch
    if is_apple_silicon():
        print("\nDetected Apple Silicon (M1/M2/M3/M4)...")
        print("Installing PyTorch with MPS support...")
        
        try:
            # 安装针对 Apple Silicon 优化的 PyTorch
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "torch", "torchvision", "torchaudio"
            ])
            
            # 验证 MPS 是否可用
            verify_script = """
import torch
print("PyTorch version:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())
if torch.backends.mps.is_available():
    print("MPS device:", torch.device("mps"))
    x = torch.rand(5, 3).to("mps")
    print("Tensor on MPS:", x.device)
"""
            print("\nVerifying PyTorch MPS support...")
            subprocess.run([sys.executable, "-c", verify_script])
            
        except subprocess.CalledProcessError as e:
            print(f"Error installing PyTorch: {e}")
    else:
        print("\nInstalling standard PyTorch...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "torch", "torchvision", "torchaudio"
        ])
    
    # 安装 transformers
    print("\nInstalling transformers...")
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "transformers>=4.30.0"
    ])
    
    # 创建配置文件以启用 MPS 加速
    if is_apple_silicon():
        config_dir = os.path.expanduser("~/.cache/huggingface/accelerate")
        os.makedirs(config_dir, exist_ok=True)
        
        config_path = os.path.join(config_dir, "default_config.yaml")
        with open(config_path, "w") as f:
            f.write("""compute_environment: LOCAL_MACHINE
distributed_type: 'NO'
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: 'no'
num_machines: 1
num_processes: 1
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
""")
        print(f"\nCreated accelerate config at {config_path}")
    
    print("\nAll required packages installed!")
    print("\nTo use Hugging Face models, you'll need to login with your access token:")
    print("huggingface-cli login")
    
    if is_apple_silicon():
        print("\nOptimization for Apple Silicon M4 complete!")
        print("Your system is now configured to use MPS acceleration.")

if __name__ == "__main__":
    install_requirements() 
@echo off
echo 创建CUDA链接...

set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
set BB_PATH=%USERPROFILE%\Desktop\QWEN\qwen_audio_env\Lib\site-packages\bitsandbytes\

if not exist "%BB_PATH%\cuda_setup" mkdir "%BB_PATH%\cuda_setup"

copy "%CUDA_PATH%\bin\cudart64_*.dll" "%BB_PATH%\"
copy "%CUDA_PATH%\bin\cublas64_*.dll" "%BB_PATH%\"
copy "%CUDA_PATH%\bin\cublasLt64_*.dll" "%BB_PATH%\"

echo 设置完成!
pause
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from huggingface_hub import snapshot_download
from faster_whisper import WhisperModel
def transcribe_audio(audio_file_path):
    # 设置本地缓存目录
    cache_dir = "./whisper_cache"
    print("正在下载/检查模型...")
    try:
        local_model_path = snapshot_download(
            repo_id="Systran/faster-whisper-small",
            repo_type="model",
            cache_dir=cache_dir
        )
        print(f"✅ 模型位置：{local_model_path}")
        model = WhisperModel(local_model_path, device="auto", compute_type="float16")
    except Exception as e:
        print(f"❌ 模型下载或加载失败：{e}")
        print("💡 尝试切换到 CPU 模式并重新下载...")
        # 容错：切换到低配模式
        try:
            model = WhisperModel(local_model_path, device="cpu", compute_type="int8")
        except:
            return None
    # 执行识别
    print("开始语音识别...")
    segments, info = model.transcribe(
        audio_file_path,
        beam_size=5,
        language="en",  # 指定英语
        task="transcribe"
    )
    result_text = ""
    for segment in segments:
        result_text += segment.text + " "
    return result_text.strip()
if __name__ == "__main__":
    # ⚠️ 请务必修改为你本地的实际文件路径
    file_path = r"test1.mp3"

    if os.path.exists(file_path):
        result = transcribe_audio(file_path)
        print("\n========== 最终结果 ==========")
        print(result)
    else:
        print(f"⚠️ 未找到音频文件：{file_path}")

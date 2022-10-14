import io
import logging

import librosa
import soundfile
import torch
import torchaudio
from flask import Flask, request, send_file
from flask_cors import CORS

from sovits import infer_tool

app = Flask(__name__)

CORS(app)

logging.getLogger('numba').setLevel(logging.WARNING)


@app.route("/voiceChangeModel", methods=["POST"])
def voice_change_model():
    request_form = request.form
    wave_file = request.files.get("sample", None)
    # 变调信息
    f_pitch_change = float(request_form.get("fPitchChange", 0))
    # DAW所需的采样率
    raw_sample = int(float(request_form.get("sampleRate", 0)))
    speaker_id = int(float(request_form.get("sSpeakId", 0)))
    # http获得wav文件并转换
    input_wav_path = io.BytesIO(wave_file.read())
    # 读取VST发送过来的音频
    origin_audio, origin_audio_sr = torchaudio.load(input_wav_path)
    # 重采样到模型所需的采样率
    model_input_audio = librosa.resample(origin_audio[0].numpy(), origin_audio_sr, target_sample)
    # 模型推理
    out_audio, out_sr = infer_tool.infer(torch.from_numpy(model_input_audio).unsqueeze(0),
                                         target_sample,
                                         speaker_id,
                                         f_pitch_change,
                                         net_g_ms,
                                         hubert_soft,
                                         feature_input)
    # 模型输出音频重采样到DAW所需采样率
    tar_audio = librosa.resample(out_audio, target_sample, raw_sample)
    # 返回音频
    out_wav_path = io.BytesIO()
    soundfile.write(out_wav_path, tar_audio, raw_sample, format="wav")
    out_wav_path.seek(0)
    return send_file(out_wav_path, download_name="temp.wav", as_attachment=True)


if __name__ == '__main__':
    # 每个模型和config是唯一对应的
    model_name = "152_epochs.pth"
    config_name = "nyarumul.json"
    # 加载模型
    net_g_ms, hubert_soft, feature_input, hps_ms = infer_tool.load_model(f"pth/{model_name}", f"configs/{config_name}")
    target_sample = hps_ms.data.sampling_rate
    # 此处与vst插件对应，不建议更改
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)

import os

import numpy as np
import soundfile
import torch
import torchaudio
from pydub import AudioSegment

import config
import hubert_model
import utils
from models import SynthesizerTrn
from preprocess_wave import FeatureInput

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = config.model_name
config_name = config.config_name
hps_ms = utils.get_hparams_from_file(f"configs/{config_name}")


def load_model():
    n_g_ms = SynthesizerTrn(
        178,
        hps_ms.data.filter_length // 2 + 1,
        hps_ms.train.segment_size // hps_ms.data.hop_length,
        n_speakers=hps_ms.data.n_speakers,
        **hps_ms.model)
    _ = utils.load_checkpoint(f"pth/{model_name}", n_g_ms, None)
    _ = n_g_ms.eval().to(dev)
    return n_g_ms


# 加载sovits模型
net_g_ms = load_model()
# 获取config参数
target_sample = hps_ms.data.sampling_rate

hubert_soft = hubert_model.hubert_soft(f'pth/{config.hubert_name}')
feature_input = FeatureInput(hps_ms.data.sampling_rate, hps_ms.data.hop_length)


def resize2d_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res


def get_units(audio, sr):
    source = torchaudio.functional.resample(audio, sr, 16000)
    source = source.unsqueeze(0).to(dev)
    with torch.inference_mode():
        units = hubert_soft.units(source)
        return units


def transcribe(source_path, length, transform):
    feature_pit = feature_input.compute_f0(source_path)
    feature_pit = feature_pit * 2 ** (transform / 12)
    feature_pit = resize2d_f0(feature_pit, length)
    coarse_pit = feature_input.coarse_f0(feature_pit)
    return coarse_pit


def infer(source_path, speaker_id, tran):
    audio, sample_rate = torchaudio.load(source_path)
    sid = torch.LongTensor([int(speaker_id)]).to(dev)
    soft = get_units(audio, sample_rate).squeeze(0).cpu().numpy()
    pitch = transcribe(source_path, soft.shape[0], tran)
    pitch = torch.LongTensor(pitch).unsqueeze(0).to(dev)
    stn_tst = torch.FloatTensor(soft)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(dev)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(dev)
        audio = \
            net_g_ms.infer(x_tst, x_tst_lengths, pitch, sid=sid, noise_scale=.3, noise_scale_w=0.5,
                           length_scale=1)[0][
                0, 0].data.float().cpu().numpy()
    return audio, audio.shape[0]


# python删除文件的方法 os.remove(path)path指的是文件的绝对路径,如：
def del_file(path_data):
    for i in os.listdir(path_data):  # os.listdir(path_data)#返回一个列表，里面是当前目录下面的所有东西的相对路径
        os.remove(path_data + i)


def cut(c_time, file_path, vocal_name, out_dir):
    audio_segment = AudioSegment.from_file(file_path, format='wav')

    total = int(audio_segment.duration_seconds / c_time)  # 计算音频切片后的个数
    for i in range(total):
        # 将音频10s切片，并以顺序进行命名
        audio_segment[i * c_time * 1000:(i + 1) * c_time * 1000].export(f"{out_dir}/{vocal_name}-{i}.wav",
                                                                        format="wav")
    audio_segment[total * c_time * 1000:].export(f"{out_dir}/{vocal_name}-{total}.wav", format="wav")  # 缺少结尾的音频片段


def wav_resample(audio_path, tar_sample):
    raw_audio, raw_sample_rate = torchaudio.load(audio_path)
    tar_audio = torchaudio.transforms.Resample(orig_freq=raw_sample_rate, new_freq=tar_sample)(raw_audio)[0]
    soundfile.write(audio_path, tar_audio, tar_sample)
    return tar_audio, tar_sample


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])

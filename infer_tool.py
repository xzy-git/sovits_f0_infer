import os

import numpy as np
import soundfile
import torch
import torchaudio
from pydub import AudioSegment

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_units(path, hubert_soft):
    source, sr = torchaudio.load(path)
    source = torchaudio.functional.resample(source, sr, 16000)
    source = source.unsqueeze(0).to(dev)
    with torch.inference_mode():
        units = hubert_soft.units(source)
        return units


def transcribe(path, length, transform, feature_input):
    feature_pit = feature_input.compute_f0(path)
    feature_pit = feature_pit * 2 ** (transform / 12)
    feature_pit = resize2d_f0(feature_pit, length)
    coarse_pit = feature_input.coarse_f0(feature_pit)
    return coarse_pit


def infer(file_name, speaker_id, tran, target_sample, net_g_ms, hubert_soft, feature_input):
    source_path = "./wav_temp/input/" + file_name
    audio, sample_rate = torchaudio.load(source_path)
    input_size = audio.shape[-1]

    sid = torch.LongTensor([int(speaker_id)]).to(dev)
    soft = get_units(source_path, hubert_soft).squeeze(0).cpu().numpy()
    pitch = transcribe(source_path, soft.shape[0], tran, feature_input)
    pitch = torch.LongTensor(pitch).unsqueeze(0).to(dev)
    stn_tst = torch.FloatTensor(soft)
    with torch.no_grad():
        x_tst = stn_tst.unsqueeze(0).to(dev)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(dev)
        audio = \
            net_g_ms.infer(x_tst, x_tst_lengths, pitch, sid=sid, noise_scale=.3, noise_scale_w=0.5,
                           length_scale=1)[0][
                0, 0].data.float().cpu().numpy()
    soundfile.write("./wav_temp/output/" + file_name, audio,
                    int(audio.shape[0] / input_size * target_sample))


def resize2d_f0(x, target_len):
    source = np.array(x)
    source[source < 0.001] = np.nan
    target = np.interp(np.arange(0, len(source) * target_len, len(source)) / target_len, np.arange(0, len(source)),
                       source)
    res = np.nan_to_num(target)
    return res


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
    audio_22050 = torchaudio.transforms.Resample(orig_freq=raw_sample_rate, new_freq=tar_sample)(raw_audio)[0]
    soundfile.write(audio_path, audio_22050, tar_sample)


def fill_a_to_b(a, b):
    if len(a) < len(b):
        for _ in range(0, len(b) - len(a)):
            a.append(a[0])

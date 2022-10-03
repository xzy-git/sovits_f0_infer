import os
import shutil

import demjson
import soundfile
import torch
import torchaudio

import infer_tool
import utils
from wav_temp import merge

# 自行创建pth文件夹，放置hubert、sovits模型，创建raw、results文件夹
# 可填写音源文件列表，音源文件格式为wav，放置于raw文件夹下
clean_names = ["像我这样的人", "嘘月"]
# bgm、trans分别对应歌曲列表，若能找到相应文件、则自动合并伴奏，若找不到bgm，则输出干声（不使用bgm合成多首歌时，可只随意填写一个不存在的bgm名）
bgm_names = ["bgm1"]
# 合成多少歌曲时，若半音数量不足、自动补齐相同数量（按第一首歌的半音）
trans = [3]  # 加减半音数（可为正负）s
# 每首歌同时输出的speaker_id
id_list = [3]

# 每次合成长度，建议30s内，太高了爆显存(gtx1066一次30s以内）
cut_time = 20
model_name = "220_epochs"  # 模型名称（pth文件夹下）
config_name = "sovits_pre.json"  # 模型配置（config文件夹下）

# 以下内容无需修改
hps_ms = utils.get_hparams_from_file(f"configs/{config_name}")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取config参数
target_sample = hps_ms.data.sampling_rate

# 自动补齐
infer_tool.fill_a_to_b(bgm_names, clean_names)
infer_tool.fill_a_to_b(trans, clean_names)
# 遍历列表
for clean_name, bgm_name, tran in zip(clean_names, bgm_names, trans):
    infer_tool.wav_resample(f'./raw/{clean_name}.wav', target_sample)
    for speaker_id in id_list:
        speakers = demjson.decode_file(f"configs/{config_name}")["speakers"]
        out_audio_name = model_name + f"_{clean_name}_{speakers[speaker_id]}"
        # 清除缓存文件
        infer_tool.del_file("./wav_temp/input/")
        infer_tool.del_file("./wav_temp/output/")

        raw_audio_path = f"./raw/{clean_name}.wav"
        audio, sample_rate = torchaudio.load(raw_audio_path)
        audio_time = audio.shape[-1] / target_sample

        # 源音频切割方案
        if audio_time > 1.3 * int(cut_time):
            infer_tool.cut(int(cut_time), raw_audio_path, out_audio_name, "./wav_temp/input")
        else:
            shutil.copy(f"./raw/{clean_name}.wav", f"./wav_temp/input/{out_audio_name}-0.wav")

        count = 0
        file_list = os.listdir("./wav_temp/input")
        for file_name in file_list:
            source_path = "./wav_temp/input/" + file_name
            out_audio, out_sr = infer_tool.infer(source_path, speaker_id, tran)
            soundfile.write("./wav_temp/output/" + file_name, out_audio, target_sample)
            count += 1
            print("%s success: %.2f%%" % (file_name, 100 * count / len(file_list)))
        merge.run(out_audio_name, bgm_name, out_audio_name)

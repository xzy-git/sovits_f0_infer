import soundfile
import torch

import infer_tool
import utils

model_name = "220_epochs"  # 模型名称（pth文件夹下）
config_name = "sovits_pre.json"  # 模型配置（config文件夹下）

file_path = ""

# 以下内容无需修改
hps_ms = utils.get_hparams_from_file(f"configs/{config_name}")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 获取config参数
target_sample = hps_ms.data.sampling_rate

infer_tool.wav_resample(file_path, target_sample)

out_audio, out_sr = infer_tool.infer(source_path, speaker_id, tran)
soundfile.write("./wav_temp/output/" + file_name, out_audio,
                int(out_sr / input_size * target_sample))

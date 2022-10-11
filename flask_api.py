import os
from datetime import datetime

import soundfile
from flask import Flask, request
from flask import send_from_directory
from flask_cors import CORS

from sovits import infer_tool

app = Flask(__name__)

CORS(app)


def get_timestamp():
    timestamp = datetime.now()
    return timestamp.strftime("%Y%m%d_%H%M%S")


@app.route("/voiceChangeModel", methods=["POST"])
def voiceChangeModel():
    """
    :return:
    """

    request_form = request.form
    request_files = request.files
    wave_file = request_files.get("sample", None)
    f_pitch_change = int(float(request_form.get("fPitchChange", 0)))
    save_file_name = f"{http_temp_path}/{get_timestamp()}.wav"
    w_file = wave_file.stream.read()
    with open(save_file_name, 'wb') as fop:
        fop.write(w_file)
    out_audio, out_sr = infer_tool.infer(save_file_name, speaker_id, int(f_pitch_change), net_g_ms, hubert_soft,
                                         feature_input)
    soundfile.write(f"{http_temp_path}/http_out.wav", out_audio, target_sample)
    result_file = f"{http_temp_path}/http_out.wav"
    return send_from_directory(http_temp_path, os.path.basename(result_file), as_attachment=True)


if __name__ == '__main__':
    speaker_id = 0
    model_name = "152_epochs.pth"
    config_name = "nyarumul.json"
    http_temp_path = "./wav_temp/http"
    infer_tool.mkdir(["./wav_temp", http_temp_path])
    net_g_ms, hubert_soft, feature_input, hps_ms = infer_tool.load_model(f"pth/{model_name}", f"configs/{config_name}")
    target_sample = hps_ms.data.sampling_rate
    app.run(port=6842, host="0.0.0.0", debug=False, threaded=False)

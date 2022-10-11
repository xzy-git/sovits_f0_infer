# SoftVC VITS Singing Voice Conversion

## 模型简介

歌声音色转换模型，通过Soft-VC内容编码器提取源音频语音特征，并提取音频f0，将两者结合输入VITS替换原本的文本输入达到歌声转换的效果。

## 使用方式

1、main.py为一键合成长时间音频（数分钟以上），使用方法参考注释

2、gui开启127.0.0.1:7860网页，可在页面加载模型、转换语音（太长可能爆显存）

3、http_server可以转换http方式传输的音频，使用方法参考http_cli

4、vst插件实时转换（慢半句左右）

4.1 自行下载vst插件并搜索安装教程

https://github.com/zhaohui8969/VST_NetProcess-/releases

4.2 新建pth文件夹，放入此模型

https://huggingface.co/spaces/xiaolang/sovits_f0/resolve/main/152_epochs.pth

自行下载hubert-soft-0d54a1f4.pt改名为hubert.pt放置于pth文件夹下

https://github.com/bshall/hubert/releases/tag/v0.1

4.3 c盘根目录新建/temp/vst文件夹

4.4 安装requirements.txt

4.5 首先运行！！！flask_api.py，待出现运行网址127.0.0.1:6842后，再打开宿主软件

4.6 给某声道挂载vst插件，打开录音准备、监听，录音输入增益可以调高一些，即可使用
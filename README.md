# OpenVINO Japanese chatbot demo (youri-7b-chat)

## Description
This is an LLM Japanese chatbot demo program using Intel OpenVINO toolkit. The demo uses [`rinna/youri-7b-chat`](https://huggingface.co/rinna/youri-7b-chat) model developed by [Rinna Co.,Ltd](https://rinna.co.jp).
This program can use either one of 'CPU', 'GPU.0' (integrated GPU), and 'GPU.1' Intel discrete GPU for inferencing.

## Requirement
- At least main memory of 32GB. 64GB is recommended. 

## Install prerequisites
Recommend to use Python virtual env.<br>
You need to have Python installed.
```sh
# (optional) create a python venv and enable it.
python -m venv venv
(win) venv/Scripts/activate
(Lnx) source venv/bin/activate

pip install -r requirements0-uninstall.txt
pip install -r requirements1-install.txt
```

## Files

|file|description|
|---|---|
|`config.yml`|The system setting file. You can specify huggingface model cache directory and device for inferencing|
|`youri-7b-chat-openvino.py`|WebUI chatbot demo using `rinna/youri-7b-chat` model. This program uses OpenVINO as inference engine.<br>You need to run `model_download.py` to download and convert the model into OpenVINO IR before you run this demo.|
|`model_download.py`|This program downloads `rinna/youri-7b-chat` model from huggingface and convert it into FP16 OpenVINO IR model.<br>The converted model will be stored in `./youri-7b-chat/FP16/` directory. Also, the original model downloaded from hugging face will be stored in `./cache/huggingface/hub/` directory.|
|`benchmark_pyt.py`|Simple benchmark program using PyTorch.|
|`benchmark_ov.py`|Simple benchmark program using OpenVINO.|

## Demo screenshot
![example](./resources/example.png)

## Test environment
- System 1
	- Software
		- Windows 11 23H2
		- OpenVINO 2023.1.0
	- Hardware
		- CPU: Core i7-10700K
		- MEM: 64GB
		- GPU: Intel A380 (discrete GPU is optional)
- System 2
	- Software
		- Ubuntu 22.04.3
		- OpenVINO 2023.1.0
	- Hardware
		- CPU: Xeon Platinum 8358
		- MEM: 256GB

## Special thanks
Rinna Co.,Ltd - Original developer of rinna/youri-7b-chat model.

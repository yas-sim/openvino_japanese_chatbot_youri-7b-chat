# youri-7b-chat OpenVINO demo

## Description
This is a LLM chatbot demo program using Intel OpenVINO toolkit. The demo uses [`rinna/youri-7b-chat`](https://huggingface.co/rinna/youri-7b-chat) model developed by Microsoft.

|file|description|
|---|---|
|`config.yml`|The system setting file. You can specify huggingface model cache directory and device for inferencing|
|`youri-7b-chat-openvino.py`|WebUI chatbot demo using rinna/youri-7b-chat model. This program uses OpenVINO as inference engine.<br>You need to run `model_download.py` to download and convert the model into OpenVINO IR before you run this demo.|
|`model_download.py`|Download rinna/youri-7b-chat model from huggingface and convert it into FP16 OpenVINO IR model.<br>The converted model will be stored in `./youri-7b-chat/FP16/` directory. Also, the original model downloaded from hugging face will be stored in `./cache/huggingface/hub/` directory.|
|`benchmark_pyt.py`|Benchmark program using PyTorch.|
|`benchmark_ov.py`|Benchmark program using OpenVINO.|

import time, sys
from pathlib import Path
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer, AutoConfig

from transformers import AutoTokenizer

import openvino as ov

from config import read_config

niter = 10
if len(sys.argv)>1:
    niter = int(sys.argv[1])
    if niter <1:
        niter = 10
print(niter)

system_config = read_config()

model_id = { "youri-7b-chat" : { 
                "model_id":"rinna/youri-7b-chat",
                "tokenizer_kwargs": {"add_special_tokens":False},
            }}

model_name = "youri-7b-chat"

model_configuration = model_id[model_name]
pt_model_id = model_configuration["model_id"]
cache_dir = system_config['hf_cache_dir']
device = system_config['device']

core = ov.Core()

model_dir = system_config['ir_model_dir']
print(model_dir)
ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}

tok = AutoTokenizer.from_pretrained(model_configuration["model_id"], trust_remote_code=True, cache_dir=cache_dir)

model_class = OVModelForCausalLM
ov_model = model_class.from_pretrained(model_dir, device=device, ov_config=ov_config, config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True, cache_dir=cache_dir), trust_remote_code=True, cache_dir=cache_dir)

model_name       = model_configuration["model_id"]
tokenizer_kwargs = model_configuration.get("tokenizer_kwargs", {})

max_new_tokens = 200

prompt = """\
設定: 次の日本語を英語に翻訳してください。
ユーザー: 自然言語による指示に基づきタスクが解けるよう学習させることを Instruction tuning と呼びます。
システム: """
input_tokens = tok(prompt, return_tensors="pt", add_special_tokens=False)

ttl = 0
for i in range(niter):
    print(i)
    stime = time.time()
    answer = ov_model.generate(**input_tokens, max_new_tokens=max_new_tokens,
                temperature=0.5,
                pad_token_id=0, #tok.pad_token_id,
                bos_token_id=tok.bos_token_id,
                eos_token_id=tok.eos_token_id
    )
    etime = time.time()
    ttl += etime - stime
    print(tok.batch_decode(answer)[0])

print(ttl / niter)

# Core i7-10700K CPU int8
#設定: 次の日本語を英語に翻訳してください。
#ユーザー: 自然言語による指示に基づきタスクが解けるよう学習させることを Instruction tuning と呼びます。
#システム:  Learning to solve tasks based on natural language instructions is called instruction tuning.</s>
#7.4164869546890255

# Core i7-10700K CPU fp16
#設定: 次の日本語を英語に翻訳してください。
#ユーザー: 自然言語による指示に基づきタスクが解けるよう学習させることを Instruction tuning と呼びます。
#システム:  Learning to solve tasks based on natural language instructions is called instruction tuning.</s>
#12.865802192687989

# Core i7-10700K GPU.0 int8
#設定: 次の日本語を英語に翻訳してください。
#ユーザー: 自然言語による指示に基づきタスクが解けるよう学習させることを Instruction tuning と呼びます。
#システム:  Learning to solve tasks based on natural language instructions is called instruction tuning.</s>
#12.180223822593689

# Core i7-10700K GPU.0 fp16
#設定: 次の日本語を英語に翻訳してください。
#ユーザー: 自然言語による指示に基づきタスクが解けるよう学習させることを Instruction tuning と呼びます。
#システム:  Learning to solve tasks based on natural language instructions is called instruction tuning.</s>
#16.45764605998993

# Core i7-10700K GPU.1 A380 int8
#設定: 次の日本語を英語に翻訳してください。
#ユーザー: 自然言語による指示に基づきタスクが解けるよう学習させることを Instruction tuning と呼びます。
#システム:  Learning to solve tasks based on natural language instructions is called instruction tuning.</s>
#37.863298773765564

# Core i7-10700K GPU.1 A380 fp16
#設定: 次の日本語を英語に翻訳してください。
#ユーザー: 自然言語による指示に基づきタスクが解けるよう学習させることを Instruction tuning と呼びます。
#システム:  Learning to solve tasks based on natural language instructions is called instruction tuning.</s>
#80.55444238185882
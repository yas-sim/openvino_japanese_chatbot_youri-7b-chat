import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

from config import read_config
system_config = read_config()

cache_dir = system_config['hf_cache_dir']
tokenizer = AutoTokenizer.from_pretrained("rinna/youri-7b-chat", cache_dir = cache_dir)
model = AutoModelForCausalLM.from_pretrained("rinna/youri-7b-chat", cache_dir = cache_dir)

prompt = """\
設定: 次の日本語を英語に翻訳してください。
ユーザー: 自然言語による指示に基づきタスクが解けるよう学習させることを Instruction tuning と呼びます。
システム: """

token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

niter = 10
if len(sys.argv)>1:
    niter = int(sys.argv[1])
    if niter <1:
        niter = 10
print(niter)

ttl = 0
with torch.no_grad():
    for i in range(niter):
        print(i)
        stime = time.time()
        output_ids = model.generate(
            token_ids.to(model.device),
            max_new_tokens=200,
            do_sample=True,
            temperature=0.5,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        etime = time.time()
        ttl += etime - stime
        output = tokenizer.decode(output_ids.tolist()[0])
        print(output)

print(ttl/niter)
# Core i7-10700K CPU
#設定: 次の日本語を英語に翻訳してください。
#ユーザー: 自然言語による指示に基づきタスクが解けるよう学習させることを Instruction tuning と呼びます。
#システム:  Teaching a student to solve tasks based on natural language instructions is called instructional tuning.</s>
#20.129881048202513

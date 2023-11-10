import gc
from pathlib import Path

from optimum.intel import OVQuantizer
from transformers import AutoModelForCausalLM
from optimum.intel.openvino import OVModelForCausalLM

from config import read_config
system_config = read_config()

model_name = 'youri-7b-chat'
model_id = f'rinna/{model_name}'
cache_dir = system_config['hf_cache_dir']

compressed_model_dir = Path(model_name) / "INT8_compressed_weights"
fp16_model_dir = Path(model_name) / "FP16"


# INT8
#pt_model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=cache_dir)
#quantizer = OVQuantizer.from_pretrained(pt_model)
#quantizer.quantize(save_directory=compressed_model_dir, weights_only=True)
#del quantizer
#del pt_model
#gc.collect()

# FP16
ov_model = OVModelForCausalLM.from_pretrained(model_id, export=True, compile=False, cache_dir=cache_dir)
ov_model.half()
ov_model.save_pretrained(fp16_model_dir)
del ov_model
gc.collect()

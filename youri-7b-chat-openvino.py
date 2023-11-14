from pathlib import Path
from threading import Event, Thread
from uuid import uuid4
from typing import Tuple, List

from config import read_config

import openvino as ov
from optimum.intel.openvino import OVModelForCausalLM

import torch
from transformers import AutoTokenizer, AutoConfig, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

import gradio as gr

# Youri-7b-chat special tokens
#id	token	
#0	'<unk>'	
#1	'<s>'	bos_token_id
#2	'</s>'	eos_token_id
#None		pad_token_id

DEFAULT_SYSTEM_PROMPT_JP = """\
あなたは親切で、礼儀正しく、誠実なアシスタントです。 常に安全を保ちながら、できるだけ役立つように答えてください。 回答には、有害、非倫理的、人種差別的、性差別的、有毒、危険、または違法なコンテンツを含めてはいけません。 回答は社会的に偏見がなく、本質的に前向きなものであることを確認してください。
質問が意味をなさない場合、または事実に一貫性がない場合は、正しくないことに答えるのではなく、その理由を説明してください。 質問の答えがわからない場合は、誤った情報を共有しないでください。\
"""

system_config = read_config()

def llama_partial_text_processor(partial_text, new_text):
    new_text = new_text.replace("[INST]", "").replace("[/INST]", "")
    partial_text += new_text
    return partial_text

model_id = { "youri-7b-chat" : { 
                "model_id":"rinna/youri-7b-chat",
                "start_message": f"設定: {DEFAULT_SYSTEM_PROMPT_JP}\n",
                "history_template": "ユーザー: {user}\nシステム: {assistant}\n",
                "current_message_template": "ユーザー: {user}\nシステム: {assistant}",
                "tokenizer_kwargs": {"add_special_tokens":False},
                "partial_text_processor": llama_partial_text_processor
                }}

model_name = "youri-7b-chat"

model_configuration = model_id[model_name]
pt_model_id = model_configuration["model_id"]
cache_dir = system_config['hf_cache_dir']

device = system_config['device']
ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
model_dir = system_config['ir_model_dir']

core = ov.Core()
tok = AutoTokenizer.from_pretrained(model_configuration["model_id"], trust_remote_code=True, cache_dir=cache_dir)
model_class = OVModelForCausalLM
ov_model = model_class.from_pretrained(model_dir, device=device, ov_config=ov_config, config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True, cache_dir=cache_dir), trust_remote_code=True, cache_dir=cache_dir)

model_name               = model_configuration["model_id"]
history_template         = model_configuration["history_template"]
current_message_template = model_configuration["current_message_template"]
start_message            = model_configuration["start_message"]
stop_tokens              = model_configuration.get("stop_tokens")
tokenizer_kwargs         = model_configuration.get("tokenizer_kwargs", {})

#start_message = ''

max_new_tokens = 256

class StopOnTokens(StoppingCriteria):
    def __init__(self, token_ids):
        self.token_ids = token_ids
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        for stop_id in self.token_ids:
            if input_ids[0][-1] == stop_id:
                return True
        return False

if stop_tokens is not None:
    if isinstance(stop_tokens[0], str):
        stop_tokens = tok.convert_tokens_to_ids(stop_tokens)
        
    stop_tokens = [StopOnTokens(stop_tokens)]

def default_partial_text_processor(partial_text:str, new_text:str):
    """
    helper for updating partially generated answer, used by de
    
    Params:
      partial_text: text buffer for storing previosly generated text
      new_text: text update for the current step
    Returns:
      updated text string
    
    """
    partial_text += new_text
    return partial_text

text_processor = model_configuration.get("partial_text_processor", default_partial_text_processor)
    
def convert_history_to_text(history:List[Tuple[str, str]]):
    """
    function for conversion history stored as list pairs of user and assistant messages to string according to model expected conversation template
    Params:
      history: dialogue history
    Returns:
      history in text format
    """
    text = start_message + "".join(
        [
            "".join(
                [
                    history_template.format(user=item[0], assistant=item[1])
                ]
            )
            for item in history[:-1]
        ]
    )
    text += "".join(
        [
            "".join(
                [
                    current_message_template.format(user=history[-1][0], assistant=history[-1][1])
                ]
            )
        ]
    )
    print(text)
    return text



def user(message, history):
    """
    callback function for updating user messages in interface on submit button click
    
    Params:
      message: current message
      history: conversation history
    Returns:
      None
    """
    # Append the user's message to the conversation history
    return "", history + [[message, ""]]


def bot(history, temperature, top_p, top_k, repetition_penalty, conversation_id):
    """
    callback function for running chatbot on submit button click
    
    Params:
      history: conversation history
      temperature:  parameter for control the level of creativity in AI-generated text. 
                    By adjusting the `temperature`, you can influence the AI model's probability distribution, making the text more focused or diverse.
      top_p: parameter for control the range of tokens considered by the AI model based on their cumulative probability.
      top_k: parameter for control the range of tokens considered by the AI model based on their cumulative probability, selecting number of tokens with highest probability.
      repetition_penalty: parameter for penalizing tokens based on how frequently they occur in the text.
      conversation_id: unique conversation identifier.
    
    """

    # Construct the input message string for the model by concatenating the current system message and conversation history
    messages = convert_history_to_text(history)

    # Tokenize the messages string
    tokens = tok(messages, return_tensors="pt", **tokenizer_kwargs)
    input_ids = tokens.input_ids
    atten_mask = tokens.attention_mask
    if input_ids.shape[1] > 2000:
        history = [history[-1]]
        messages = convert_history_to_text(history)
        input_ids = tok(messages, return_tensors="pt", **tokenizer_kwargs).input_ids
    streamer = TextIteratorStreamer(tok, timeout=30.0, skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=temperature > 0.0,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        streamer=streamer,
        attention_mask=atten_mask,
        pad_token_id=0, #tok.pad_token_id,
        bos_token_id=tok.bos_token_id,
        eos_token_id=tok.eos_token_id
    )
    if stop_tokens is not None:
        generate_kwargs["stopping_criteria"] = StoppingCriteriaList(stop_tokens)

    stream_complete = Event()

    def generate_and_signal_complete():
        """
        genration function for single thread
        """
        global start_time
        ov_model.generate(**generate_kwargs)
        stream_complete.set()

    t1 = Thread(target=generate_and_signal_complete)
    t1.start()

    # Initialize an empty string to store the generated text
    partial_text = ""
    for new_text in streamer:
        partial_text = text_processor(partial_text, new_text)
        history[-1][1] = partial_text
        yield history


def get_uuid():
    """
    universal unique identifier for thread
    """
    return str(uuid4())


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=".disclaimer {font-variant-caps: all-small-caps;}",
) as demo:
    conversation_id = gr.State(get_uuid)
    gr.Markdown(
        f"""<h1><center>OpenVINO {model_name} Chatbot</center></h1>"""
    )
    chatbot = gr.Chatbot(height=500)
    with gr.Row():
        with gr.Column():
            msg = gr.Textbox(
                label="Chat Message Box",
                placeholder="Chat Message Box",
                show_label=False,
                container=False
            )
        with gr.Column():
            with gr.Row():
                submit = gr.Button("Submit")
                stop = gr.Button("Stop")
                clear = gr.Button("Clear")
    with gr.Row():
        with gr.Accordion("Advanced Options:", open=False):
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        temperature = gr.Slider(
                            label="Temperature",
                            value=0.5,
                            minimum=0.0,
                            maximum=1.0,
                            step=0.1,
                            interactive=True,
                            info="Higher values produce more diverse outputs",
                        )
                with gr.Column():
                    with gr.Row():
                        top_p = gr.Slider(
                            label="Top-p (nucleus sampling)",
                            value=1.0,
                            minimum=0.0,
                            maximum=1,
                            step=0.01,
                            interactive=True,
                            info=(
                                "Sample from the smallest possible set of tokens whose cumulative probability "
                                "exceeds top_p. Set to 1 to disable and sample from all tokens."
                            ),
                        )
                with gr.Column():
                    with gr.Row():
                        top_k = gr.Slider(
                            label="Top-k",
                            value=50,
                            minimum=0.0,
                            maximum=200,
                            step=1,
                            interactive=True,
                            info="Sample from a shortlist of top-k tokens — 0 to disable and sample from all tokens.",
                        )
                with gr.Column():
                    with gr.Row():
                        repetition_penalty = gr.Slider(
                            label="Repetition Penalty",
                            value=1.1,
                            minimum=1.0,
                            maximum=2.0,
                            step=0.1,
                            interactive=True,
                            info="Penalize repetition — 1.0 to disable.",
                        )
    gr.Examples([
        ["日本で一番高い山は？"],
        ["秋田県の名物料理は何ですか？"],
        ["こんにちは！ お元気ですか？"],
        ["OpenVINOって何ですか?"],
        ["あなたは誰ですか?"],
        ["Python プログラミング言語とは何なのか簡単に説明してもらえますか?"],
        ["シンデレラのあらすじを簡潔に説明して下さい。"],
        ["コードを書くときに避けるべきよくある間違いは何ですか?"],
        ["「人工知能と OpenVINO の利点」について 100語程度でブログ記事を書いてください"]
    ], 
        inputs=msg, 
        label="Click on any example and press the 'Submit' button"
    )

    submit_event = msg.submit(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            conversation_id,
        ],
        outputs=chatbot,
        queue=True,
    )
    submit_click_event = submit.click(
        fn=user,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False,
    ).then(
        fn=bot,
        inputs=[
            chatbot,
            temperature,
            top_p,
            top_k,
            repetition_penalty,
            conversation_id,
        ],
        outputs=chatbot,
        queue=True,
    )
    stop.click(
        fn=None,
        inputs=None,
        outputs=None,
        cancels=[submit_event, submit_click_event],
        queue=False,
    )
    clear.click(lambda: None, None, chatbot, queue=False)

demo.queue(max_size=2)
# if you are launching remotely, specify server_name and server_port
#  demo.launch(server_name='your server name', server_port='server port in int')
# if you have any issue to launch on your platform, you can pass share=True to launch method:
# demo.launch(share=True)
# it creates a publicly shareable link for the interface. Read more in the docs: https://gradio.app/docs/
demo.launch(inbrowser=True)

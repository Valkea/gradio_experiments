import os
import time
import json
import random
import requests

import gradio as gr
from huggingface_hub import InferenceClient
from huggingface_hub.utils._errors import HfHubHTTPError


# --- GET KEYS ---

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

endpoint_url = os.getenv("API_URL")
hf_token = os.getenv("HF_TOKEN")

# --- DEFINE VARIABLES ---

# -- streaming client
client = InferenceClient(endpoint_url, token=hf_token)

# -- generation parameter
gen_kwargs = dict(
    max_new_tokens=256,
    top_k=30,
    top_p=0.9,
    temperature=0.2,
    repetition_penalty=1.02,
    stop_sequences=["###", "</s>"],
)

# --- DEFINE FUNCTIONS ---


def create_prompt_formats(user_input, context=None):
    INTRO_BLURB = "Below is an instruction that describes a task. Write a short response that appropriately completes the request. But avoid giving too much details."
    INSTRUCTION_KEY = "### Instruction:"
    INPUT_KEY = "Input:"
    RESPONSE_KEY = "### Response:"

    blurb = f"{INTRO_BLURB}"
    instruction = f"{INSTRUCTION_KEY}\n{user_input}"
    input_context = f"{INPUT_KEY}\n{context}" if context else None
    response = f"{RESPONSE_KEY}\n"

    parts = [part for part in [blurb, instruction, input_context, response] if part]
    formatted_prompt = "\n\n".join(parts)
    return formatted_prompt


def predict(message, history):
    cnt_retry = 0
    max_retry = 60 # 60*15= 15 min
    wait_time = 15
    error_msg = "The Inference server is unreachable. Please try again later."
    wait_msg = "The Inference server was off. We will spin it for you! \nPlease be patient (this model can take up to 10 minutes to load)\n"

    while True:
        try:
            prompt = create_prompt_formats(message, None)
            stream = client.text_generation(
                prompt, stream=True, details=True, **gen_kwargs
            )
            answer = ""

            # -- yield each generated token
            for r in stream:
                # -- skip special tokens
                if r.token.special:
                    continue

                # -- stop if we encounter a stop sequence
                if r.token.text in gen_kwargs["stop_sequences"]:
                    break

                # -- yield the generated token
                print(r.token.text, end="")
                answer += r.token.text
                yield answer
            break

        except Exception:
            cnt_retry += 1
            wait_msg += "."
            yield wait_msg
            time.sleep(wait_time)

        if cnt_retry >= max_retry:
            yield error_msg
            raise gr.error(error_msg)
            break


# --- DEFINE INTERFACE ---

demo = gr.ChatInterface(predict)

# demo.launch()
demo.queue(concurrency_count=75).launch(debug=True)

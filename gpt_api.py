from openai import OpenAI
from retry import retry

from joblib import Memory
from dotenv import load_dotenv

from helper import timeout

import os

# Initialize Cache
memory = Memory("./cache", verbose=0)

# Load ENV variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPEN_AI_KEY"))

# ChatGPT API Calls
@memory.cache
def call_chat_gpt_api(*args, **kwargs):
    return call_chat_gpt_api_no_cache(*args, **kwargs) 


@retry(tries=3, delay=15)
@timeout(30)
def call_chat_gpt_api_no_cache(prompt, model="gpt-3.5-turbo-1106", temp=0.8, n=1, **kwargs):
    if model in ['Llama-2-7b-chat-hf']:
        client.base_url = "http://localhost:8000/v1"

    return client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        n=n,
        **kwargs
    )
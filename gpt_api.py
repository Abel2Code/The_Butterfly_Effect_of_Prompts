import openai

from joblib import Memory
from dotenv import load_dotenv

import os

# Initialize Cache
memory = Memory("./cache", verbose=0)

# Load ENV variables
load_dotenv()
openai.api_key = os.getenv("OPEN_AI_KEY")

# ChatGPT API Calls
@memory.cache
def call_chat_gpt_api(*args, **kwargs):
    return call_chat_gpt_api_no_cache(*args, **kwargs) 
 
def call_chat_gpt_api_no_cache(prompt, model="gpt-3.5-turbo", temp=0.8, n=1, **kwargs):
    return openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temp,
        n=n,
        **kwargs
    )
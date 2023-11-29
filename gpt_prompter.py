from gpt_api import call_chat_gpt_api

def call_prompt(self, prompt, temp=0, n=1, **kwargs):
    res = call_chat_gpt_api(prompt, temp=0, n=1, **kwargs)
    return [c.message.content for c in res.choices]
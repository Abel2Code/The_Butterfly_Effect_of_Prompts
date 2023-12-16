import sys
sys.path.append("..")

from .gpt_api import call_chat_gpt_api
from helper import Factory, update_cache, load_cache

import time

class OpenAI_Model:
    def __init__(self, name="ChatGPT", model="gpt-3.5-turbo-1106", temp=0, mode="default", **kwargs):
        self.temp = temp
        self.name = name
        self.kwargs = kwargs
        self.model = model
        self.cache_file_name = f'custom_cache/{name}-{mode}.json'
        self.cache = load_cache(self.cache_file_name)
        
    def chat(self, prompt, **kwargs):
        if prompt in self.cache:
            return self.cache[prompt]
        else:    
            res = call_chat_gpt_api(prompt, temp=self.temp, n=1, model=self.model, **kwargs, **self.kwargs)
            content = res.choices[0].message.content
            update_cache(prompt, content, self.cache_file_name, self.cache)
        return content
    
class RepeatPromptModel:
    def __init__(self, *args, **kwargs):
        self.name = "RepeatPromptModel"
        
    def chat(self, prompt, **kwargs):
        return prompt

model_factory = Factory()
              
model_factory_dict = {
    "ChatGPT": OpenAI_Model(),
    "ChatGPT_JSON": OpenAI_Model(mode="json", response_format={ "type": "json_object" }),
    "LLaMA-7B": OpenAI_Model(name="LLaMA-7B", model="Llama-2-7b-chat-hf"),
    "Prompt": RepeatPromptModel()
    # "ChatGPT_Crazy": ChatGPT_Model(temp=2)
}

for k, v in model_factory_dict.items():
    model_factory.register_builder(k, v)

import sys
sys.path.append("..")

from .gpt_api import call_chat_gpt_api
# from .fireworks_api import call_fireworks_api
from helper import Factory, update_cache, load_cache

import time

class API_Model:
    def __init__(self, name="ChatGPT", model="gpt-3.5-turbo-1106", temp=0, mode="default", **kwargs):
        self.temp = temp
        self.name = name
        self.kwargs = kwargs
        self.model = model
        self.mode = mode
        self.cache = None
        self.is_cache_initialized = False

        if model in ["gpt-3.5-turbo-1106", "Llama-2-7b-chat-hf", "accounts/fireworks/models/llama-v2-70b-chat"]:
            self.api_func = call_chat_gpt_api
        else:
            raise Exception("Unsupported model used")

    def initialize_cache(self, task_name):
        self.is_cache_initialized = True
        self.cache_file_name = f'custom_cache/{self.name}/{task_name}.json'
        self.cache = load_cache(self.cache_file_name)
        
    def chat(self, prompt, key, **kwargs):
        assert self.is_cache_initialized

        if key not in self.cache:
            self.cache[key] = {}
        
        if prompt in self.cache[key]:
            return self.cache[key][prompt]
        else:
            res = self.api_func(prompt, temp=self.temp, n=1, model=self.model, **kwargs, **self.kwargs)
            content = res.choices[0].message.content
            self.save_to_cache(prompt, content, key)
        return content

    def save_to_cache(self, prompt, content, key):
        assert self.is_cache_initialized

        if key not in self.cache:
            self.cache[key] = {}
        
        if prompt in self.cache[key]:
            assert str(self.cache[key][prompt]) == str(content), (prompt, self.cache[key][prompt], content)
        else:
            self.cache[key][prompt] = content

    def write_cache_to_file(self):
        update_cache(self.cache_file_name, self.cache, self.name)
        
class RepeatPromptModel:
    def __init__(self, *args, **kwargs):
        self.name = "RepeatPromptModel"

    def initialize_cache(self, task_name):
        pass
        
    def chat(self, prompt, key, **kwargs):
        return prompt

    def save_to_cache(self, *args):
        pass

    def write_cache_to_file(self, *args):
        pass

model_factory = Factory()
              
model_factory_dict = {
    "ChatGPT": API_Model(),
    "ChatGPT_JSON": API_Model(mode="json", response_format={ "type": "json_object" }),
    "LLaMA-7B": API_Model(name="LLaMA-7B", model="Llama-2-7b-chat-hf"),
    "LLaMA-70B": API_Model(name="LLaMA-70B", model="accounts/fireworks/models/llama-v2-70b-chat"),
    "Prompt": RepeatPromptModel()
    # "ChatGPT_Crazy": ChatGPT_Model(temp=2)
}

for k, v in model_factory_dict.items():
    model_factory.register_builder(k, v)

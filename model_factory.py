from gpt_api import call_chat_gpt_api
from helper import Factory, update_cache, load_cache

import time

class ChatGPT_Model:
    def __init__(self, temp=0, mode="default", **kwargs):
        self.temp = temp
        self.kwargs = kwargs
        self.cache_file_name = f'custom_cache/ChatGPT-{mode}.json'
        self.cache = load_cache(self.cache_file_name)
        
    def chat(self, prompt, **kwargs):
        if prompt in self.cache:
            return self.cache[prompt]
        else:            
            res = call_chat_gpt_api(prompt, temp=self.temp, n=1, **kwargs, **self.kwargs)
            content = res.choices[0].message.content
            update_cache(prompt, content, self.cache_file_name, self.cache)
        return content

model_factory = Factory()
              
model_factory_dict = {
    "ChatGPT": ChatGPT_Model(),
    "ChatGPT_JSON": ChatGPT_Model(mode="json", response_format={ "type": "json_object" })
    # "ChatGPT_Crazy": ChatGPT_Model(temp=2)
}

for k, v in model_factory_dict.items():
    model_factory.register_builder(k, v)
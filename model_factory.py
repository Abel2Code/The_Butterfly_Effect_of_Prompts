from gpt_api import call_chat_gpt_api
from factory import Factory

class ChatGPT_Model:
    def __init__(self, temp=0, **kwargs):
        self.temp = temp
        self.kwargs = kwargs
        
    def chat(self, prompt, **kwargs):
        res = call_chat_gpt_api(prompt, temp=self.temp, n=1, **kwargs)
        return res.choices[0].message.content

model_factory = Factory()
              
model_factory_dict = {
    "ChatGPT": ChatGPT_Model(),
    "ChatGPT_JSON": ChatGPT_Model(response_format={ "type": "json_object" }),
    "ChatGPT_Crazy": ChatGPT_Model(temp=2)
}

for k, v in model_factory_dict.items():
    model_factory.register_builder(k, v)
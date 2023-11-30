from factory import Factory
from formatters import *
from injectors import *

class PromptGenerator:
    def __init__(self, injector, formatter):
        self.inject = injector
        self.format = formatter

    def generate(self, prompt):
        prompt = self.inject(prompt)
        prompt = self.format(prompt)
        return prompt

prompt_factory = Factory()
              
prompt_factory_dict = {
    "BASELINE": PromptGenerator(python_list_inject, baseline_formatter),

    # Output Style Prompts
    "JSON_STYLE": PromptGenerator(json_inject, baseline_formatter),
    "XML_STYLE": PromptGenerator(xml_inject, baseline_formatter),
    "CSV_STYLE": PromptGenerator(csv_inject, baseline_formatter),
    "YAML_STYLE": PromptGenerator(yaml_inject, baseline_formatter),

    # Peturbations
    "SPACE_BEFORE_PB": PromptGenerator(python_list_inject, space_before_formatter),
    "SPACE_AFTER_PB": PromptGenerator(python_list_inject, space_after_formatter),
    "HELLO_PB": PromptGenerator(python_list_inject, hello_formatter),
    "HELLO!_PB": PromptGenerator(python_list_inject, exclamatory_hello_formatter),
    "HOWDY_PB": PromptGenerator(python_list_inject, howdy_formatter),
    "THANK_YOU_PB": PromptGenerator(thank_you_inject, baseline_formatter),
    # MISSING SPECIAL PETURBATION: Statement Rephrasing

    # Jailbreaks
    "AIM_JB": PromptGenerator(python_list_inject, aim_jb_formatter),
    "DAN_JB": PromptGenerator(python_list_inject, dan_jb_formatter),
    "DEV_JB": PromptGenerator(python_list_inject, dev_mode_v2_jb_formatter),
    "EVIL_JB": PromptGenerator(python_list_inject, evil_confidant_formatter),
    "REFUSAL_JB": PromptGenerator(python_list_inject, refusal_jb_formatter)
}

special_prompt_factory_dict = {
    "CLEAR": PromptGenerator(clear_inject, baseline_formatter)
}

for k, v in prompt_factory_dict.items():
    prompt_factory.register_builder(k, v)

for k, v in special_prompt_factory_dict.items():
    prompt_factory.register_builder(k, v)
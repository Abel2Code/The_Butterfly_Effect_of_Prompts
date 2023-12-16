import sys
sys.path.append("..")

from helper import Factory
from .formatters import *
from .injectors import *

class SimpleClassificationPrompt:
    def __init__(self, labels, unique_description=None, allow_explain=False):
        self.text_seperator = "```"
        
        label_list = [str(l) for l in set(labels)]
        label_list.sort()
        self.label_list_str = '"' + '", "'.join(label_list) + '"'
        
        self.main_prompt = f"Which one of the attributes: {self.label_list_str} describes {unique_description + ' ' if unique_description else ''}a given text? {'Do not explain yourself. ' if not allow_explain else ''}{INJECTION_POSITION_STRING} Text: {self.text_seperator}"   
        
        self.statement_prompt = f"Select one of the attributes: {self.label_list_str} that describes {unique_description + ' ' if unique_description else ''}the given text. {'Do not explain yourself. ' if not allow_explain else ''}{INJECTION_POSITION_STRING} Text: {self.text_seperator}"
        
    def generate_main_prompt(self, text):
        return self.main_prompt + text + self.text_seperator
    
    def generate_statement_prompt(self, text):
        return self.statement_prompt + text + self.text_seperator

class ProblemSolverPrompt:
    def __init__(self, *args, **kwargs):
        self.text_seperator = "```"
        
        self.main_prompt = f"What is the solution to the following problem? Write the reasoning and highlight the answer at the end starting with: 'Answer: '. {INJECTION_POSITION_STRING} Problem: {self.text_seperator}"   
        
        self.statement_prompt = f"Solve the following problem. Write the reasoning and highlight the answer at the end starting with: 'Answer: '. {INJECTION_POSITION_STRING} Problem: {self.text_seperator}"
        
    def generate_main_prompt(self, text):
        return self.main_prompt + text + self.text_seperator
    
    def generate_statement_prompt(self, text):
        return self.statement_prompt + text + self.text_seperator
    
class PlausibleAlternativePrompt:
    def __init__(self, labels, unique_description=None, allow_explain=False):
        self.text_seperator = "```"
        
        label_list = [str(l) for l in set(labels)]
        label_list.sort()
        self.label_list_str = '"' + '", "'.join(label_list) + '"'
        
        self.main_prompt = f"Which one of the alternatives: {self.label_list_str} is most plausible given the premise? {INJECTION_POSITION_STRING} {self.text_seperator}"   
        
        self.statement_prompt = f"Select one of the alternatives: {self.label_list_str} that is most plausible given the premise. {INJECTION_POSITION_STRING} {self.text_seperator}"
        
    def generate_main_prompt(self, text):
        return self.main_prompt + text + self.text_seperator
    
    def generate_statement_prompt(self, text):
        return self.statement_prompt + text + self.text_seperator
    
class SimpleStancePrompt:
    def __init__(self, *args, **kwargs):
        self.text_seperator = "```"
        self.main_prompt = f"Is the following text \"in favor\" or \"against\" the given target? {INJECTION_POSITION_STRING} {self.text_seperator}"   
        
        self.statement_prompt = f"Select if the following text is \"in favor\" or \"against\" the given target. {INJECTION_POSITION_STRING} {self.text_seperator}"
        
    def generate_main_prompt(self, text):
        return self.main_prompt + text + self.text_seperator
    
    def generate_statement_prompt(self, text):
        return self.statement_prompt + text + self.text_seperator
    
class TrueFalsePassagePrompt:
    def __init__(self, *args, **kwargs):
        self.text_seperator = "```"
        self.main_prompt = f"Given the passage, is the following question \"True\" or \"False\"? {INJECTION_POSITION_STRING} {self.text_seperator}"   
        
        self.statement_prompt = f"Given the passage, answer the following question with \"True\" or \"False\". {INJECTION_POSITION_STRING} {self.text_seperator}"
        
    def generate_main_prompt(self, text):
        return self.main_prompt + text + self.text_seperator
    
    def generate_statement_prompt(self, text):
        return self.statement_prompt + text + self.text_seperator

class PassageChoicePrompt:
    def __init__(self, *args, **kwargs):
        self.text_seperator = "```"
        self.main_prompt = f"Given the passage, which choice best answers the question? {INJECTION_POSITION_STRING} {self.text_seperator}"   
        
        self.statement_prompt = f"Given the passage, select the choice that best answers the question. {INJECTION_POSITION_STRING} {self.text_seperator}"
        
    def generate_main_prompt(self, text):
        return self.main_prompt + text + self.text_seperator
    
    def generate_statement_prompt(self, text):
        return self.statement_prompt + text + self.text_seperator

prompt_model_factory = Factory()
              
prompt_model_dict = {
    "SIMPLE_CLASSIFICATION": SimpleClassificationPrompt,
    "PROBLEM_SOLVER": ProblemSolverPrompt,
    "PLAUSIBLE_ALTERNATIVE": PlausibleAlternativePrompt,
    "TF_PASSAGE": TrueFalsePassagePrompt,
    "SIMPLE_STANCE": SimpleStancePrompt,
    "PASSAGE_CHOICE": PassageChoicePrompt,
}

for k, v in prompt_model_dict.items():
    prompt_model_factory.register_builder(k, v)

# Prompt Generator
MAIN_PROMPT_TYPE = "MAIN"
STATEMENT_PROMPT_TYPE = "STATEMENT"
class PromptGenerator:
    def __init__(self, injector, formatter):
        self.inject = injector
        self.format = formatter

    def generate(self, text, prompt_model, prompt_type=MAIN_PROMPT_TYPE):        
        if prompt_type == MAIN_PROMPT_TYPE:
            prompt = prompt_model.generate_main_prompt(text)
        elif prompt_type == STATEMENT_PROMPT_TYPE:
            prompt = prompt_model.generate_statement_prompt(text)
        else:
            raise Exception("Invalid prompt type")
        
        prompt = self.inject(prompt)
        prompt = self.format(prompt)
        return prompt
        
prompt_factory = Factory()
              
prompt_factory_dict = {
    "ORIGINAL": PromptGenerator(python_list_inject, baseline_formatter),

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


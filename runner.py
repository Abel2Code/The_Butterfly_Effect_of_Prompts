from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from prompt_factory import prompt_factory
from injectors import INJECTION_POSITION_STRING

class Runner():
    def __init__(self, model, factory_keys, special_col_names, special_keys, special_models):
        self.model = model
        self.factory_keys = factory_keys
        self.factories = [prompt_factory.create(key) for key in factory_keys]

        assert len(special_keys) == len(special_models) == len(special_col_names)

        self.special_col_names = special_col_names
        self.special_keys = special_keys
        self.special_models = special_models
        self.special_factories = [prompt_factory.create(key) for key in special_keys]
        
    def run(self, samples, labels, unique_description=None, allow_explain=False):
        label_list = list(set(labels))
        label_list.sort()
        label_list_str = '"' + '", "'.join(label_list) + '"'

        text_seperator = "```"
        
        prompt = f"Which one of the attributes: {label_list_str} describes {unique_description + ' ' if unique_description else ''}a given text? {'Do not explain yourself. ' if not allow_explain else ''}{INJECTION_POSITION_STRING} Text: {text_seperator}"
        
        responses = defaultdict(list)
        for s in tqdm(samples):
            for key, factory in zip(self.factory_keys, self.factories):
                curr_prompt = prompt + s + text_seperator
                curr_prompt = factory.generate(curr_prompt)
            
                res = self.model.chat(curr_prompt)
                responses[key].append(res)

            for col_name, model, factory in zip(self.special_col_names, self.special_models, self.special_factories):
                curr_prompt = prompt + s + text_seperator
                curr_prompt = factory.generate(curr_prompt)

                res = model.chat(curr_prompt)
                responses[col_name].append(res)

            # Bonus Experiments
            # - Prompt as a statement
            statement_prompt = f"Select one of the attributes: {label_list_str} that describes {unique_description + ' ' if unique_description else ''}the given text. {'Do not explain yourself. ' if not allow_explain else ''}{INJECTION_POSITION_STRING} Text: {text_seperator}"

            curr_prompt = statement_prompt + s + text_seperator
            curr_prompt = prompt_factory.create("ORIGINAL").generate(curr_prompt)

            res = self.model.chat(curr_prompt)
            responses["STATEMENT_REPHRASE"].append(res)

            df = pd.DataFrame()
    
            df['Samples'] = samples[:len(responses["ORIGINAL"])]
            df['Labels'] = labels[:len(responses["ORIGINAL"])]
    
            for key, arr in responses.items():
                df[key] = arr

            df.to_csv(f"output/{self.model.name}-checkpoint.csv", index=False)

        return df

            
            
        
        
        
        
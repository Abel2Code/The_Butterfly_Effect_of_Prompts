from collections import defaultdict

import pandas as pd
from tqdm import tqdm

from prompt_tools.prompt_factory import prompt_factory, prompt_model_factory, STATEMENT_PROMPT_TYPE
from prompt_tools.injectors import INJECTION_POSITION_STRING

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
        
    def run(self, samples, labels, task, unique_description=None, allow_explain=False):
        prompt_model = prompt_model_factory.create(task)(labels, unique_description, allow_explain)
        
        responses = defaultdict(list)
        for s in tqdm(samples):
            for key, factory in zip(self.factory_keys, self.factories):
                curr_prompt = factory.generate(s, prompt_model)
            
                res = self.model.chat(curr_prompt)
                responses[key].append(res)

            for col_name, model, factory in zip(self.special_col_names, self.special_models, self.special_factories):
                curr_prompt = factory.generate(s, prompt_model)

                res = model.chat(curr_prompt)
                responses[col_name].append(res)

            # Bonus Experiments
            # - Prompt as a statement
            statement_factory = prompt_factory.create("ORIGINAL")
            curr_prompt = statement_factory.generate(s, prompt_model, prompt_type=STATEMENT_PROMPT_TYPE)

            res = self.model.chat(curr_prompt)
            responses["STATEMENT_REPHRASE"].append(res)

            df = pd.DataFrame()
    
            df['Samples'] = samples[:len(responses["ORIGINAL"])]
            df['Labels'] = labels[:len(responses["ORIGINAL"])]
    
            for key, arr in responses.items():
                df[key] = arr

            df.to_csv(f"output/{self.model.name}-checkpoint.csv", index=False)

        return df

            
            
        
        
        
        
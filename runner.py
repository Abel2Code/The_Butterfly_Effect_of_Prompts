from collections import defaultdict
import os

import pandas as pd
from tqdm import tqdm

from prompt_tools.prompt_factory import prompt_factory, prompt_model_factory, STATEMENT_PROMPT_TYPE
from prompt_tools.injectors import INJECTION_POSITION_STRING

class Runner():
    def __init__(self, model, factory_keys, special_col_names, special_keys, special_models, task_name, output_path):
        self.model = model
        self.model.initialize_cache(task_name)
        for model in special_models: model.initialize_cache(task_name)
        
        self.factory_keys = factory_keys
        self.factories = [prompt_factory.create(key) for key in factory_keys]

        assert len(special_keys) == len(special_models) == len(special_col_names)

        self.special_col_names = special_col_names
        self.special_keys = special_keys
        self.special_models = special_models
        self.special_factories = [prompt_factory.create(key) for key in special_keys]
        
        self.output_path = output_path
        
    def run(self, samples, labels, task_type, unique_description=None, allow_explain=False):
        prompt_model = prompt_model_factory.create(task_type)(labels, unique_description, allow_explain)

        # If file exists, load previous_output
        previous_df = pd.DataFrame()
        if os.path.isfile(self.output_path):
            previous_df = pd.read_csv(self.output_path)
        
        responses = defaultdict(list)
        for i, s in tqdm(list(enumerate(samples))):
            if i < len(previous_df):
                assert s == previous_df['Samples'][i]
                
            for key, factory in zip(self.factory_keys, self.factories):
                curr_prompt = factory.generate(s, prompt_model)

                if i < len(previous_df) and key in previous_df:
                    res = previous_df[key][i]
                    self.model.save_to_cache(curr_prompt, res, key)
                else:
                    res = self.model.chat(curr_prompt, key)
                responses[key].append(res)

            for key, model, factory in zip(self.special_col_names, self.special_models, self.special_factories):
                curr_prompt = factory.generate(s, prompt_model)

                if i < len(previous_df) and key in previous_df:
                    res = previous_df[key][i]
                    model.save_to_cache(curr_prompt, res, key) # Note we save to cache over original model
                else:
                    res = model.chat(curr_prompt, key)
                    model.save_to_cache(curr_prompt, res, key)
                responses[key].append(res)

            # Bonus Experiments
            # - Prompt as a statement
            key = "STATEMENT_REPHRASE"
            statement_factory = prompt_factory.create("ORIGINAL")
            curr_prompt = statement_factory.generate(s, prompt_model, prompt_type=STATEMENT_PROMPT_TYPE)

            if i < len(previous_df) and key in previous_df:
                res = previous_df[key][i]
                self.model.save_to_cache(curr_prompt, res, key)
            else:
                res = self.model.chat(curr_prompt, key)
            responses[key].append(res)

            key = "CLEAR_STATEMENT_REPHRASE"
            statement_factory = prompt_factory.create("NO_STYLE")
            curr_prompt = statement_factory.generate(s, prompt_model, prompt_type=STATEMENT_PROMPT_TYPE)

            if i < len(previous_df) and key in previous_df:
                res = previous_df[key][i]
                self.model.save_to_cache(curr_prompt, res, key)
            else:
                res = self.model.chat(curr_prompt, key)
            responses[key].append(res)
            # End Bonus Experiments

            df = pd.DataFrame()
    
            df['Samples'] = samples[:len(responses["ORIGINAL"])]
            df['Labels'] = labels[:len(responses["ORIGINAL"])]
    
            for key, arr in responses.items():
                df[key] = arr

            if i % 10 == 0:
                df.to_csv(f"output/{self.model.name}-checkpoint.csv", index=False)

        for col in df.columns:
            if col in previous_df:
                for v1, v2 in zip(previous_df[col], df[col]):
                    assert v1 == v2, (v1, v2, col)

        for model in [self.model] + list(self.special_models):
            model.write_cache_to_file()

        return df    
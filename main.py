import argparse
import pandas as pd
import os

from runner import Runner
from prompt_tools.prompt_factory import prompt_factory_dict
from models.model_factory import model_factory

supported_models = model_factory._builders.keys()

def main(in_csv_path, model_name, sample_col, label_col, include_cols, task, special_description, allow_explain):
    assert model_name in supported_models

    model = model_factory.create(model_name)
    special_col_names = []
    special_factory_keys = []
    special_models = []
    
    factory_keys = prompt_factory_dict.keys()
    
    if model_name == "ChatGPT":
        special_col_names = ["ChatGPT_JSON_PARAM"]
        special_factory_keys = ["JSON_STYLE"]
        special_models = [model_factory.create("ChatGPT_JSON")]

    task_name = in_csv_path.split('/', maxsplit=1)[-1].split('.')[0]
    output_path = os.path.join("output", model_name, f"{task_name}.csv")

    runner = Runner(model, factory_keys, special_col_names, special_factory_keys, special_models, task_name, output_path)

    in_df = pd.read_csv(in_csv_path)

    try:
        out_df = runner.run(in_df[sample_col], in_df[label_col], task_type=task, unique_description=special_description, allow_explain=allow_explain)
    except KeyboardInterrupt:
        for model in [runner.model] + list(special_models):
            model.write_cache_to_file()
        raise KeyboardInterrupt
    except Exception as e:
        for model in [runner.model] + list(special_models):
            model.write_cache_to_file()
        raise e
        
    out_df.to_csv(output_path, index=False) # In case include_cols crash, save what we have.

    for col in include_cols:
        out_df[col] = in_df[col]
        
    out_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_csv_path")
    parser.add_argument("model")
    parser.add_argument("sample_column")
    parser.add_argument("label_column")
    parser.add_argument('-ic','--include_cols', action='append')
    parser.add_argument('-sd', '--special_description')
    parser.add_argument('-de', '--disallow_explain', action='store_true')
    parser.add_argument('--task', default="SIMPLE_CLASSIFICATION")

    args = parser.parse_args()

    in_csv_path = args.input_csv_path
    model = args.model
    sample_col = args.sample_column
    label_col = args.label_column
    include_cols = args.include_cols if args.include_cols else []
    special_description = args.special_description
    allow_explain = not args.disallow_explain
    task = args.task

    
    # in_csv_path = "datasets/test.csv"
    # sample_col = "samples"
    # label_col = "label"
    # output_path = "output/test.csv"
    
    main(in_csv_path, model, sample_col, label_col, include_cols, task, special_description, allow_explain)

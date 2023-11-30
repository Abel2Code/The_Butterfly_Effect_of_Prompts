import argparse
import pandas as pd

from runner import Runner
from prompt_factory import prompt_factory_dict
from model_factory import model_factory

def chatgpt_main(in_csv_path, sample_col, label_col, output_path):
    model = model_factory.create("ChatGPT")
    special_model = model_factory.create("ChatGPT_JSON")

    factory_keys = prompt_factory_dict.keys()

    special_col_names = ["ChatGPT_JSON_PARAM"]
    special_factory_keys = ["CLEAR"]
    special_models = [special_model]

    runner = Runner(model, factory_keys, special_col_names, special_factory_keys, special_models)

    in_df = pd.read_csv(in_csv_path)

    out_df = runner.run(in_df[sample_col], in_df[label_col])
    out_df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("input_csv_path")
    parser.add_argument("sample_column")
    parser.add_argument("label_column")
    parser.add_argument("output_csv_path")

    args = parser.parse_args()

    in_csv_path = args.input_csv_path
    sample_col = args.sample_column
    label_col = args.label_column
    out_csv_path = args.output_csv_path

    
    # in_csv_path = "datasets/test.csv"
    # sample_col = "samples"
    # label_col = "label"
    # output_path = "output/test.csv"
    
    chatgpt_main(in_csv_path, sample_col, label_col, out_csv_path)
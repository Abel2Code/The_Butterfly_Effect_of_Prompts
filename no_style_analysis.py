import pandas as pd
import re
from collections import Counter, defaultdict
import os
import json
import re

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from scipy.stats import pearsonr

from parse_tools.choice_parsers import choice_parsers, find_choice_text
from parse_tools.parser_factory import parser_factory
from parse_tools.parsers import BAD_COLS, INVALID
from helper import load_file, accuracy_score_filtered, count_differences, plot_bar, plot_pca, type2col_map

models = {
    "ChatGPT": "ChatGPT",
}

datasets = {
    "CoLA": "CoLA",
    "CoPA": "CoPA",
    "ColBERT": "ColBERT",
    "NLI": "SuperGlueNLI",
    "Sarcasm": "iSarcasm",
    "Sentiment": "IMDBSentiment",
    "Stance": "TweetStance",
    "Toxicity": "Jigsaw_Toxicity",

    "BoolQA": "BoolQA",
    "Math": "MathQA",
    "ReAd": "ReAD"
}

dataset_types = {
    "CoLA": "SIMPLE_CLASSIFICATION",
    "CoPA": "SIMPLE_CLASSIFICATION",
    "ColBERT": "SIMPLE_CLASSIFICATION",
    "NLI": "SIMPLE_CLASSIFICATION",
    "Sarcasm": "SIMPLE_CLASSIFICATION",
    "Sentiment": "SIMPLE_CLASSIFICATION",
    "Stance": "SIMPLE_CLASSIFICATION",
    "Toxicity": "SIMPLE_CLASSIFICATION",
    
    "BoolQA": "BOOL",
    "Math": "PROBLEM_SOLVER",
    "ReAd": "PASSAGE_CHOICE"
}

label_formatters = {
    "Math": lambda x: x if x in BAD_COLS else str(float(x))
}

ignore_label_types = ["PROBLEM_SOLVER"]

COLUMNS = [
     'NO_STYLE',
    # 'ORIGINAL', 'NO_STYLE',
    #        'JSON_STYLE', 'XML_STYLE', 'CSV_STYLE',
    #         'YAML_STYLE', 'SPACE_BEFORE_PB', 'SPACE_AFTER_PB', 'HELLO_PB',
    #         'HELLO!_PB', 'HOWDY_PB', 'THANK_YOU_PB', 'AIM_JB', 'DEV_JB', # 'DAN_JB',
    #         'EVIL_JB', 'REFUSAL_JB', 'ChatGPT_JSON_PARAM', 'STATEMENT_REPHRASE',
    #         "WONT_TIP", "TIP_1", "TIP_10", "TIP_100", "TIP_1000",
          
            # Clear Cols
           "CLEAR_SPACE_BEFORE_PB", "CLEAR_SPACE_AFTER_PB", "CLEAR_HELLO_PB", "CLEAR_HELLO!_PB", "CLEAR_HOWDY_PB", "CLEAR_THANK_YOU_PB", "CLEAR_STATEMENT_REPHRASE",
           "CLEAR_WONT_TIP", "CLEAR_TIP_1", "CLEAR_TIP_10", "CLEAR_TIP_100", "CLEAR_TIP_1000",
           "CLEAR_AIM_JB", "CLEAR_DEV_JB", "CLEAR_EVIL_JB", "CLEAR_REFUSAL_JB" ]

STYLE_COLS = {col for col in type2col_map['Output Formats']}

OUTPUT_FOLDER = "parsed_output/clear"
FIGURE_FOLDER = os.path.join(OUTPUT_FOLDER, "figures")

def split_jb_subsection(text, should_return_classic):
    lock_count = text.count('🔓')

    if lock_count == 0:
        return text if should_return_classic else ""
    elif lock_count == 1:
        classic_text, jb_text = text.split('🔓')
        return classic_text if should_return_classic else jb_text
    else:
        classic_text, jb_text = text, ""
        return classic_text if should_return_classic else jb_text   

def analyze(model, dataset, columns, label_col):
    model_name = models[model]
    dataset_name = datasets[dataset]
    data_path = os.path.join("output", model_name, f"{dataset_name}.csv")

    df = pd.read_csv(data_path, dtype=str)

    assert all(col in df for col in columns)

    label_formatter = label_formatters.get(dataset, (lambda x: x))
    df['Labels'] = df['Labels'].map(label_formatter)

    labels = list(set(df['Labels']))
    label_list = df['Labels']

    label_list_file = os.path.join(OUTPUT_FOLDER, "label_list.json")
    label_list_map = load_file(label_list_file)
    label_list_map[dataset] = list(label_list)
    with open(label_list_file, 'w') as f:
        json.dump(label_list_map, f)

    # Split Jailbreaks with Multiple Personalities
    special_jailbreaks = ['CLEAR_DAN_JB', 'CLEAR_DEV_JB']
    for col in special_jailbreaks:
        if col in columns:
            classic_col = f'{col}_CLASSIC'
            jailbreak_col = f'{col}_JAILBREAK'
            
            df[classic_col] = df[col].map(lambda x: split_jb_subsection(x, True))
            df[jailbreak_col] = df[col].map(lambda x: split_jb_subsection(x, False))

            columns.remove(col)
            
            columns.append(classic_col)
            columns.append(jailbreak_col)

    # Load Parser
    current_parser_factory = parser_factory.create(dataset_types[dataset])
    parser_dict = {key: current_parser_factory.create(key) for key in columns}

    # Convert labels into special sequences
    # This prevents issues when some labels are subsets of others
    # I.E. ("Not found", "found")
    use_special_labels = any(l1 in l2 and l1 != l2 for l1 in labels for l2 in labels) and dataset_types[dataset] not in ignore_label_types
    if use_special_labels:
        special_label_map = {label: f"LABEL_ITEM_NUMBER_{i}"for i, label in enumerate(labels)}
        reversed_special_label_map = {v:k for k,v in special_label_map.items()}
        special_labels = list(special_label_map.values())

        assert not any(label in special_labels for label in labels)
    
        def process_text(text, labels=labels, special_label_map=special_label_map):
            sorted_labels = sorted(list(labels), key=len, reverse=True)
            for label in sorted_labels:
                text = re.sub(label, special_label_map[label], text, flags=re.IGNORECASE)
                underscore_label = label.replace(' ', '_')
                text = text.replace(underscore_label, special_label_map[label])
    
            return text
    
        for col in columns:
            df[col] = df[col].map(process_text)

    # Parse Labels
    choice_parser = choice_parsers.get(dataset, None)
    extracted_labels = defaultdict(list)
    invalid_text = defaultdict(lambda: defaultdict(list))
    for i, row in df.iterrows():
        for col in columns:
            temp_labels = special_labels if use_special_labels else labels
            if dataset_types[dataset] == "PROBLEM_SOLVER":
                temp_labels = [label_list[i]]
            text = row[col]
            parsed_label = parser_dict[col].parse(text, temp_labels)
            if use_special_labels and parsed_label in reversed_special_label_map: parsed_label = reversed_special_label_map[parsed_label]
            parsed_label = label_formatter(parsed_label)

            if choice_parser and parsed_label in BAD_COLS:
                choices, label_options = choice_parser(row['Samples'], labels)
                parsed_choice_index = find_choice_text(text, choices)
                    
                if parsed_choice_index is not None:
                    parsed_label = label_options[parsed_choice_index] 

            extracted_labels[col].append(parsed_label)
            
            if parsed_label in BAD_COLS:
                invalid_text[col][parsed_label].append(text)

    invalid_text_file = os.path.join(OUTPUT_FOLDER, "invalid_responses.json")
    invalid_text_json = load_file(invalid_text_file)
    if model not in invalid_text_json: invalid_text_json[model] = {}
    invalid_text_json[model][dataset] = invalid_text
    with open(invalid_text_file, 'w') as f:
        json.dump(invalid_text_json, f)

    # Merge Special Jailbreaks
    for col in special_jailbreaks:
        classic_col = f'{col}_CLASSIC'
        jailbreak_col = f'{col}_JAILBREAK'
        if classic_col in columns and jailbreak_col in columns:
            extracted_labels[col] = [cl if jb in BAD_COLS else jb for (cl, jb) in zip(extracted_labels[classic_col], extracted_labels[jailbreak_col])]

            del extracted_labels[classic_col]
            del extracted_labels[jailbreak_col]
            
            columns.append(col)
            columns.remove(classic_col)
            columns.remove(jailbreak_col)

    # Save Extracted Labels to file
    extracted_labels_file = os.path.join(OUTPUT_FOLDER, "extracted_labels.json")
    extracted_labels_file_json = load_file(extracted_labels_file)
    if model not in extracted_labels_file_json: extracted_labels_file_json[model] = {}
    extracted_labels_file_json[model][dataset] = extracted_labels

    with open(extracted_labels_file, 'w') as f:
        json.dump(extracted_labels_file_json, f)

    # Save Number of Invalid Labels to file
    num_invalid_labels_file = os.path.join(OUTPUT_FOLDER, "num_invalid_labels.json")
    num_invalid_labels_file_json = load_file(num_invalid_labels_file)
    if model not in num_invalid_labels_file_json: num_invalid_labels_file_json[model] = {}
    num_invalid_labels = {col: len([l for l in extracted_labels[col] if l in BAD_COLS]) for col in extracted_labels}
    num_invalid_labels_file_json[model][dataset] = num_invalid_labels

    with open(num_invalid_labels_file, 'w') as f:
        json.dump(num_invalid_labels_file_json, f)

    # Plot PCA
    pca_plot_path = os.path.join(FIGURE_FOLDER, model, f"{dataset}-PCA.png")
    plot_pca(extracted_labels_file_json[model], [dataset], columns, label_list_map, pca_plot_path)

    # Get Overall Statistics
    overall_bars = Counter()
    all_total = 0
    for key in extracted_labels.keys():
        c = Counter(extracted_labels[key])
        
        for k, v in c.items():
            overall_bars[k] += v

    overall_stats_file = os.path.join(OUTPUT_FOLDER, "overall_stats.json")
    overall_stats = load_file(overall_stats_file)
    if model not in overall_stats: overall_stats[model] = {}
    overall_stats[model][dataset] = overall_bars

    with open(overall_stats_file, 'w') as f:
        json.dump(overall_stats, f)

    overall_stats_plot_path = os.path.join(FIGURE_FOLDER, model, f"{dataset}-overall-stats.png")
    plot_bar(overall_bars.keys(), overall_bars.values(), overall_stats_plot_path, type2color_map=None, col2type_map=None)

    # Compute and save overall accuracies
    overall_accuracy_file = os.path.join(OUTPUT_FOLDER, "overall_accuracy.json")
    overall_accuracy = load_file(overall_accuracy_file)
    if model not in overall_accuracy: overall_accuracy[model] = {}
    overall_accuracy_scores = {key: (accuracy_score(label_list, extracted_labels[key]) if len(label_list) != 0 else 0) for key in extracted_labels}
    overall_accuracy[model][dataset] = overall_accuracy_scores

    with open(overall_accuracy_file, 'w') as f:
        json.dump(overall_accuracy, f)

    overall_accuracy_plot_path = os.path.join(FIGURE_FOLDER, model, f"{dataset}-overall-accuracy.png")
    plot_bar(overall_accuracy_scores.keys(), overall_accuracy_scores.values(), overall_accuracy_plot_path, show_value_as_perc=True, should_sort=True, reverse=True, force_start_order=['NO_STYLE'])

    # Compute and save accuracies (Not counting invalid responses)
    filtered_accuracy_file = os.path.join(OUTPUT_FOLDER, "filtered_accuracy.json")
    filtered_accuracy = load_file(filtered_accuracy_file)
    if model not in filtered_accuracy: filtered_accuracy[model] = {}
    filtered_accuracy_scores = {key: accuracy_score_filtered(label_list, extracted_labels[key]) for key in extracted_labels}
    filtered_accuracy[model][dataset] = filtered_accuracy_scores

    with open(filtered_accuracy_file, 'w') as f:
        json.dump(filtered_accuracy, f)
        
    filtered_accuracy_plot_path = os.path.join(FIGURE_FOLDER, model, f"{dataset}-filtered-accuracy.png")
    plot_bar(filtered_accuracy_scores.keys(), filtered_accuracy_scores.values(), filtered_accuracy_plot_path, show_value_as_perc=True, should_sort=True, reverse=True, force_start_order=['NO_STYLE'])

    # Compute and save number of labels changed
    python_labels_changed_file = os.path.join(OUTPUT_FOLDER, "python_labels_changed.json")
    python_labels_changed = load_file(python_labels_changed_file)
    if model not in python_labels_changed: python_labels_changed[model] = {}
    python_labels_changed_scores = {key: count_differences(extracted_labels['NO_STYLE'], extracted_labels[key]) for key in extracted_labels if key != 'NO_STYLE'}
    python_labels_changed[model][dataset] = python_labels_changed_scores

    with open(python_labels_changed_file, 'w') as f:
        json.dump(python_labels_changed, f)

    python_labels_changed_plot_path = os.path.join(FIGURE_FOLDER, model, f"{dataset}-python-labels-changed.png")
    plot_bar(python_labels_changed_scores.keys(), python_labels_changed_scores.values(), python_labels_changed_plot_path, red_values=[num_invalid_labels[k] for k in python_labels_changed_scores], should_sort=True)

    # # Compute and save number of labels changed from styles
    # style_labels_changed_file = os.path.join(OUTPUT_FOLDER, "style_labels_changed.json")
    # style_labels_changed = load_file(style_labels_changed_file)
    # if model not in style_labels_changed: style_labels_changed[model] = {}
    # style_labels_changed_scores = {key: count_differences(extracted_labels['NO_STYLE'], extracted_labels[key]) for key in extracted_labels if key != 'NO_STYLE'}
    # style_labels_changed[model][dataset] = style_labels_changed_scores

    # with open(style_labels_changed_file, 'w') as f:
    #     json.dump(style_labels_changed, f)

    # style_labels_changed_plot_path = os.path.join(FIGURE_FOLDER, model, f"{dataset}-style-labels-changed.png")
    # plot_bar(style_labels_changed_scores.keys(), style_labels_changed_scores.values(), style_labels_changed_plot_path, red_values=[num_invalid_labels[k] for k in style_labels_changed_scores], should_sort=True)

    # Compute and save percent of labels changed
    perc_labels_changed_file = os.path.join(OUTPUT_FOLDER, "perc_labels_changed.json")
    perc_labels_changed = load_file(perc_labels_changed_file)
    if model not in perc_labels_changed: perc_labels_changed[model] = {}
    perc_labels_changed_scores = {key: (count_differences(extracted_labels['NO_STYLE'], extracted_labels[key]) / len(df)) for key in extracted_labels if key != 'NO_STYLE'}
    perc_labels_changed[model][dataset] = perc_labels_changed_scores

    with open(perc_labels_changed_file, 'w') as f:
        json.dump(perc_labels_changed, f)

    perc_labels_changed_plot_path = os.path.join(FIGURE_FOLDER, model, f"{dataset}-perc-labels-changed.png")
    plot_bar(perc_labels_changed_scores.keys(), perc_labels_changed_scores.values(), perc_labels_changed_plot_path, show_value_as_perc=True, should_sort=True)

def aggregate_analyze(model, data_list, columns, label_col):
    def plot_aggregated(data, plot_path, red_value_data={}, variations=[], aggregate_func=lambda arr: (sum(arr) / len(data_list)), force_start_order=[], show_value_as_perc=False, should_sort=False, reverse=False):
        averaged_data = defaultdict(list)
        total_red_values = Counter()
        for d in data_list:
            for col in columns:
                curr_data = data[model][d]
                if col in curr_data:
                    averaged_data[col].append(curr_data[col])

                if model in red_value_data and d in red_value_data[model] and col in red_value_data[model][d]:
                    total_red_values[col] += red_value_data[model][d][col]                    

        averaged_data = {k: aggregate_func(v) for k, v in averaged_data.items() if not variations or k in variations}
        
        plot_bar(averaged_data.keys(), averaged_data.values(), plot_path, red_values=[total_red_values[k] for k in averaged_data], show_value_as_perc=show_value_as_perc, should_sort=should_sort, reverse=reverse, force_start_order=force_start_order)
        
    # Overall Accuracy
    overall_accuracy_file = os.path.join(OUTPUT_FOLDER, "overall_accuracy.json")
    overall_accuracy = load_file(overall_accuracy_file)
    averaged_accuracy_plot_path = os.path.join(FIGURE_FOLDER, model, f"aggregate-average-accuracy.png")
    plot_aggregated(overall_accuracy, averaged_accuracy_plot_path, show_value_as_perc=True, should_sort=True, reverse=True, force_start_order=['NO_STYLE'])
    
    # Overall Filtered Accuracies
    filtered_accuracy_file = os.path.join(OUTPUT_FOLDER, "filtered_accuracy.json")
    filtered_accuracy = load_file(filtered_accuracy_file)
    filtered_accuracy_plot_path = os.path.join(FIGURE_FOLDER, model, f"aggregate-filtered-accuracy.png")
    plot_aggregated(filtered_accuracy, filtered_accuracy_plot_path, show_value_as_perc=True, should_sort=True, reverse=True,)

    # Overall Perc Changed
    perc_labels_changed_file = os.path.join(OUTPUT_FOLDER, "perc_labels_changed.json")
    perc_labels_changed = load_file(perc_labels_changed_file)
    perc_labels_plot_path = os.path.join(FIGURE_FOLDER, model, f"aggregate-perc-labels.png")
    plot_aggregated(perc_labels_changed, perc_labels_plot_path, show_value_as_perc=True, should_sort=True)
    
    # Overall Labels Changed
    num_invalid_labels_file = os.path.join(OUTPUT_FOLDER, "num_invalid_labels.json")
    num_invalid_labels_file_json = load_file(num_invalid_labels_file)
    
    python_labels_changed_file = os.path.join(OUTPUT_FOLDER, "python_labels_changed.json")
    python_labels_changed = load_file(python_labels_changed_file)
    python_labels_plot_path = os.path.join(FIGURE_FOLDER, model, f"aggregate-python-labels.png")
    plot_aggregated(python_labels_changed, python_labels_plot_path, red_value_data=num_invalid_labels_file_json, variations=[col for col in COLUMNS if col not in STYLE_COLS], aggregate_func=lambda arr: sum(arr), should_sort=True)

    # PCA
    label_list_file = os.path.join(OUTPUT_FOLDER, "label_list.json")
    label_list_map = load_file(label_list_file)
    
    extracted_labels_file = f"{OUTPUT_FOLDER}/extracted_labels.json"
    extracted_labels = load_file(extracted_labels_file)[model]
    aggregate_pca_plot_path = os.path.join(FIGURE_FOLDER, model, f"aggregate-PCA.png")
    plot_pca(extracted_labels, data_list, columns, label_list_map, aggregate_pca_plot_path)

    # MDS
    aggregate_mds_plot_path = os.path.join(FIGURE_FOLDER, model, f"aggregate-MDS.png")
    plot_pca(extracted_labels, data_list, columns, label_list_map, aggregate_mds_plot_path, func='MDS')

def main(model, data_list, columns, label_col):
    for model in models:
        for data_name in data_list:
            analyze(model, data_name, columns, label_col)

    aggregate_analyze(model, data_list, columns, label_col)

if __name__ == "__main__":
    model = "ChatGPT"
    data_list = list(datasets.keys())
    main(model, data_list, COLUMNS, "Labels")
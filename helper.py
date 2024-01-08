from collections import defaultdict, Counter
import errno
import os
import signal
import functools
import json

import matplotlib.pyplot as plt

import matplotlib.patches as mpatches
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.metrics import accuracy_score

from parse_tools.parsers import BAD_COLS

colors = list(plt.get_cmap('Dark2').colors)  # Get colors from 'tab20' colormap
type2col_map = {
    "Output Formats": ["ORIGINAL", "JSON_STYLE", "ChatGPT_JSON_PARAM", "XML_STYLE", "CSV_STYLE", "YAML_STYLE", "NO_STYLE"],
    "Perturbations": ["SPACE_BEFORE_PB", "SPACE_AFTER_PB", "HELLO_PB", "HELLO!_PB", "HOWDY_PB", "THANK_YOU_PB", "STATEMENT_REPHRASE"],
    "Special Cases": [],
    "Jailbreaks": ["AIM_JB", "EVIL_JB", "REFUSAL_JB", "DAN_JB", "DEV_JB"],
    "Tipping": ["WONT_TIP", "TIP_1", "TIP_10", "TIP_100", "TIP_1000"]
}

# Add clear versions
for key, arr in type2col_map.items():
    for a in list(arr):
        arr.append(f"CLEAR_{a}")

col2type_map = {v:k for k,v_list in type2col_map.items() for v in v_list}
type2color_map = {key: (colors[i % len(colors) + 1] if key != "ORIGINAL" else "black") for i, key in enumerate(type2col_map.keys())}
type2color_map[col2type_map['JSON_STYLE']] =  (1.0, 0.4980392156862745, 0.0)

markers = ["^", "s", "P", "d", "*", "X", ">"]

clean_name_map = {
    'ORIGINAL': 'Python List Format',
    'THANK_YOU_PB': "End with \"Thank you\"",
    "NO_STYLE": "No Specified Format",
    'CSV_STYLE': "CSV Format",
    'SPACE_AFTER_PB': "End with Space",
    'SPACE_BEFORE_PB': "Start with Space",
    'HOWDY_PB': "Start with \"Howdy!\"",
    'HELLO_PB': "Start with \"Hello.\"",
    'HELLO!_PB': "Start with \"Hello!\"",
    'STATEMENT_REPHRASE': "Rephrase as Statement",
    'JSON_STYLE': "JSON Format",
    'REFUSAL_JB': "Refusal Suppression",
    'XML_STYLE': "XML Format",
    'ChatGPT_JSON_PARAM': "ChatGPT's JSON Checkbox",
    'YAML_STYLE': "YAML Format",
    
    # Jailbreaks
    'DAN_JB': "Do Anything Now",
    'DEV_JB': "Dev Mode v2",
    'EVIL_JB': "Evil Confidant",
    'AIM_JB': "AIM",
    # 'DAN_JB_JAILBREAK': "Do Anything Now 2",
    # 'DEV_JB_JAILBREAK': "Developer Mode 2",
    # 'DAN_JB_CLASSIC': "Do Anything Now 1",
    # 'DEV_JB_CLASSIC': "Developer Mode 1",

    # Tipping
    "WONT_TIP": "Won't Tip",
    "TIP_1": "Tip $1",
    "TIP_10": "Tip $10",
    "TIP_100": "Tip $100",
    "TIP_1000": "Tip $1000",

    # Tasks
    'CoLA': 'CoLA',
    'CoPA': 'CoPA',
    'ColBERT': 'ColBERT',
    'NLI': "GLUE Diagnostic",
    'Sarcasm': 'iSarcasm',
    'Sentiment': 'IMDBSentiment',
    'Stance': 'TweetStance',
    'Toxicity': 'Jigsaw Toxicity',
    'BoolQA': 'BoolQ',
    'Math': 'MathQA',
    'ReAd': 'RACE'
}

# Add clear versions
for key, value in list(clean_name_map.items()):
    clean_name_map[f"CLEAR_{key}"] = value
    

def update_cache(path, data, model_name):
    try:
        original_data = load_cache(path)

        # Verify all previous data is still there
        for k in data:
            if k not in original_data:
                original_data[k] = data[k]
                continue

            for prompt in data[k]:
                if prompt not in original_data[k]:
                    original_data[k][prompt] = data[k][prompt]
                else:
                    assert original_data[k][prompt] == data[k][prompt]
        
        with open(path, 'w') as file:
            json.dump(original_data, file)
    except Exception as e:
        file_name = path.split('/')[-1].split('.')[0]
        with open(f"./temp-cache-{model_name}({file_name}).json", 'w') as file:
                json.dump(data, file)

        print(e)

def load_cache(path):
    return load_file(path)

def load_file(path):
    if os.path.exists(path):
        with open(path, 'r') as file:
            return json.load(file)
    else:
        return {}

def timeout(seconds=10, error_message=os.strerror(errno.ETIME)):
    def decorator(func):
        def _handle_timeout(signum, frame):
            raise TimeoutError(error_message)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            signal.signal(signal.SIGALRM, _handle_timeout)
            signal.alarm(seconds)
            try:
                result = func(*args, **kwargs)
            finally:
                signal.alarm(0)
            return result

        return wrapper

    return decorator

def accuracy_score_filtered(label_list, values, bad_cols=BAD_COLS):
        new_label_pair = [(l, v) for l, v in zip(label_list, values) if v not in bad_cols]
        new_label_list, new_values = [l[0] for l in new_label_pair], [l[1] for l in new_label_pair]
        
        if len(new_label_list) == 0:
            return 0
            
        acc = accuracy_score(new_label_list, new_values)
        return acc

def count_differences(label_list, values):
        return sum(1 for k1, k2 in zip(label_list, values) if k1 != k2)


def plot_bar(keys, values, save_path, red_values=None, should_sort=False, reverse=False, col2type_map=col2type_map, type2color_map=type2color_map, figsize=(10, 6), show_values=True, show_value_as_perc=False, y_label=None, title=None, force_start_order=[]):
    assert bool(col2type_map) == bool(type2color_map)

    if red_values is None:
        red_values = [0 for _ in values]
    
    if should_sort:
        combined = list(zip(keys, values, red_values))        
        sorted_combined = sorted(combined, key=lambda x: x[1], reverse=reverse)
        keys = [x[0] for x in sorted_combined]
        values = [x[1] for x in sorted_combined]
        red_values = [x[2] for x in sorted_combined]

    if force_start_order:
        new_start_keys = []
        new_start_values = []
        new_start_red_values = []
        for k in force_start_order:
            idx = keys.index(k)
            
            new_start_keys.append(keys.pop(idx))
            new_start_values.append(values.pop(idx))
            new_start_red_values.append(red_values.pop(idx))

        keys = new_start_keys + keys
        values = new_start_values + values
        red_values = new_start_red_values + red_values
        
    plt.figure(figsize=figsize)  # Adjust the figure size if needed
    colors = [type2color_map[col2type_map[k]] for k in keys] if type2color_map else None

    formatted_keys = [clean_name_map.get(k, k) for k in keys]
    plt.bar(formatted_keys, values, color=colors)

    plt.bar(formatted_keys, red_values, color='red')
    
    # Add values above each bar
    if show_values:
        for i, val in enumerate(values):
            text = f"{val * 100:.1f}%" if show_value_as_perc else val
            plt.text(i, val, text, ha='center', va='bottom', fontsize='x-small')

    if y_label:
        plt.ylabel(y_label)
    
    if title:
        plt.title(title)
    
    plt.xticks(rotation=75, ha="right")  # Rotate x-axis labels for better readability if needed

    if type2color_map:
        patches = [mpatches.Patch(color=c, label=k) for k, c in type2color_map.items() if any(key in type2col_map[k] for key in keys)]
        if len(patches) > 1:
            plt.legend(handles=patches, loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    plt.savefig(save_path)

    plt.close()

def vectorize_data(data, tasks, categories, true_labels, bad_cols=BAD_COLS):
    output_vectors = defaultdict(list)
    for task in tasks:
        task_data = data[task]
        labels = list(set(l for arr in task_data.values() for l in arr if l not in bad_cols))

        for cat in categories:
            assert len(task_data[cat]) == len(true_labels[task])
            label_map = Counter({True: 1, False: -1})
            new_vector = []
            for pred, label in zip(task_data[cat], true_labels[task]):
                if pred in bad_cols:
                    new_vector.append(0)
                else:
                    if task == 'Math':
                        result = float(pred) == float(label)
                    else:
                        result = pred == label
                        
                    new_vector.append(label_map[result])
            output_vectors[cat] += new_vector

    return output_vectors

def plot_pca(data, tasks, categories, true_labels, save_path, type2col_map=type2col_map, type2color_map=type2color_map, black_key=None, func='PCA'):
    vector_data = vectorize_data(data, tasks, categories, true_labels)

    r_type_map = {v:k for k,v_list in type2col_map.items() for v in v_list}

    # Extract vectors and keys
    keys = [key for type, type_cols in type2col_map.items() for key in type_cols if key in categories]
    assert len(keys) == len(categories)
    vectors = list(vector_data.values())
    
    # Apply PCA for dimensionality reduction to 2 dimensions
    if func == 'PCA':
        pca = PCA(n_components=2, random_state=60)
    elif func == 'MDS':
        pca = MDS(n_components=2, random_state=60)
    else:
        raise Exception("Invalid func type")
        
    transformed_data = pca.fit_transform(vectors)
    
    # Visualize the transformed data with annotations
    plt.figure(figsize=(8, 6))
    type_map_keys = list(type2col_map)

    if black_key in keys:
        x, y = transformed_data[keys.index(black_key)]
        plt.scatter(x, y, label=clean_name_map[black_key], marker="o", color='black')
        
    for i, key in enumerate(keys):
        if key == black_key:
            continue
            
        x, y = transformed_data[i]
        
        key_type = r_type_map[key]
        marker = markers[type2col_map[key_type].index(key) % len(markers)]
        
        plt.scatter(x, y, label=clean_name_map[key], marker=marker, color=type2color_map[key_type])

    plt.tick_params(left = False, right = False , labelleft = False , 
                labelbottom = False, bottom = False) 
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(False)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

class Factory:
    def __init__(self):
        self._builders = {}

    def register_builder(self, key, builder):
        self._builders[key] = builder

    def create(self, key, **kwargs):
        builder = self._builders.get(key)
        if not builder:
            raise ValueError(key)
            
        return builder
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
from sklearn.metrics import accuracy_score

from parse_tools.parsers import BAD_COLS

colors = plt.get_cmap('Dark2').colors  # Get colors from 'tab20' colormap
type2col_map = {
    "ORIGINAL": ["ORIGINAL"],
    "STYLES": ["JSON_STYLE", "ChatGPT_JSON_PARAM", "XML_STYLE", "CSV_STYLE", "YAML_STYLE", "NO_STYLE"],
    "PB": ["SPACE_BEFORE_PB", "SPACE_AFTER_PB", "HELLO_PB", "HELLO!_PB", "HOWDY_PB", "THANK_YOU_PB"],
    "Special Cases": ["STATEMENT_REPHRASE"],
    "JB": ["AIM_JB", "EVIL_JB", "REFUSAL_JB", "DAN_JB", "DEV_JB"],
    "TIP": ["WONT_TIP", "TIP_1", "TIP_10", "TIP_100", "TIP_1000"]
}

col2type_map = {v:k for k,v_list in type2col_map.items() for v in v_list}
type2color_map = {key: (colors[i % len(colors)] if key != "ORIGINAL" else "black") for i, key in enumerate(type2col_map.keys())}

markers = ["^", "s", "P", "d", "*", "X", ">"]


def update_cache(path, data):
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
        with open(f"./temp-cache({file_name}).json", 'w') as file:
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
        for k in force_start_order:
            idx = keys.index(k)
            keys.pop(idx)

        keys = force_start_order + keys
        
    plt.figure(figsize=figsize)  # Adjust the figure size if needed
    colors = [type2color_map[col2type_map[k]] for k in keys] if type2color_map else None
    plt.bar(keys, values, color=colors)

    plt.bar(keys, red_values, color='red')
    
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
        plt.legend(handles=[mpatches.Patch(color=c, label=k) for k, c in type2color_map.items()], loc='center left', bbox_to_anchor=(1, 0.5))
    
    plt.tight_layout()
    
    plt.savefig(save_path)

    plt.close()

def vectorize_data(data, tasks, categories, true_labels, bad_cols=BAD_COLS):
    output_vectors = defaultdict(list)
    for task in tasks:
        task_data = data[task]
        labels = list(set(l for arr in task_data.values() for l in arr if l not in bad_cols))
    
        if len(labels) == 2:
            for cat in categories:
                label_map = Counter({labels[0]: -1, labels[1]: 1})
                output_vectors[cat] += [label_map[pred] for pred in task_data[cat]]
        elif len(labels) < 35:
            # One hot encode
            for cat in categories:
                new_vector = []
                for pred in task_data[cat]:
                    for label in labels:
                        new_vector.append(int(label == pred))
    
                    new_vector.append(int(pred in bad_cols))

                output_vectors[cat] += new_vector
        else:
            for cat in categories:
                assert len(task_data[cat]) == len(true_labels[task])
                label_map = Counter({True: -1, False: 1})
                new_vector = []
                for pred, label in zip(task_data[cat], true_labels[task]):
                    if pred in bad_cols:
                        new_vector.append(0)
                    else:
                        new_vector.append(label_map[float(pred) == float(label)])
                output_vectors[cat] += new_vector

    return output_vectors

def plot_pca(data, tasks, categories, true_labels, save_path, type2col_map=type2col_map, type2color_map=type2color_map):
    vector_data = vectorize_data(data, tasks, categories, true_labels)

    r_type_map = {v:k for k,v_list in type2col_map.items() for v in v_list}

    # Extract vectors and keys
    keys = [key for type, type_cols in type2col_map.items() for key in type_cols if key in categories]
    assert len(keys) == len(categories)
    vectors = list(vector_data.values())
    
    # Apply PCA for dimensionality reduction to 2 dimensions
    pca = PCA(n_components=2)
    transformed_data = pca.fit_transform(vectors)
    
    # Visualize the transformed data with annotations
    plt.figure(figsize=(8, 6))
    type_map_keys = list(type2col_map)
        
    for i, key in enumerate(keys):
        x, y = transformed_data[i]
        
        key_type = r_type_map[key]
        marker = markers[type2col_map[key_type].index(key)] if key != "ORIGINAL" else "o"
        
        plt.scatter(x, y, label=key, marker=marker, color=type2color_map[key_type])
        # plt.text(x, y, key, fontsize=9, ha='right', va='bottom')
    
    plt.title('Similarity of N-dimensional Vectors')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.grid(True)
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
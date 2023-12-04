import ast
from collections import Counter
import json
from lxml import etree
import re

EQUAL_LABEL_COUNTS = "EQUAL"
INVALID = "INVALID"
REFUSAL = "REFUSAL"
EMPTY = "EMPTY_STR"

def simple_equality(text, labels):
    text = text.lower()
    
    if text[0] in '[' and text[-1] in ']':
        text = text[1:-1]
    
    if text[0] in '\'"' and text[-1] in '\'"':
        text = text[1:-1]
        
    lower_labels = [l.lower() for l in labels]
    if text in lower_labels: 
        return labels[lower_labels.index(text)]

    raise Exception("Label not found")

def list_parser(text, labels):
    text = text.replace("'", "\"").strip()
    
    if text[0] == '{' and text[-1] == '}':
        text = '[' + text[1:-1] + ']'
    
    arr = json.loads(text)
    assert type(arr) == list
    
    arr = [a.lower() for a in arr]
    lower_labels = [l.lower() for l in labels]
    
    for a in arr:
        assert a in lower_labels
        
    frequencies = Counter(arr)
    max_frequency = max(frequencies.values())
    frequent_items = [k for k,v in frequencies.items() if v == max_frequency]
    
    if len(frequent_items) != 1:
        return EQUAL_LABEL_COUNTS
    
    lower_label = frequent_items[0]
    
    return labels[lower_labels.index(lower_label)]

def json_parser(text, labels):            
    j = json.loads(text, strict=False)
    assert type(j) == dict
    
    if any(l.lower() == "false" for l in labels):
        print("JSON_PARSER DOES NOT SUPPORT FALSE LABELS")
        raise Exception("Invalid label types")

    for k, v in j.items():
        if type(v) == str and v.strip().lower() == "false":
            j[k] = False
           
    new_j = {}
           
    # Remove False Labels 
    for k, v in j.items():
        if v is False:
            if len(labels) == 2:
                other_label = [l for l in labels if l != k][0]
                new_j[other_label] = True
        else:
            new_j[k] = v
    
    # Check if any label is a key
    labels_with_keys = [l for l in labels if l in j]
    labels_with_values = [l for l in labels for v in j.values() if l == v]
           
    if labels_with_keys:
        # Try removing any labels where 
        frequencies = Counter(labels_with_keys)
    elif labels_with_values:
        frequencies = Counter(labels_with_values)
    else:
        raise Exception("Label not found")
           
    max_frequency = max(frequencies.values())
    frequent_items = [k for k,v in frequencies.items() if v == max_frequency]

    if len(frequent_items) != 1:
        return EQUAL_LABEL_COUNTS

    return frequent_items[0]  

def xml_parser(text, labels):    
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(text, parser=parser)
    
    label, attrib = tree.tag, tree.text
    
    if any(l.lower() == "false" for l in labels):
        print("XML_PARSER DOES NOT SUPPORT FALSE LABELS")
        raise Exception("Invalid label types")
        
    if label in labels and attrib in labels and label != attrib:
        raise Exception("Invalid format")
    elif label in labels:
        if attrib.lower().strip() == "false":
            if len(label) == 2:
                other_label = [l for l in labels if l != k][0]
                return other_label
            else:
                raise Exception("Label is not binary")
        else:
            return label
    elif attrib in labels:
        return attrib
    
    raise Exception("Label not found")
    
def angle_brace_parser(text, labels):
    assert text[0] == "<" and text[-1] == '>'
    text = text[1:-1]
    
    if text[-1] == '/': text = text[:-1]
    
    if text in labels:
        return text
    else:
        raise Exception("Label not found")
        
def yaml_parser(text, labels):
    assert text.count(":") == 1
    
    lower_labels = [l.lower() for l in labels]
    
    tag, attrib = text.split(':')
    tag, attrib = tag.lower().strip(), attrib.lower().strip()
    
    out_tag = None
    if tag in lower_labels and attrib in lower_labels and tag != attrib:
        raise Exception("Invalid format")
    elif tag in lower_labels:
        if attrib == "false":
            if len(labels) == 2:
                other_label = [l for l in lower_labels if l != tag][0]
                out_tag = other_label
            else:
                print("HERE")
                raise Exception("Label is not binary")
        else:
            out_tag = tag
    elif attrib in lower_labels:
        out_tag = attrib
        
    assert out_tag
    
    return labels[lower_labels.index(out_tag)]

def regex_parser(text, labels):
    def count_str(text, pattern, add_bounds):
        if add_bounds:
            pattern = r'(?:^|\s)' + pattern + r'\b' 
        matches = re.findall(pattern, text, re.IGNORECASE)  
        return len(matches)

    def extract_label_helper(text, labels, label_patterns, add_bounds=False):
        found_label = None

        c = Counter()
        max_count = -1
        for label, pattern in zip(labels, label_patterns):
            count = count_str(text, pattern, add_bounds)
            c[label] = count
            if count > max_count: max_count = count

        max_labels = [l for l in labels if c[l] == max_count]
        if len(max_labels) != 1:
            return False

        return max_labels[0]
    
    # Attempt to find label in quotes
    extracted_label = extract_label_helper(text, labels, [f'"{label}\.?"' for label in labels])
    if extracted_label: return extracted_label

    extracted_label = extract_label_helper(text, labels, [f"'{label}\.?'" for label in labels])
    if extracted_label: return extracted_label

    extracted_label = extract_label_helper(text, labels, labels, add_bounds=True)
    if extracted_label: return extracted_label

    raise Exception("Neither Label Found")

def refusal_parser(text, labels):
    text = text.lower()
    
    refusal_substrings = ["sorry", "cannot fulfill"]
    assert any(s in text for s in refusal_substrings)
    
    return REFUSAL

def empty_string_parser(text, labels):
    if text.strip() == '':
        return EMPTY
    
    raise Exception("String not empty")

def set_invalid_parser(text, labels):
    return INVALID
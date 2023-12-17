import ast
from collections import Counter
import json
from lxml import etree
import re

EQUAL_LABEL_COUNTS = "EQUAL"
INVALID = "INVALID"
REFUSAL = "REFUSAL"
EMPTY = "EMPTY_STR"

BAD_COLS = [EQUAL_LABEL_COUNTS, INVALID, REFUSAL, EMPTY]

def simple_equality(text, labels):
    text = text.lower().split('.')[0]
    
    if text[0] in '[' and text[-1] in ']':
        text = text[1:-1]
    
    if text[0] in '\'"' and text[-1] in '\'"':
        text = text[1:-1]

    text = text.strip()
        
    lower_labels = [l.lower() for l in labels]
    if text in lower_labels: 
        return labels[lower_labels.index(text)]

    raise Exception("Label not found")

def quote_finder(text, labels):
    text = text.lower()

    counts = Counter()
    for label in labels:
        lower_label = label.lower()
        counts[label] += text.count(f'"{lower_label}"')
        counts[label] += text.count(f"'{lower_label}'")

    max_count = max(counts.values())
    frequent_labels = [k for k,v in counts.items() if v == max_count]

    if max_count == 0:
        raise Exception("Label not found")
        
    if len(frequent_labels) != 1:
        return EQUAL_LABEL_COUNTS
    
    return frequent_labels[0]

def list_parser(text, labels):
    text = text.replace("'", "\"").strip()
    
    if text[0] == '{' and text[-1] == '}':
        text = '[' + text[1:-1] + ']'

    if text.count('[') == 1 and text.count(']') == 1:
        text = text[text.index('['):text.index(']') + 1]

    
    try:
        arr = json.loads(text)
    except:
        return simple_equality(text, labels)
        
    assert type(arr) == list
    
    lower_labels = [l.lower() for l in labels]
    arr = [a.lower() for a in arr if a.lower() in lower_labels]
    assert arr
        
    frequencies = Counter(arr)
    max_frequency = max(frequencies.values())
    frequent_items = [k for k,v in frequencies.items() if v == max_frequency]
    
    if len(frequent_items) != 1:
        return EQUAL_LABEL_COUNTS
    
    lower_label = frequent_items[0]
    
    return labels[lower_labels.index(lower_label)]

def true_false_json_parser(text, labels):
    j = json.loads(text, strict=False)
    assert type(j) == dict

    assert not any(k.lower() in ["true", "false"] for k in j)
    labels_with_values = [l for l in labels for v in j.values() if l.lower() == str(v).lower()]

    frequencies = Counter(labels_with_values)
    max_frequency = max(frequencies.values())
    frequent_items = [k for k,v in frequencies.items() if v == max_frequency]
    
    if len(frequent_items) != 1:
        return EQUAL_LABEL_COUNTS
    
    return frequent_items[0]
    
def json_parser(text, labels):
    # Check for true false labels
    if all(l.lower() in ["true", "false"] for l in labels):
        return true_false_json_parser(text, labels)
        
    # Main Function    
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

def true_false_xml_parser(text, labels):
    # Main Function
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(text, parser=parser)

    lower_labels = [l.lower() for l in labels]

    stack = [tree]
    while stack:
        tree = stack.pop()
        
        for child in tree.getchildren():
            stack.append(child)
            
        label, attrib = tree.tag, tree.text

        if attrib.lower() in lower_labels:
            return attrib

        for value in tree.attrib.values():
            if value.lower() in lower_labels:
                return value
    
    raise Exception("Label not found")

def xml_parser(text, labels):    
    # Check for true false labels
    if all(l.lower() in ["true", "false"] for l in labels):
        return true_false_xml_parser(text, labels)

    # Main Function
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

def csv_parser(text, labels):
    lower_labels = [l.lower() for l in labels]
    sections = [s.strip() for s in text.split(',')]
    frequencies = Counter([s for s in sections if s.lower() in lower_labels])

    if len(frequencies) == 0:
        raise Exception("Label not found")
        
    if len(frequencies) > 1:
        return EQUAL_LABEL_COUNTS
    
    return list(frequencies.keys())[0]
    
        
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
            pattern = r'\b' + pattern + r'\b' 
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

# Pasage Parsers

def single_char_and_dot_search(text, labels):
    assert all(len(l) == 1 for l in labels)

    labels_found = [l for l in labels if f"{l}." in text]

    assert len(labels_found) == 1

    return labels_found[0]

def single_char_bracket_search(text, labels):
    assert all(len(l) == 1 for l in labels)

    labels_found = [l for l in labels if f"[{l}]" in text]

    assert len(labels_found) == 1

    return labels_found[0]

def single_char_special_yaml_search(text, labels):
    assert all(len(l) == 1 for l in labels)

    replace_token = "REPLACE"
    templates = [f"{replace_token}: true", f"answer: {replace_token}", f"choice: {replace_token}"]

    labels_found = []
    for template in templates:
        for label in labels:
            pattern = template.replace(replace_token, label) + r'\b' 
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches: labels_found.append(label)

    assert len(labels_found) == 1

    return labels_found[0]

def xml_choice_parser(text, labels):
    # Main Function
    parser = etree.XMLParser(recover=True)
    tree = etree.fromstring(text, parser=parser)

    labels_found = set()

    stack = [tree]
    while stack:
        tree = stack.pop()
        for child in tree.getchildren():
            stack.append(child)
            
        tag, text, attrib = tree.tag, tree.text, tree.attrib

        if tag not in ['choice', 'option']:
            continue

        if "correct" not in attrib:
            continue

        if str(attrib['correct']).lower() != "true":
            continue

        
        if text in labels:
            labels_found.add(text)
            continue

        try:
            label = single_char_and_dot_search(text, labels)
            labels_found.add(label)
        except:
            pass

    assert len(labels_found) == 1

    return list(labels_found)[0]

# Math Parsers
def trim_to_answer(func):
    def trim_func(text, labels):
        trim_strs = ["Answer:", "answer:"]
        for s in trim_strs:
            if s in text:
                text = text.split(s, maxsplit=1)[1].strip()
                break
        return func(text, labels)
         
    return trim_func

@trim_to_answer
def regex_number_finder(text, labels):
    pattern = '(-?\d+([,\.\d]+)?)'
    matches = [m[0] for m in re.findall(pattern, text, re.IGNORECASE)]
    matches = [m.replace(',', '') for m in matches if all(len(s) == 3 for s in m.split('.', maxsplit=1)[0].split(',')[1:])] # Check for 3 digits between commas
    matches = [m[:-1] if m.endswith('.') else m for m in matches]

    if len(matches) != 1:
        raise Exception("No Numbers Found")

    assert len(matches) == 1

    match = matches[0]
    
    return match

@trim_to_answer
def list_number_finder(text, labels):
    assert text.count("[") == 1 and text.count("]") == 1
    text = text[text.index("[")+1:text.index("]")]
    text = text.replace('"', '').replace("'", "")
    
    numbers_found = []

    for arr_val in text.split(', '):
        arr_val = arr_val.strip()
        # Check for valid characters
        if not all(c in ".,+-$" or c.isnumeric() for c in arr_val):
            continue

        # Check that comma separated numbers come 3 at a time
        if "," in arr_val and not all(len(s) == 3 for s in arr_val.split('.', maxsplit=1)[0].split(',')[1:]):
            continue
        
        arr_val = arr_val.replace(',', '').replace('$', '')

        try:
            numbers_found.append(float(arr_val))
        except:
            continue

    if len(numbers_found) != 1:
        raise Exception("No Numbers Found")

    assert len(numbers_found) == 1
    return numbers_found[0]

@trim_to_answer
def json_number_finder(text, labels):
    assert text.count("{") == 1 and text.count("}") == 1
    text = text[text.index("{")+1:text.index("}")]
    text = "[" + text.replace(':', ',') + "]"

    return list_number_finder(text, labels)

@trim_to_answer
def csv_number_finder(text, labels):
    text = "[" + text.replace(",", ", ") + "]"

    return list_number_finder(text, labels)

@trim_to_answer
def regex_find_label(text, labels):
    assert len(labels) == 1
    label = labels[0]
    
    pattern = '(-?\d+([,\.\d]+)?)'
    matches = [m[0] for m in re.findall(pattern, text, re.IGNORECASE)]
    matches = [m.replace(',', '') for m in matches if all(len(s) == 3 for s in m.split('.', maxsplit=1)[0].split(',')[1:])] # Check for 3 digits between commas
    matches = [m[:-1] if m.endswith('.') else m for m in matches]

    for m in matches:
        if float(label) == float(m):
            return label

    raise Exception("Label not found")

    

# Invalid Parsers
def set_invalid_parser(text, labels):
    return INVALID

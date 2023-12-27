import re

from .parsers import xml_choice_parser

def get_CoPA_choices(sample, labels):
    label_options = ["Alternative 1", "Alternative 2"]
    assert len(labels) == len(label_options)
    assert all(l in labels for l in label_options)

    assert all(sample.count(f"{o}:") == 1 for o in label_options)

    _, sample = sample.split(f"{label_options[0]}:")
    choices = sample.split(f"{label_options[1]}:")
    choices = [c.strip() for c in choices]
    
    assert len(choices) == len(label_options) 
    return choices, label_options 

def get_ReAd_choices(sample, labels):
    label_options = ["A", "B", "C", "D"]
    assert len(labels) == len(label_options)
    assert all(l in labels for l in label_options)

    sample = sample.split("Choices:", maxsplit=1)[-1]

    assert all(sample.count(f"{o}.") == 1 for o in label_options)

    _, sample = sample.split(f"{label_options[0]}.")
    choice_a, more_choices = sample.split(f"{label_options[1]}.")
    choice_b, more_choices = more_choices.split(f"{label_options[2]}.")
    choice_c, choice_d = more_choices.split(f"{label_options[3]}.")
    choices = [choice_a, choice_b, choice_c, choice_d]
    choices = [c.strip() for c in choices]
    
    assert len(choices) == len(label_options) 
    return choices, label_options 

choice_parsers = {
    "CoPA": get_CoPA_choices,
    "ReAd": get_ReAd_choices
}

def find_choice_text(response, choices):        
    if response.startswith("```"):
        response = response[3:]

    if response.startswith("xml") or response.startswith("XML"):
        response = response[3:]
        
    if response.endswith("```"):
        response = response[:-3]

    response = response.strip()

    if response.startswith('<') and response.endswith('>'):
        try:
            choice = xml_choice_parser(response, choices)
            return choices.index(choice)
        except Exception:
            pass

    # Trim choices
    choices = [c[:-1] if c.endswith('.') else c for c in choices]
    choices = [c.split(' ') for c in choices]

    while len(set([c[0] for c in choices])) == 1:
        choices = [c[1:] for c in choices]

    while len(set([c[-1] for c in choices])) == 1:
        choices = [c[:-1] for c in choices]

    choices = [' '.join(c) for c in choices]

    assert all(choices)
    
    results = []
    
    for choice in choices:
        if re.findall(r"\b" + re.escape(choice) + r"\b", response, re.IGNORECASE):
            results.append(choice)
    
    if len(results) != 1:
        return None

    return choices.index(results[0])
    
import sys
sys.path.append("..")

from helper import Factory
from .parsers import *

class Parser:
    def __init__(self, main_parser, backup_parsers=[], final_parser=set_invalid_parser, override_labels=None, verbose=False):
        self.main_parser = main_parser
        self.backup_parsers = [p for p in backup_parsers]
        self.backup_parsers.append(final_parser)
        self.override_labels = override_labels
        
    def parse(self, text, labels):
        if self.override_labels:
            labels = self.override_labels

        # Initial JB Cleaning
        jb_prefixes = ["😈:", "😈", "AIM:"]
        for p in jb_prefixes:
            while text.startswith(p):
                text = text[len(p):].strip()
                
        if text.endswith("😈"): text = text[:-1].strip()
        
        # Initial General Cleaning
        if text.startswith('```'): text = text[3:].strip()
        if text.endswith('```'): text = text[:-3].strip()
        
        starter_formats = ['json', 'xml', 'yaml', 'python']
        for f in starter_formats:
            if text.startswith(f):
                text = text[len(f):].strip()
                break
                
        if text.startswith('-'): text = text[1:].strip()
        
        # Parsing
        out = None
        
        parser_list = [self.main_parser] + self.backup_parsers           

        for parser in parser_list:
            try:
                out = parser(text, labels)
                return out
            except Exception as e:
                continue

        raise Exception("The following text could not be parsed: " + text)

NO_STYLE_TYPE = "NO_STYLE"
PYTHON_TYPE = "PYTHON"
JSON_TYPE = "JSON"
XML_TYPE = "XML"
CSV_TYPE = "CSV"
YAML_TYPE = "YAML"
JB_TYPE = "JAILBREAK"
def generate_parser_factory_dict(parser_type_dict):
    return {
        "ORIGINAL": parser_type_dict[PYTHON_TYPE],

        # Output Style Prompts
        "NO_STYLE": parser_type_dict[NO_STYLE_TYPE],
        "JSON_STYLE": parser_type_dict[JSON_TYPE],
        "XML_STYLE": parser_type_dict[XML_TYPE],
        "CSV_STYLE": parser_type_dict[CSV_TYPE],
        "YAML_STYLE": parser_type_dict[YAML_TYPE],
        # Special Output Style
        "ChatGPT_JSON_PARAM": parser_type_dict[JSON_TYPE],
    
        # Peturbations
        "SPACE_BEFORE_PB": parser_type_dict[PYTHON_TYPE],
        "SPACE_AFTER_PB": parser_type_dict[PYTHON_TYPE],
        "HELLO_PB": parser_type_dict[PYTHON_TYPE],
        "HELLO!_PB": parser_type_dict[PYTHON_TYPE],
        "HOWDY_PB": parser_type_dict[PYTHON_TYPE],
        "THANK_YOU_PB": parser_type_dict[PYTHON_TYPE],
        # Special Peturbations
        "STATEMENT_REPHRASE": parser_type_dict[PYTHON_TYPE],
    
        # Jailbreaks
        "AIM_JB": parser_type_dict[JB_TYPE],
        "DAN_JB": parser_type_dict[JB_TYPE],
        "DEV_JB": parser_type_dict[JB_TYPE],
        "EVIL_JB": parser_type_dict[JB_TYPE],
        "REFUSAL_JB": parser_type_dict[JB_TYPE],
        # Special Jailbreaks
        "DAN_JB_CLASSIC": parser_type_dict[JB_TYPE],
        "DEV_JB_CLASSIC": parser_type_dict[JB_TYPE],
        "DAN_JB_JAILBREAK": parser_type_dict[JB_TYPE],
        "DEV_JB_JAILBREAK": parser_type_dict[JB_TYPE],

        # Tipping
        "WONT_TIP": parser_type_dict[PYTHON_TYPE],
        "TIP_1": parser_type_dict[PYTHON_TYPE],
        "TIP_10": parser_type_dict[PYTHON_TYPE],
        "TIP_100": parser_type_dict[PYTHON_TYPE],
        "TIP_1000": parser_type_dict[PYTHON_TYPE],

        # Clear
        "CLEAR_SPACE_BEFORE_PB": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_SPACE_AFTER_PB": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_HELLO_PB": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_HELLO!_PB": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_HOWDY_PB": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_THANK_YOU_PB": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_STATEMENT_REPHRASE": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_WONT_TIP": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_TIP_1": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_TIP_10": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_TIP_100": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_TIP_1000": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_AIM_JB": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_DEV_JB": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_EVIL_JB": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_REFUSAL_JB" : parser_type_dict[NO_STYLE_TYPE],

        "CLEAR_DAN_JB_CLASSIC": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_DEV_JB_CLASSIC": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_DAN_JB_JAILBREAK": parser_type_dict[NO_STYLE_TYPE],
        "CLEAR_DEV_JB_JAILBREAK": parser_type_dict[NO_STYLE_TYPE],
    }

def create_parser_factory(parser_type_dict):
    factory = Factory()
    factory_dict = generate_parser_factory_dict(parser_type_dict)

    for k, v in factory_dict.items():
        factory.register_builder(k, v)

    return factory

NO_STYLE_BACKUP_PARSERS = [regex_parser, refusal_parser, empty_string_parser]
PYTHON_BACKUP_PARSERS = [simple_equality, regex_parser]
JSON_BACKUP_PARSERS = [list_parser, regex_parser]
XML_BACKUP_PARSERS = [angle_brace_parser, quote_finder, regex_parser]
CSV_BACKUP_PARSERS = [simple_equality, regex_parser]
YAML_BACKUP_PARSERS = [yaml_tag_parser, simple_equality, regex_parser]
JB_BACKUP_PARSERS = [simple_equality, regex_parser, refusal_parser, empty_string_parser]

# Simple Classification Parser
simple_classification_parser_factory = create_parser_factory({
    PYTHON_TYPE: Parser(list_parser, backup_parsers=PYTHON_BACKUP_PARSERS),
    JSON_TYPE: Parser(json_parser, backup_parsers=JSON_BACKUP_PARSERS),
    XML_TYPE: Parser(xml_parser, backup_parsers=XML_BACKUP_PARSERS),
    CSV_TYPE: Parser(csv_parser, backup_parsers=CSV_BACKUP_PARSERS),
    YAML_TYPE: Parser(yaml_parser, backup_parsers=YAML_BACKUP_PARSERS),
    JB_TYPE: Parser(list_parser, backup_parsers=JB_BACKUP_PARSERS),
    NO_STYLE_TYPE: Parser(list_parser, backup_parsers=NO_STYLE_BACKUP_PARSERS)
})

# Passage Choice
passage_parsers_to_exclude = [regex_parser]
passage_parsers_to_add = [letter_choice_list_parser, json_parser, json_choice_parser, single_char_and_dot_search, single_char_bracket_search]
jb_fail_parsers = [refusal_parser, empty_string_parser]
passage_choice_parser_factory = create_parser_factory({
    PYTHON_TYPE: Parser(list_parser, 
                        backup_parsers=[p for p in PYTHON_BACKUP_PARSERS if p not in passage_parsers_to_exclude] + passage_parsers_to_add),
    JSON_TYPE: Parser(json_parser, 
                      backup_parsers=[p for p in JSON_BACKUP_PARSERS if p not in passage_parsers_to_exclude] + passage_parsers_to_add),
    XML_TYPE: Parser(xml_choice_parser, 
                     backup_parsers=[p for p in XML_BACKUP_PARSERS if p not in passage_parsers_to_exclude] + passage_parsers_to_add + 
                     [xml_parser]),
    CSV_TYPE: Parser(csv_parser, 
                     backup_parsers=[p for p in CSV_BACKUP_PARSERS if p not in passage_parsers_to_exclude] + passage_parsers_to_add),
    YAML_TYPE: Parser(yaml_parser, 
                      backup_parsers=[p for p in YAML_BACKUP_PARSERS if p not in passage_parsers_to_exclude] + passage_parsers_to_add + 
                      [single_char_special_yaml_search]),
    JB_TYPE: Parser(list_parser, 
                    backup_parsers=[p for p in JB_BACKUP_PARSERS if p not in passage_parsers_to_exclude] + passage_parsers_to_add + jb_fail_parsers),
    NO_STYLE_TYPE: Parser(list_parser, 
                          backup_parsers=[p for p in NO_STYLE_BACKUP_PARSERS if p not in passage_parsers_to_exclude] + passage_parsers_to_add + jb_fail_parsers)
})

# Number Finder
number_finder_backups = [regex_find_label, regex_number_finder, find_first_number]
number_finder_parser_factory = create_parser_factory({
    PYTHON_TYPE: Parser(list_number_finder, 
                        backup_parsers=number_finder_backups),
    JSON_TYPE: Parser(json_number_finder, 
                      backup_parsers=number_finder_backups),
    XML_TYPE: Parser(list_number_finder, 
                     backup_parsers=number_finder_backups),
    CSV_TYPE: Parser(csv_number_finder, 
                     backup_parsers=number_finder_backups),
    YAML_TYPE: Parser(list_number_finder, 
                      backup_parsers=number_finder_backups),
    JB_TYPE: Parser(list_number_finder, 
                    backup_parsers=number_finder_backups + jb_fail_parsers),
    NO_STYLE_TYPE: Parser(list_number_finder, backup_parsers=number_finder_backups + jb_fail_parsers)
})

bool_parsers_to_add = [yes_no_to_bool_parser]
bool_parser_factory = create_parser_factory({
    PYTHON_TYPE: Parser(list_parser, 
                        backup_parsers=PYTHON_BACKUP_PARSERS + bool_parsers_to_add),
    JSON_TYPE: Parser(json_parser, 
                      backup_parsers=JSON_BACKUP_PARSERS + bool_parsers_to_add),
    XML_TYPE: Parser(xml_parser, 
                     backup_parsers=XML_BACKUP_PARSERS + bool_parsers_to_add),
    CSV_TYPE: Parser(csv_parser, 
                     backup_parsers=CSV_BACKUP_PARSERS + bool_parsers_to_add),
    YAML_TYPE: Parser(yaml_parser, 
                      backup_parsers=YAML_BACKUP_PARSERS + bool_parsers_to_add),
    JB_TYPE: Parser(list_parser, 
                    backup_parsers=JB_BACKUP_PARSERS + bool_parsers_to_add + jb_fail_parsers),
    NO_STYLE_TYPE: Parser(list_parser, 
                          backup_parsers=NO_STYLE_BACKUP_PARSERS + bool_parsers_to_add + jb_fail_parsers)
})


# Factory Factory
parser_factory = Factory()

parser_factory_dict = {
    "SIMPLE_CLASSIFICATION": simple_classification_parser_factory,
    "PROBLEM_SOLVER": number_finder_parser_factory,
    "PASSAGE_CHOICE": passage_choice_parser_factory,
    "BOOL": bool_parser_factory,
}

for k, v in parser_factory_dict.items():
    parser_factory.register_builder(k, v)
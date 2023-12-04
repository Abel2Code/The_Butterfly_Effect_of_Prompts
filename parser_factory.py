from helper import Factory
from parsers import *

class Parser:
    def __init__(self, main_parser, backup_parsers=[], verbose=False):
        self.main_parser = main_parser
        self.backup_parsers = backup_parsers

    def parse(self, text, labels):
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

parser_factory = Factory()

PYTHON_PARSER = Parser(list_parser, backup_parsers=[set_invalid_parser])
JSON_PARSER = Parser(json_parser, backup_parsers=[list_parser, set_invalid_parser])
XML_PARSER = Parser(xml_parser, backup_parsers=[angle_brace_parser, set_invalid_parser])
CSV_PARSER = Parser(simple_equality, backup_parsers=[set_invalid_parser])
YAML_PARSER = Parser(yaml_parser, backup_parsers=[simple_equality, set_invalid_parser])

JB_PARSER = Parser(list_parser, backup_parsers=[simple_equality, regex_parser, refusal_parser, empty_string_parser, set_invalid_parser])

              
parser_factory_dict = {
    "ORIGINAL": PYTHON_PARSER,

    # Output Style Prompts
    "JSON_STYLE": JSON_PARSER,
    "XML_STYLE": XML_PARSER,
    "CSV_STYLE": CSV_PARSER,
    "YAML_STYLE": YAML_PARSER,
    # Special Output Style
    "ChatGPT_JSON_PARAM": JSON_PARSER,

    # Peturbations
    "SPACE_BEFORE_PB": PYTHON_PARSER,
    "SPACE_AFTER_PB": PYTHON_PARSER,
    "HELLO_PB": PYTHON_PARSER,
    "HELLO!_PB": PYTHON_PARSER,
    "HOWDY_PB": PYTHON_PARSER,
    "THANK_YOU_PB": PYTHON_PARSER,
    # Special Peturbations
    "STATEMENT_REPHRASE": PYTHON_PARSER,

    # Jailbreaks
    "AIM_JB": JB_PARSER,
    "DAN_JB": JB_PARSER,
    "DEV_JB": JB_PARSER,
    "EVIL_JB": JB_PARSER,
    "REFUSAL_JB": JB_PARSER,
    # Special Jailbreaks
    "DAN_JB_CLASSIC": JB_PARSER,
    "DEV_JB_CLASSIC": JB_PARSER,
    "DAN_JB_JAILBREAK": JB_PARSER,
    "DEV_JB_JAILBREAK": JB_PARSER
}

for k, v in parser_factory_dict.items():
    parser_factory.register_builder(k, v)

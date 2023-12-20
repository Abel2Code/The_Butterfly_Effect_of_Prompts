INJECTION_POSITION_STRING = "[>>INJECT HERE<<]"

def verify_format(func):
    def verification_func(prompt):
        assert f" {INJECTION_POSITION_STRING} " in prompt
        return func(prompt)
         
    return verification_func

# Style Injectors
@verify_format
def clear_inject(prompt):
    return prompt.replace(f" {INJECTION_POSITION_STRING} ", " ")

@verify_format
def python_list_inject(prompt):
    return prompt.replace(INJECTION_POSITION_STRING, "Write your answer in the form of a Python list containing the appropriate attribute.")

@verify_format
def json_inject(prompt):
    return prompt.replace(INJECTION_POSITION_STRING, "Write your answer in JSON format containing the appropriate attribute.")

@verify_format
def xml_inject(prompt):
    return prompt.replace(INJECTION_POSITION_STRING, "Write your answer in XML format containing the appropriate attribute.")

@verify_format
def csv_inject(prompt):
    return prompt.replace(INJECTION_POSITION_STRING, "Write your answer in CSV format containing the appropriate attribute.")

@verify_format
def yaml_inject(prompt):
    return prompt.replace(INJECTION_POSITION_STRING, "Write your answer in YAML format containing the appropriate attribute.")

# Misc Injections
@verify_format
def thank_you_inject(prompt):
    prompt = prompt.replace(INJECTION_POSITION_STRING, INJECTION_POSITION_STRING + " Thank you.")
    return python_list_inject(prompt)

@verify_format
def coT_inject(prompt):
    prompt = prompt.replace(INJECTION_POSITION_STRING, INJECTION_POSITION_STRING + " Let's think step by step.")
    return python_list_inject(prompt)


@verify_format
def wont_tip_inject(prompt):
    prompt = prompt.replace(INJECTION_POSITION_STRING, INJECTION_POSITION_STRING + " I won't tip by the way.")
    return python_list_inject(prompt)

def tip_x_injector(x):
    @verify_format
    def tip_inject(prompt):
        prompt = prompt.replace(INJECTION_POSITION_STRING, INJECTION_POSITION_STRING + f" I'm going to tip ${x} for a perfect response!")
        return python_list_inject(prompt)

    return tip_inject
        
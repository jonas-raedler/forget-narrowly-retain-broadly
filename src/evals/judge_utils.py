import numpy as np


def check_yes_no(input_str:str):
    """
    Returns 1 for a yes-variant, 0 for a no-variant; raises ValueError for
    empty input or anything unrecognized.
    """
    yes_variants = {"yes", "yeah"}
    no_variants = {"no", "nope", "not sure"}

    normalized_input = input_str.strip().lower()

    if not normalized_input:
        raise ValueError("Input cannot be empty or just whitespace.")

    if normalized_input in yes_variants:
        return 1
    if normalized_input in no_variants:
        return 0
    else:
        raise ValueError("Input not recognized as 'yes' or 'no'.")

def average_case_acc(data:dict, prefix:str="outs_"):
    relevant_keys = [k for k in data if k.startswith(prefix)]
    data = [data[key] for key in relevant_keys if key in data]

    if isinstance(data[0], list):
        # Mean across the judge repeats (axis=0), then across the questions.
        arr = np.array(data)
        mean_across = np.mean(arr, axis=0)
        final_mean = np.mean(mean_across)
    else:
        arr = np.array(data)
        final_mean = np.mean(arr, axis=0)

    return final_mean




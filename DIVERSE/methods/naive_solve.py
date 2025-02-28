import itertools
import re
import numpy as np
from functools import partial
from model.gpt.gpt_utils import gpt



def get_answer (task, 
                x, 
                generate_method, 
                prompt_used,
                number_generate_sample,
                language):
    """
    task - task object
    x - input
    n_generate_sample - number of samples to be generated
    prompt_sample 
    stop - stopping condition 
    """

    if generate_method == 'standard':
        list_of_prompt = [task.standard_prompt_wrap(task, x=x)]
    elif generate_method == 'cot':
        list_of_prompt = task.cot_prompt_wrap(task, x=x, prompt=prompt_used)
    else:
        raise ValueError(f'prompt_sample {generate_method} not recognized')
    
    # Generate samples using Gemma
    list_of_answer = []
    for curr_prompt in list_of_prompt:
        curr_answer = gpt(prompt=curr_prompt, max_tokens=1000, n=number_generate_sample)
        list_of_answer.append(curr_answer)

    return list_of_answer



def naive_solve(args, task, idx, to_print=True):
    """
    A simple solver using GPT to generate responses.

    Args:
        args - argument object containing method configurations
        task - the problem-solving task instance
        idx - index of the input in the dataset
        to_print - whether to print the results

    Returns:
        Generated response and metadata
    """
    # Get the input for the given index
    x = task.get_input(idx)

    # Generate samples using GPT
    ys = get_answer(task, 
                    x, 
                    args.generate_method,
                    args.prompt_used,
                    args.number_generate_sample, 
                    language=args.lang)

    if to_print:
        print(f"Generated response: {ys}")

    return ys
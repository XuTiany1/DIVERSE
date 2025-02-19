import itertools
import re
import numpy as np
from functools import partial
from model.gpt import gpt 



def get_answer (task, 
                x, 
                n_generate_sample, 
                generate_method, 
                language):
    """
    task - task object
    x - input
    n_generate_sample - number of samples to be generated
    prompt_sample 
    stop - stopping condition 
    """

    if generate_method == 'standard':
        prompt = task.standard_prompt_wrap(x=x, lang=language)
    elif generate_method == 'cot':
        prompt = task.cot_prompt_wrap(task, x=x, lang=language)
    else:
        raise ValueError(f'prompt_sample {generate_method} not recognized')
    
    # Generate samples using Gemma
    answer = gpt(prompt=prompt, max_tokens=500)

    return answer



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
                    args.n_generate_sample, 
                    args.generate_method, 
                    language=args.lang)

    if to_print:
        print(f"Generated response: {ys}")

    return ys, {}
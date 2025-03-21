import itertools
import re
import numpy as np
from functools import partial
from model.gpt.gpt_utils import gpt
import asyncio
from playground.playground import generate_text_async  # Import the async GPT call



def get_answer (task, 
                x, 
                generate_method, 
                prompt_used,
                number_generate_sample,
                language,
                model):
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
        curr_answer = gpt(prompt=curr_prompt, max_tokens=1000, n=number_generate_sample, model=model)
        list_of_answer.append(curr_answer)

    return list_of_answer


# generate_dataset.py
async def generate_dataset(task, 
                           question, 
                           prompt_list, 
                           model):
    prompts = task.prompt_wrap(task, question=question, list_of_prompt=prompt_list)

    tasks = []
    for prompt in prompts:
        tasks.append(asyncio.create_task(
            generate_text_async(
                messages=[{"role": "user", "content": prompt}],
                model=model,
                temperature=0.7,
                max_tokens=1000,
                n=1
            )
        ))

    return await asyncio.gather(*tasks)


# naive_solver.py
def naive_solve(args, task, idx, to_print=True):
    x = task.get_input(idx)

    ys = asyncio.run(
        generate_dataset(task, x, args.prompt_used, model=args.generator_model)
    )

    if to_print:
        for i, resp in enumerate(ys):
            print(f"Prompt {i} -> Response: {resp}")

    return ys
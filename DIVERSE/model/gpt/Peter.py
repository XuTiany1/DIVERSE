import asyncio
from typing import Dict, List
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI, AsyncOpenAI
import tiktoken
import os
import time
# import weave
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
openai_async_clint = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
CHAT_MODELS = [
    "gpt-3.5-turbo-1106",
    "gpt-3.5-turbo-0125",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4-turbo-2024-04-09",
    "o1-mini-2024-09-12",
]
REASONING_MODELS = [
    "o3-mini-2025-01-31",
]
ENCODER = {
    k: tiktoken.encoding_for_model(k)
    for k in CHAT_MODELS + REASONING_MODELS
}
MAX_TOKENS = {
    # GPT-3.5 Turbo models
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo-0125": 16385,
    # GPT-4o and mini models
    "gpt-4o-mini-2024-07-18": 128000,
    "gpt-4o-2024-08-06": 128000,
    "gpt-4o-2024-11-20": 128000,
    # GPT-4 Turbo models
    "gpt-4-turbo-2024-04-09": 128000,
    # o1 and o3 models
    "o1-mini-2024-09-12": 128000,
    "o3-mini-2025-01-31": 200000,
}
def construct_messages(model, **kwargs):
    """Construct messages for OpenAI API"""
    messages = []
    for role, content in kwargs.items():
        messages.append({"role": role, "content": content[:int(MAX_TOKENS[model]*4.5)]})
    return messages
def construct_messages_with_system(model, system, **kwargs):
    """Construct messages for OpenAI API with system prompt"""
    messages = [{"role": "system", "content": system}]
    for role, content in kwargs.items():
        messages.append({"role": role, "content": content[:int(MAX_TOKENS[model]*4.5)]})
    return messages
def trim_text(text, model="gpt-3.5-turbo"):
    """Trim text to fit within max tokens"""
    tokens = len(ENCODER[model].encode(text))
    if tokens > MAX_TOKENS[model]:
        first_half = text[:MAX_TOKENS[model]//2]
        second_half = text[-MAX_TOKENS[model]//2:]
        return ENCODER[model].decode(ENCODER[model].encode(first_half) + ENCODER[model].encode(second_half))
    return text
# https://cookbook.openai.com/examples/how_to_count_tokens_with_tiktoken
def num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using o200k_base encoding.")
        encoding = tiktoken.get_encoding("o200k_base")
    if model in {
        "gpt-3.5-turbo-0125",
        "gpt-4-0314",
        "gpt-4-32k-0314",
        "gpt-4-0613",
        "gpt-4-32k-0613",
        "gpt-4o-mini-2024-07-18",
        "gpt-4o-2024-08-06"
    }:
        tokens_per_message = 3
        tokens_per_name = 1
    elif "gpt-3.5-turbo" in model:
        print("Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0125.")
        return num_tokens_from_messages(messages, model="gpt-3.5-turbo-0125")
    elif "gpt-4o-mini" in model:
        print("Warning: gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-mini-2024-07-18.")
        return num_tokens_from_messages(messages, model="gpt-4o-mini-2024-07-18")
    elif "gpt-4o" in model:
        print("Warning: gpt-4o and gpt-4o-mini may update over time. Returning num tokens assuming gpt-4o-2024-08-06.")
        return num_tokens_from_messages(messages, model="gpt-4o-2024-08-06")
    elif "gpt-4" in model:
        print("Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613.")
        return num_tokens_from_messages(messages, model="gpt-4-0613")
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}."""
        )
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens
# @weave.op()
def generate_text(
    messages,
    model="gpt-3.5-turbo",
    model_parameters={"temperature": 0.0},
    delay=0.2,
    return_response=False,
    raise_error=False,
):
    """Process a single prompt using OpenAI API with caching"""
    # Check if result exists in cache
    try:
        response = openai_client.chat.completions.create(
            model=model, messages=messages, **model_parameters
        )
        time.sleep(delay)
        if return_response:
            return response
        return response.choices[0].message.content
    except Exception as e:
        if raise_error:
            raise e
        return f"Error processing prompt: {e}"
from functools import partial
from tqdm.contrib.concurrent import process_map
def generate_text_batch(
    messages_list,
    model="gpt-3.5-turbo",
    model_parameters={"temperature": 0.0},
    delay=0.2,
    return_response=False,
    raise_error=False,
    num_workers=4,
) -> List[str | Dict]:
    """Process a batch of prompts using OpenAI API with caching"""
    results = process_map(
        partial(
            generate_text,
            model=model,
            model_parameters=model_parameters,
            delay=delay,
            return_response=return_response,
            raise_error=raise_error,
        ),
        messages_list,
        max_workers=num_workers,
        # chunksize=1,
    )
    return results
def generate_text_batch_df(
    messages_list,
    model="gpt-3.5-turbo",
    model_parameters={"temperature": 0.0},
    delay=0.2,
    return_response=False,
    raise_error=False,
    num_workers=4,
) -> List[str | Dict]:
    """Process a batch of prompts using OpenAI API with caching"""
    import pandas as pd
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=False)
    series = pd.Series(messages_list, name="messages")
    print(series.head())
    return series.parallel_apply(
        partial(
            generate_text,
            model=model,
            model_parameters=model_parameters,
            delay=delay,
            return_response=return_response,
            raise_error=raise_error
        )
    ).tolist()
async def generate_text_async(
    messages,
    model="gpt-3.5-turbo",
    model_parameters={"temperature": 0.0},
    delay=0.2,
    return_response=False,
    raise_error=False,
):
    """Async version of generate_text"""
    try:
        response = await openai_async_clint.chat.completions.create(
            model=model, messages=messages, **model_parameters
        )
        await asyncio.sleep(delay)
        if return_response:
            return response
        return response.choices[0].message.content
    except Exception as e:
        if raise_error:
            raise e
        return f"Error processing prompt: {e}"
async def generate_text_batch_async(
    messages_list,
    model="gpt-3.5-turbo",
    model_parameters={"temperature": 0.0},
    delay=0.2,
    return_response=False,
    raise_error=False,
    num_workers=4,
) -> List[str | Dict]:
    """Process a batch of prompts using OpenAI API with caching asynchronously"""
    import asyncio
    from tqdm.asyncio import tqdm
    # Create semaphore to limit concurrent API calls
    semaphore = asyncio.Semaphore(num_workers)
    async def process_with_semaphore(messages):
        async with semaphore:
            return await generate_text_async(
                messages,
                model=model,
                model_parameters=model_parameters,
                delay=delay,
                return_response=return_response,
                raise_error=raise_error
            )
    # Create tasks for all messages
    tasks = [process_with_semaphore(messages) for messages in messages_list]
    # return await asyncio.gather(*tasks)
    return await tqdm.gather(*tasks)
# The rest of your code remains the same
def get_models():
    """Get available models from OpenAI API"""
    # response =
    for model in openai_client.models.list():
        print(model)
    return openai_client.models
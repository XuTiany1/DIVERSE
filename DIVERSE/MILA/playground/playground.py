import os
import asyncio
import backoff
from openai import AsyncOpenAI
from typing import List, Dict
from tqdm.asyncio import tqdm

# Load OpenAI API key
api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    print("Warning: OPENAI_API_KEY is not set")

# Initialize async OpenAI client
openai_async_client = AsyncOpenAI(api_key=api_key)

# Global token tracking
completion_tokens = prompt_tokens = 0

# Backoff retry for API calls
@backoff.on_exception(backoff.expo, Exception, max_tries=5)
async def async_completion_call(**kwargs):
    """Call OpenAI API asynchronously with backoff."""
    # Debug: before API call
    print("DEBUG: Calling OpenAI API with kwargs:", kwargs)
    response = await openai_async_client.chat.completions.create(**kwargs)
    # Debug: after API call
    print("DEBUG: Received API response. Usage:", response.usage)
    return response

# Generate response from model asynchronously (single prompt)
async def generate_text_async(
    messages: List[Dict[str, str]],
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
    n=1,
    stop=None
) -> List[str]:
    """Process a single prompt asynchronously."""
    global completion_tokens, prompt_tokens
    try:
        response = await async_completion_call(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=n,
            stop=stop
        )
        completion_tokens += response.usage.completion_tokens
        prompt_tokens += response.usage.prompt_tokens
        # Debug: show token usage for this call
        print(f"DEBUG: Tokens - Prompt: {response.usage.prompt_tokens}, Completion: {response.usage.completion_tokens}")
        return [choice.message.content for choice in response.choices]
    except Exception as e:
        print("DEBUG: Exception in generate_text_async:", e)
        return [f"Error: {e}"]

# Process a batch of prompts asynchronously with limited concurrency
async def generate_text_batch_async(
    messages_list: List[List[Dict[str, str]]],
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000,
    n=1,
    stop=None,
    num_workers=8
) -> List[List[str]]:
    """Process a batch of prompts asynchronously with limited concurrency."""
    semaphore = asyncio.Semaphore(num_workers)

    async def process_with_semaphore(messages):
        async with semaphore:
            # Debug: entering semaphore-protected call
            print("DEBUG: Processing messages with semaphore:", messages)
            result = await generate_text_async(messages, model, temperature, max_tokens, n, stop)
            # Debug: exiting semaphore-protected call
            print("DEBUG: Processed messages. Result:", result)
            return result

    tasks = [process_with_semaphore(messages) for messages in messages_list]
    # Debug: starting batch processing with tqdm progress
    print("DEBUG: Starting batch processing for", len(tasks), "tasks.")
    return await tqdm.gather(*tasks)

def get_usage(backend="gpt-4o-mini"):
    """Return token usage and estimated cost."""
    global completion_tokens, prompt_tokens
    cost_per_token = {
        "gpt-4o-mini": (0.005, 0.0025),
        "gpt-4o": (0.005, 0.0025),
        "gpt-4": (0.06, 0.03),
        "gpt-3.5-turbo": (0.002, 0.0015)
    }
    comp_cost, prompt_cost = cost_per_token.get(backend, (0, 0))
    cost = (completion_tokens / 1000 * comp_cost) + (prompt_tokens / 1000 * prompt_cost)
    return {
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "cost": cost
    }

# Playground: main function to test our async functions
async def main():
    # --- Single Prompt Test ---
    single_message = [{"role": "user", "content": "What is the capital of France?"}]
    print("\nDEBUG: Testing generate_text_async (single prompt)")
    single_response = await generate_text_async(single_message)
    print("DEBUG: Response from generate_text_async:", single_response)

    # --- Batch Processing Test ---
    messages_list = [
        [{"role": "user", "content": "Explain the theory of relativity in simple terms."}],
        [{"role": "user", "content": "What is the significance of the number pi?"}],
        [{"role": "user", "content": "Define quantum entanglement."}],
    ]
    print("\nDEBUG: Testing generate_text_batch_async (batch processing)")
    batch_responses = await generate_text_batch_async(messages_list, num_workers=2)
    print("DEBUG: Responses from generate_text_batch_async:")
    for idx, res in enumerate(batch_responses):
        print(f"Response {idx + 1}: {res}")

    # --- Print Token Usage and Cost ---
    usage_info = get_usage("gpt-4o-mini")
    print("\nDEBUG: Final Token Usage and Cost:", usage_info)

if __name__ == "__main__":
    # Run the playground example in async mode
    asyncio.run(main())

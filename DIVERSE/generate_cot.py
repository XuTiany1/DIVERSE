import argparse
import os
from datetime import datetime
from methods.naive_solve import naive_solve
from tasks.AFRI_MGSM import AfriMgsmTask
import re
import sys
import evaluate
from model.deberta_v3.deberta_verifier import Verifier
import json  # We'll use JSON to dump our records

# Define argparse namespace
args = argparse.Namespace( 
    task='MGSM', 
    lang='en',
    naive_run=False, 
    generate_method='cot', 
    number_generate_sample=1, 
    prompt_used=[0,1,2,3,4],
    selection_method='voting',
    checkpoint_path='/home/mila/x/xut/github/DIVERSE/DIVERSE/model/deberta_v3/checkpoint-6565',
    tokenizer_name='microsoft/deberta-v3-large',
    generator_model='gpt-4o'
)

# Define test range
num_samples = 250
correct_count = 0

# Load verifier only if using chain-of-thought generation
if args.generate_method == "cot":
    verifier = Verifier(args)

# de is incomplete, chinese has some repeted stuff (around 5-6 at the start)
# languages = ['es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn']
# CHINESE IS ON 227
# SWAHILI REPEATED UP AND INCLUDING LINE 7

# languages = ['de'], 204
# chinese not sure where
# languages = ['zh']

# 'amh', 'hao
languages = ['kin', 'lin', 'lug', 'orm', 'sna', 'sot', 'swa', 'twi', 'vai', 'wol', 'xho', 'yor', 'zul']
prompts_to_use = [[0,1,2,3,4]]

for lang in languages:
    for pr in prompts_to_use:
        total_count = 0
        correct_count = 0
        args.lang = lang
        args.prompt_used = pr

        # Create your MGSM task instance
        task = AfriMgsmTask(args)

        # Create a run folder (using a timestamp) under logs/MGSM/{lang}/{generator_model}/debug/{len(prompt_used)}
        run_folder = os.path.join("logs", "AFRI_MGSM-cot-dataset", args.lang, "koko", str(len(pr)))
        os.makedirs(run_folder, exist_ok=True)

        # List to log questions that cause errors.
        error_questions = []
        
        # Prepare the hyperparameter log file
        hyperparams_path = os.path.join(run_folder, "hyperparams.log")
        with open(hyperparams_path, "w") as hp:
            hp.write("--- Hyperparameters / Args ---\n")
            for k, v in vars(args).items():
                hp.write(f"{k}: {v}\n")

        # Instead of accumulating all records in memory, open a JSON Lines file in append mode.
        json_filename = os.path.join(run_folder, "chain_of_thought_records.jsonl")
        os.makedirs(os.path.dirname(json_filename), exist_ok=True)

        remove_count = 0

        # Main loop over test instances
        for idx in range(1, num_samples):
            proper_idx = idx - remove_count

            print(f"\n--- Running Test {idx} ---")

            # Run the solver to get the model output and questions.
            model_output = naive_solve(args, task, idx, to_print=False)
            model_question = task.get_input(idx)
            english_question = task.get_english_input(proper_idx)


            try:
                # Ground truth
                ground_truth_answer = task.ground_truth_answer(idx)
                # Convert to Python int in case it's a numpy.int64
                ground_truth_answer = int(ground_truth_answer)
            except Exception as e:
                print(f"Error converting ground truth for test {idx}: {e}")
                error_questions.append(english_question)

            # Skipping over empty questions
            if model_question == '"':
                num_samples = num_samples + 1
                remove_count = remove_count + 1
                continue

            # Compute a dictionary mapping each reasoning path to its verifier probability.
            reasoning_path_probability = task.compute_probability(task, model_output, english_question, verifier)

            # Build a record for this test instance.
            record = {
                "question": english_question,
                "ground_truth": ground_truth_answer,
                "reasoning_paths": []
            }
            for reasoning_path, probability in reasoning_path_probability.items():
                record["reasoning_paths"].append({
                    "chain": reasoning_path,
                    "probability": probability
                })

            # Append the record to the JSON Lines file.
            with open(json_filename, "a") as fp:
                fp.write(json.dumps(record) + "\n")

            print(f"Recorded instance {idx}")

        # After processing all instances for this language-prompt experiment,
        # log the questions that caused errors.
        if error_questions:
            error_log_path = os.path.join(run_folder, "error_questions.log")
            with open(error_log_path, "w") as ef:
                for q in error_questions:
                    ef.write(q + "\n")
            print(f"Error log saved to {error_log_path}")



        print(f"Chain of thought records saved to {json_filename}")

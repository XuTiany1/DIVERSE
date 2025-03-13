import argparse
import os
from datetime import datetime
from methods.naive_solve import naive_solve
from tasks.MGSM import MgsmTask
import re
import sys
import evaluate
from model.deberta_v3.deberta_verifier import Verifier
import json


# Define argparse namespace
args = argparse.Namespace( 
    task='MGSM', 
    prompt_used=[0,1,2,3,4],
    selection_method = 'voting',
    lang = 'en'
)

# Define test range
num_samples = 250
correct_count = 0

# Load evaluation metrics
exact_match_metric = evaluate.load("exact_match")

# You can test multiple languages here if you want
# languages = ['es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn']
languages = ['es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn']

# prompts_to_use = [[0,1,2,3,4], [0,1,2,3], [0,1,2], [0,1]]
prompts_to_use = [[0,1,2,3,4], [0,1,2,3], [0,1,2], [0,1]]

for lang in languages:
    for pr in prompts_to_use:
        total_count = 0
        correct_count = 0

        # Create your MGSM task
        task = MgsmTask(args)


        run_folder = os.path.join("logs", "MGSM-eval", args.lang, "evalaution", str(len(pr)))
        os.makedirs(run_folder, exist_ok=True)

        evaluation_path = os.path.join(run_folder, args.selection_method, "evaluation.log")
        os.makedirs(os.path.dirname(evaluation_path), exist_ok=True)


        # Load JSON Data (Ensure your JSON file path is correct)
        jsonl_file_path = "/home/mila/x/xut/github/DIVERSE/logs/MGSM/te/gpt-4o/debug/5/chain_of_thought_records.jsonl"

        json_data = []
        with open(jsonl_file_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():  # Ignore empty lines
                    json_data.append(json.loads(line))  # Load each line as a JSON object


        # 3. Write hyperparameters
        with open(evaluation_path, "w") as hp:
            hp.write("--- Evaluation Detail ---\n")
            for k, v in vars(args).items():
                hp.write(f"{k}: {v}\n")
            hp.write("-------------------------\n")


        # Lists to collect predictions and ground truths for metric computation
        all_predictions = []
        all_references = []

        # 5. Main loop
        for idx in range(min(num_samples, len(json_data))):  # Ensure we don't exceed data size
            entry = json_data[idx]

            # Extract model outputs and probabilities
            reasoning_paths = entry.get("reasoning_paths", [])
            ground_truth = entry.get("ground_truth")

            extracted_answers = []
            answer_probabilities = {}

            for path in reasoning_paths:
                response_text = path.get("chain", "")
                probability = path.get("probability", 0.0)

                extracted_answer = task.extract_final_answer(task, response_text)
                if extracted_answer:
                    extracted_answers.append((extracted_answer, probability))
                    answer_probabilities[extracted_answer] = probability
        
            # Log the results
            with open(evaluation_path, "a") as log_file:
                log_file.write(f"\nEntry {idx+1}:\n")
                log_file.write(f"Question: {entry['question']}\n")
                log_file.write(f"Ground Truth: {ground_truth}\n")
                log_file.write(f"Extracted Answers with Probabilities:\n")
                for ans, prob in extracted_answers:
                    log_file.write(f"  - Answer: {ans}, Probability: {prob:.4f}\n")
                log_file.write("-" * 40 + "\n")
                
            # Store results for evaluation
            if extracted_answers:
                best_answer = max(extracted_answers, key=lambda x: x[1])[0]  # Select highest probability answer
                all_predictions.append(best_answer)
                all_references.append(ground_truth)

        # Compute Exact Match Score
        eval_result = exact_match_metric.compute(predictions=all_predictions, references=all_references)

        # Log final evaluation results
        with open(evaluation_path, "a") as log_file:
            log_file.write("\nFinal Evaluation Metrics:\n")
            log_file.write(f"Exact Match Score: {eval_result['exact_match']:.4f}\n")

print("Evaluation completed. Check logs for details.")
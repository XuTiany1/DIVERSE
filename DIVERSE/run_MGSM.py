import argparse
import os
from datetime import datetime
from methods.naive_solve import naive_solve
from tasks.MGSM import MgsmTask
import re
import sys
import os
from model.deberta_v3.deberta_verifier import Verifier
import evaluate

# Define argparse namespace
args = argparse.Namespace( 
    task='MGSM', 
    lang='en',
    naive_run=False, 
    generate_method='standard', 
    #generate_method='cot', 
    number_generate_sample=1, 
    prompt_used=[1],
    #prompt_used=[1,2,3,4,5],
    checkpoint_path='/home/mila/x/xut/github/DIVERSE/DIVERSE/model/mderberta/checkpoint-6565',
    tokenizer_name='microsoft/mdeberta-v3-base'
    # checkpoint_path='/home/mila/x/xut/github/DIVERSE/DIVERSE/model/deberta_v3/checkpoint-6565',
    # tokenizer_name='microsoft/deberta-v3-large'
)

# Define test range
num_samples = 249
correct_count = 0

# Create a log directory if it doesn’t exist
log_dir = f"logs/MGSM/{args.lang}"
os.makedirs(log_dir, exist_ok=True)


# Load evaluation metrics
exact_match_metric = evaluate.load("exact_match")
f1_metric = evaluate.load("f1")

# Load verifier
if args.generate_method == "cot":
    verifier = Verifier(args)


#languages = ['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te']
languages = ['en']

for lang in languages:
    total_count = 0

    # Lists to collect predictions and ground truths
    all_predictions = []
    all_references = []

    # Reset count
    correct_count = 0

    args.lang = lang

    # Create task instance
    task = MgsmTask(args)

    # Create a log directory if it doesn’t exist
    log_dir = f"logs/MGSM/{args.lang}"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"Debug_testing_two")

    # Run test loop
    with open(log_file, "w") as f:
        f.write(f"--- TEST LOG START ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n")

    for idx in range(1, num_samples):
        print(f"\n--- Running Test {idx} ---")

        # Run model
        model_output = naive_solve(args, task, idx, to_print=False)
        model_question = task.get_input(idx)

        if args.generate_method == "standard":
            answer = model_output[0][0]
            final_answer = re.search(r"\d+(\.\d+)?", answer).group()  # Find number (integer or decimal)

        if args.generate_method == "cot":
            final_answer, weighted_probs, raw_probability = task.compute_final_answer(task, model_output, model_question, verifier)
            print(f"weighted_probs{weighted_probs}")

            # Compute weighted probability across all answers
            total_prob = sum(weighted_probs.values())
            if total_prob > 0:
                weighted_probs = {k: v / total_prob for k, v in weighted_probs.items()}
            else:
                weighted_probs = {k: 0 for k in weighted_probs.keys()}  # Avoid division by zero

            correct_answer_prob = weighted_probs.get(str(ground_truth_answer), 0)

            # Format raw probabilities nicely
            formatted_raw_prob = "; ".join(
                f"{ans}: [{', '.join(f'{p:.2%}' for p in probs)}]"
                for ans, probs in raw_probability.items()
            )


        # Extract ground truth and model answer
        ground_truth_answer = task.ground_truth_answer(idx)


        total_count += 1
        # Determine correctness
        if float(final_answer) == float(ground_truth_answer):
            correct_count += 1
        accuracy = correct_count / total_count

        # Save predictions and references for overall metric computation
        all_predictions.append(final_answer)
        all_references.append(str(ground_truth_answer))

        # Construct log entry
        log_entry = (
            "----------------------\n"
            f"Problem {idx}: {task.get_input(idx)}\n"
            f"Model Prediction / Ground Truth: {final_answer} / {ground_truth_answer}\n"
            f"Correct Predictions / Total Tests: {correct_count} / {total_count}\n"
            # f"Weighted Verifier Probability: {correct_answer_prob:.2%}\n"  
            # f"Verifier Raw Probability: {formatted_raw_prob}\n"
            f"Current Accuracy: {accuracy:.2%}\n"
            "----------------------\n"
        )

        # Print to console
        print(log_entry)

        # Append log entry to log file
        with open(log_file, "a") as f:
            f.write(log_entry)

    # Compute metrics over all predictions and references
    results_exact = exact_match_metric.compute(predictions=all_predictions, references=all_references)

    print("Overall Exact Match:", results_exact)
    overall_accuracy = sum(1 for p, r in zip(all_predictions, all_references) if float(p) == float(r)) / len(all_predictions)
    print("Overall Accuracy:", overall_accuracy)


    # Final summary
    final_summary = (
        "\n--- FINAL TEST SUMMARY ---\n"
        f"Total Samples Tested: {num_samples}\n"
        f"Correct Predictions: {correct_count}\n"
        f"Final Accuracy: {accuracy:.2%}\n"
        f"Overall Exact Match Score: {results_exact}\n"
        "----------------------\n"
    )
    print(final_summary)

    # Save final summary
    with open(log_file, "a") as f:
        f.write(final_summary)

    print(f"Test log saved to {log_file}")
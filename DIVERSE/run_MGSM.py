import argparse
import os
from datetime import datetime
from methods.naive_solve import naive_solve
from tasks.MGSM import MgsmTask
import re
import sys
import os
from model.deberta_v3.deberta_verifier import Verifier

# Define argparse namespace
args = argparse.Namespace( 
    task='MGSM', 
    lang='en',
    naive_run=False, 
    generate_method='cot', 
    number_generate_sample=1, 
    prompt_used=[1,2,3,4,5],
    checkpoint_path='/home/mila/x/xut/github/DIVERSE/DIVERSE/model/deberta_v3/checkpoint-6565',
    tokenizer_name='microsoft/deberta-v3-large'
)

# Define test range
num_samples = 249  
correct_count = 0

# Create a log directory if it doesn’t exist
log_dir = f"logs/MGSM/{args.lang}"
os.makedirs(log_dir, exist_ok=True)


# Load verifier
verifier = Verifier(args)


languages = ['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te']

for lang in languages:

    # Reset count
    correct_count = 0

    args.lang = lang

    # Create task instance
    task = MgsmTask(args)

    # Create a log directory if it doesn’t exist
    log_dir = f"logs/MGSM/{args.lang}"
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, f"cot_result")

    # Run test loop
    with open(log_file, "w") as f:
        f.write(f"--- TEST LOG START ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n")

    for idx in range(1, num_samples):
        print(f"\n--- Running Test {idx} ---")

        # Run model
        # ys, infos, final_answers, model_output = solve(args, task, idx, to_print=False)
        model_output = naive_solve(args, task, idx, to_print=False)

        model_question = task.get_input(idx)
        
        final_answer, weighted_probs = task.compute_final_answer(task, model_output, model_question, verifier)

        # Extract ground truth and model answer
        ground_truth_answer = task.ground_truth_answer(idx)

        # Determine correctness
        is_correct = (int(final_answer) == ground_truth_answer)
        correct_count += int(is_correct)

        # Compute accuracy
        accuracy = correct_count / idx

        weighted_probs = [float(w) for w in weighted_probs]  # Convert all to floats

        # Construct log entry
        log_entry = (
            "----------------------\n"
            f"Problem {idx}: {task.get_input(idx)}\n"
            f"Model Prediction / Ground Truth: {final_answer} / {ground_truth_answer}\n"
            f"Correct Predictions / Total Tests: {correct_count} / {idx}\n"
            f"Verifier Probability: {max(weighted_probs):.2%}\n"
            f"Current Accuracy: {accuracy:.2%}\n"
            "----------------------\n"
        )

        # Print to console
        print(log_entry)

        # Append log entry to log file
        with open(log_file, "a") as f:
            f.write(log_entry)

    # Final summary
    final_summary = (
        "\n--- FINAL TEST SUMMARY ---\n"
        f"Total Samples Tested: {num_samples}\n"
        f"Correct Predictions: {correct_count}\n"
        f"Final Accuracy: {accuracy:.2%}\n"
        "----------------------\n"
    )
    print(final_summary)

    # Save final summary
    with open(log_file, "a") as f:
        f.write(final_summary)

    print(f"Test log saved to {log_file}")
import argparse
import os
from datetime import datetime
from methods.naive_solve import naive_solve
from tasks.MGSM import MgsmTask
import re

# Define argparse namespace
args = argparse.Namespace( 
    task='MGSM', 
    lang='en',
    naive_run=False, 
    generate_method='standard', 
    n_generate_sample=2, 
)

# Define test range
num_samples = 249  
correct_count = 0

# Create a log directory if it doesn’t exist
log_dir = f"logs/MGSM/{args.lang}"
os.makedirs(log_dir, exist_ok=True)

#languages = ['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te']

languages = ['es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te']

for lang in languages:

    # Reset count
    correct_count = 0

    args.lang = lang
    
    # Create task instance
    task = MgsmTask(args)

    # Create a log directory if it doesn’t exist
    log_dir = f"logs/MGSM/{args.lang}"
    os.makedirs(log_dir, exist_ok=True)

    # log_file = os.path.join(log_dir, f"7 steps --50 -- generate: {args.n_generate_sample} -- select:{args.n_select_sample}")
    log_file = os.path.join(log_dir, f"naive_result")


    # Run test loop
    with open(log_file, "w") as f:
        f.write(f"--- TEST LOG START ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n")

    for idx in range(1, num_samples):
        print(f"\n--- Running Test {idx} ---")

        # Run model
        # ys, infos, final_answers, model_output = solve(args, task, idx, to_print=False)
        model_output, infos = naive_solve(args, task, idx, to_print=False)

        # Extract ground truth and model answer
        ground_truth_answer = task.ground_truth_answer(idx)
        model_answer = task.model_answer(model_output)

        # Determine correctness
        is_correct = (model_answer == ground_truth_answer)
        correct_count += int(is_correct)

        # Compute accuracy
        accuracy = correct_count / idx

        # Construct log entry
        log_entry = (
            "----------------------\n"
            f"Problem {idx}: {task.get_input(idx)}\n"
            f"Model Prediction / Ground Truth: {model_answer} / {ground_truth_answer}\n"
            f"Correct Predictions / Total Tests: {correct_count} / {idx}\n"
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
import argparse
import os
from datetime import datetime
from methods.naive_solve import naive_solve
from tasks.MGSM import MgsmTask
import re
import sys
import evaluate
from model.deberta_v3.deberta_verifier import Verifier

# Define argparse namespace
args = argparse.Namespace( 
    task='MGSM', 
    lang='en',
    naive_run=False, 
    generate_method='cot', 
    number_generate_sample=1, 
    prompt_used=[0],
    checkpoint_path='/home/mila/x/xut/github/DIVERSE/DIVERSE/model/deberta_v3/checkpoint-6565',
    tokenizer_name='microsoft/deberta-v3-large',
    generator_model='gpt-4o-mini'
)

# Define test range
num_samples = 249
correct_count = 0

# Load evaluation metrics
exact_match_metric = evaluate.load("exact_match")

# Load verifier only if using chain of thought
if args.generate_method == "cot":
    verifier = Verifier(args)

# You can test multiple languages here if you want
languages = ['en']

for lang in languages:
    total_count = 0
    correct_count = 0
    args.lang = lang

    # Create your MGSM task
    task = MgsmTask(args)

    # 1. Create a time-stamped subfolder for this run under logs/MGSM/{args.lang}
    run_folder = os.path.join("logs", "MGSM", args.lang, args.generator_model, "Beluga")
    os.makedirs(run_folder, exist_ok=True)

    # 2. Prepare your three log files
    hyperparams_path = os.path.join(run_folder, "hyperparams.log")
    overall_log_path = os.path.join(run_folder, "overall_experiment.log")
    error_log_path   = os.path.join(run_folder, "error.log")

    # 3. Write hyperparameters
    with open(hyperparams_path, "w") as hp:
        hp.write("--- Hyperparameters / Args ---\n")
        for k, v in vars(args).items():
            hp.write(f"{k}: {v}\n")

    # Lists to collect predictions and ground truths for metric computation
    all_predictions = []
    all_references = []

    # 4. Write a short header to overall_experiment.log
    with open(overall_log_path, "w") as f:
        f.write(f"--- TEST LOG START ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) ---\n\n")

    # 5. Main loop
    for idx in range(1, num_samples):
        print(f"\n--- Running Test {idx} ---")

        # Run the solver
        model_output = naive_solve(args, task, idx, to_print=False)
        model_question = task.get_input(idx)

        # Ground truth
        ground_truth_answer = task.ground_truth_answer(idx)

        # Extract final answer
        if args.generate_method == "cot":
            final_answer, weighted_probs, raw_probability = task.compute_final_answer(
                task, model_output, model_question, verifier, error_log_path
            )
            # Probability formatting
            total_prob = sum(weighted_probs.values())
            if total_prob > 0:
                weighted_probs = {k: v / total_prob for k, v in weighted_probs.items()}
            else:
                weighted_probs = {k: 0 for k in weighted_probs.keys()}

            correct_answer_prob = weighted_probs.get(str(ground_truth_answer), 0)

            # Format raw probabilities nicely
            formatted_raw_prob = "; ".join(
                f"{ans}: [{', '.join(f'{p:.2%}' for p in probs)}]"
                for ans, probs in raw_probability.items()
            )
        else:
            # If you had a standard generation, you'd parse differently
            answer = model_output[0][0]
            final_answer = re.search(r"\d+(\.\d+)?", answer).group()

        total_count += 1

        # Check correctness
        if final_answer is not None and float(final_answer) == float(ground_truth_answer):
            correct_count += 1
        accuracy = correct_count / total_count

        # Collect for final metric
        all_predictions.append(final_answer)
        all_references.append(str(ground_truth_answer))

        # 6. Build the log_entry
        log_entry = (
            "----------------------\n"
            f"Problem {idx}: {model_question}\n"
            f"Reasoning step: {model_output}\n"
            f"Model Prediction / Ground Truth: {final_answer} / {ground_truth_answer}\n"
            f"Correct Predictions / Total Tests: {correct_count} / {total_count}\n"
            f"Weighted Verifier Probability: {correct_answer_prob:.2%}\n"
            f"Verifier Raw Probability: {formatted_raw_prob}\n"
            f"Current Accuracy: {accuracy:.2%}\n"
            "----------------------\n"
        )

        # 7. Write to the overall experiment log
        with open(overall_log_path, "a") as f:
            f.write(log_entry)

        # 8. If model was wrong, also write to error.log
        if final_answer is None or float(final_answer) != float(ground_truth_answer):
            with open(error_log_path, "a") as ef:
                ef.write(log_entry)

        # Print to console
        print(log_entry)

    # 9. Compute final metrics
    results_exact = exact_match_metric.compute(
        predictions=all_predictions,
        references=all_references
    )
    overall_accuracy = sum(
        1 for p, r in zip(all_predictions, all_references) if float(p) == float(r)
    ) / len(all_predictions)

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

    # 10. Append the final summary to the overall log
    with open(overall_log_path, "a") as f:
        f.write(final_summary)

    print(f"Hyperparameter log:      {hyperparams_path}")
    print(f"Overall experiment log:  {overall_log_path}")
    print(f"Error log:               {error_log_path}")
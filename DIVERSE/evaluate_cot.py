import argparse
import os
import json
from tasks.MGSM import MgsmTask
from tasks.AFRI_MGSM import AfriMgsmTask

def safe_extract_answer(task, chain, error_log):
    """
    Extracts and normalizes the final answer from a chain.
    It removes commas and dollar signs, then converts the value to a rounded integer (as a string).
    If extraction or conversion fails, an error message is logged and None is returned.
    """
    raw_ans = task.extract_final_answer(task, chain)
    if raw_ans is None:
        error_message = (
            "-------------------\n"
            "ERROR: Failed to extract final answer from chain:\n"
            f"{chain}\n"
            "-------------------\n"
        )
        error_log.write(error_message)
        return None
    # Remove commas and dollar signs.
    raw_ans = raw_ans.replace(',', '').replace('$', '')
    try:
        normalized = str(int(round(float(raw_ans))))
        return normalized
    except ValueError:
        error_message = (
            "-------------------\n"
            "ERROR: Failed to convert extracted answer to a number:\n"
            f"Extracted: {raw_ans}\n"
            f"Chain: {chain}\n"
            "-------------------\n"
        )
        error_log.write(error_message)
        return None

# Set up arguments.
args = argparse.Namespace(
    task='MGSM',
    lang='en',  # will be updated in the loop
    prompt_used=[0, 1, 2, 3, 4],
    selection_method='voting'
)

num_samples = 250

# Languages and prompt sets to try.
# languages = ['bn', 'de', 'es', 'fr', 'ja', 'ru','sw', 'th', 'zh']
languages = ['ibo']
prompts_to_use = [[0],[0, 1],[0, 1, 2],[0, 1, 2, 3],[0, 1, 2, 3, 4]]

for lang in languages:
    for pr in prompts_to_use:
        total_count = 0

        # Counters for each method.
        verifier_correct_count = 0
        voting_verifier_correct_count = 0
        self_consistency_correct_count = 0

        args.lang = lang
        args.prompt_used = pr

        # Create MGSM task.
        #task = MgsmTask(args)

        task = AfriMgsmTask(args)

        # Data file (update path as needed).
        #data_file_path = f"logs/MGSM-cot-dataset/{lang}/susu/5/chain_of_thought_records.jsonl"
        data_file_path = f"logs/AFRI_MGSM-cot-dataset/{lang}/koko/5/chain_of_thought_records.jsonl"

        # Create a run folder.
        run_folder = os.path.join("logs", "AFRI_MGSM-cot-test-dataset", lang, str(len(pr)))

        # run_folder = os.path.join("logs", "MGSM-cot-test-dataset", lang, str(len(pr)))
        os.makedirs(run_folder, exist_ok=True)

        # Combined log file that logs every instance.
        combined_log_path = os.path.join(run_folder, "overall_experiment.log")
        combined_log = open(combined_log_path, "w", encoding="utf-8")

        # Error log files for each method.
        error_log_verifier_path = os.path.join(run_folder, "error_verifier.log")
        error_log_voting_verifier_path = os.path.join(run_folder, "error_voting_verifier.log")
        error_log_self_consistency_path = os.path.join(run_folder, "error_self_consistency.log")

        error_log_verifier = open(error_log_verifier_path, "w", encoding="utf-8")
        error_log_voting_verifier = open(error_log_voting_verifier_path, "w", encoding="utf-8")
        error_log_self_consistency = open(error_log_self_consistency_path, "w", encoding="utf-8")

        # Read the JSONL file.
        with open(data_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines[:num_samples], start=1):
            total_count += 1
            record = json.loads(line)
            ground_truth = record.get("ground_truth")
            english_question = record.get("question", "No question found")
            reasoning_paths = record.get("reasoning_paths", [])

            # Filter allowed reasoning paths.
            allowed_paths = [path for i, path in enumerate(reasoning_paths) if i in pr]
            combined_reasoning = " | ".join(path.get("chain", "") for path in allowed_paths)

            # Build a dictionary mapping each extracted answer to a list of probabilities.
            answer_probs = {}
            counts = {}  # for self-consistency counts
            for path in allowed_paths:
                # Use safe extraction; log errors to verifier's error log.
                norm_ans = safe_extract_answer(task, path.get("chain", ""), error_log_verifier)
                prob = path.get("probability", 0)
                if norm_ans is not None:
                    answer_probs.setdefault(norm_ans, []).append(prob)
                    counts[norm_ans] = counts.get(norm_ans, 0) + 1

            total_prob = sum(sum(probs) for probs in answer_probs.values())
            if str(ground_truth) in answer_probs and total_prob > 0:
                weighted_prob_correct = sum(answer_probs[str(ground_truth)]) / total_prob
            else:
                weighted_prob_correct = 0

            # Method 1: Verifier – choose the single chain with the highest probability.
            best_path = max(allowed_paths, key=lambda x: x.get("probability", 0))
            final_answer_verifier = safe_extract_answer(task, best_path.get("chain", ""), error_log_verifier)
            verifier_prob = best_path.get("probability", 0)

            # Method 2: Voting Verifier – sum probabilities for each answer.
            if answer_probs:
                summed = {ans: sum(probs) for ans, probs in answer_probs.items()}
                final_answer_voting_verifier = max(summed, key=lambda a: summed[a])
            else:
                final_answer_voting_verifier = None

            # Method 3: Self-Consistency – choose the answer that occurs most frequently.
            if counts:
                final_answer_self = max(counts, key=lambda a: counts[a])
            else:
                final_answer_self = None

            # Update correct counts.
            try:
                gt_val = float(ground_truth)
            except ValueError:
                gt_val = None

            if final_answer_verifier is not None and gt_val is not None and float(final_answer_verifier) == gt_val:
                verifier_correct_count += 1
            if final_answer_voting_verifier is not None and gt_val is not None and float(final_answer_voting_verifier) == gt_val:
                voting_verifier_correct_count += 1
            if final_answer_self is not None and gt_val is not None and float(final_answer_self) == gt_val:
                self_consistency_correct_count += 1

            verifier_acc = verifier_correct_count / total_count
            voting_verifier_acc = voting_verifier_correct_count / total_count
            self_consistency_acc = self_consistency_correct_count / total_count

            # Build the combined log entry.
            log_entry = (
                "----------------------\n"
                f"Problem: {english_question}\n"
                f"Reasoning step: {combined_reasoning}\n"
                f"Verifier method prediction / Ground Truth: {final_answer_verifier} / {ground_truth}\n"
                f"Voting Verifier Method Prediction / Ground Truth: {final_answer_voting_verifier} / {ground_truth}\n"
                f"Voting (Self-Consistency) Method Prediction / Ground Truth: {final_answer_self} / {ground_truth}\n"
                f"Weighted Verifier Probability: {weighted_prob_correct:.2%}\n"
                f"Current Accuracy: Verifier: {verifier_acc:.2%}, Voting Verifier: {voting_verifier_acc:.2%}, Self-Consistency: {self_consistency_acc:.2%}\n"
                "----------------------\n"
            )

            # Write the combined log entry.
            combined_log.write(log_entry)
            combined_log.flush()  # Force immediate write to file

            # If a method's prediction is wrong, append the entry to its corresponding error log.
            if final_answer_verifier is None or (gt_val is not None and float(final_answer_verifier) != gt_val):
                error_log_verifier.write(log_entry)
            if final_answer_voting_verifier is None or (gt_val is not None and float(final_answer_voting_verifier) != gt_val):
                error_log_voting_verifier.write(log_entry)
            if final_answer_self is None or (gt_val is not None and float(final_answer_self) != gt_val):
                error_log_self_consistency.write(log_entry)

        # Close all log files.
        combined_log.close()
        error_log_verifier.close()
        error_log_voting_verifier.close()
        error_log_self_consistency.close()

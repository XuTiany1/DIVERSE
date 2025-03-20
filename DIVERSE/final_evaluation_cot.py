import argparse
import os
import json
from tasks.MGSM import MgsmTask
from tasks.AFRI_MGSM import AfriMgsmTask
import evaluate


def safe_extract_answer(task, chain, error_log):
    """
    Extract and round final answer from chain
    """

    raw_ans = task.extract_final_answer(task, chain)

    # Check if extraction is good
    if (raw_ans is None) or (raw_ans == '0'):
        error_message = (
            "-------------------\n"
            "ERROR: Failed to extract final answer from chain:\n"
            f"{chain}\n"
            "-------------------\n"
        )
        error_log.write(error_message)
        return None
    
    raw_ans = raw_ans.replace(',', '').replace('$', '')

    # Check if number can be converted
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
    
def convert_pretty_printed_to_jsonl(input_path, output_path):
    """
    Reads an input file that contains multiple pretty-printed JSON objects (one after the other)
    and writes them into a JSONL file (one JSON object per line).
    """
    objects = []
    current_lines = []
    brace_count = 0
        
    # Read the input file line by line.
    with open(input_path, 'r', encoding='utf-8') as infile:
        for line in infile:
            # Skip empty lines.
            if not line.strip():
                continue

            # Update the brace counter.
            brace_count += line.count('{')
            brace_count -= line.count('}')
            current_lines.append(line)

            # When brace_count is zero, we assume a complete JSON object has been read.
            if brace_count == 0 and current_lines:
                obj_str = ''.join(current_lines)
                try:
                    obj = json.loads(obj_str)
                    objects.append(obj)
                except json.JSONDecodeError as e:
                    print("Error decoding JSON object:", e)
                # Reset the buffer for the next object.
                current_lines = []
        
    # Write each JSON object as a single line in the output JSONL file.
    with open(output_path, 'w', encoding='utf-8') as outfile:
        for obj in objects:
            json_line = json.dumps(obj, ensure_ascii=False)
            outfile.write(json_line + "\n")

# ===================================================================
# ====================Main Logic STARTS HERE ========================
# ===================================================================

# Set up arguments.
args = argparse.Namespace(
    task='MGSM',
    languages = ['ibo'],
    # languages = ['bn', 'de', 'es', 'fr', 'ja', 'ru', 'sw', 'th', 'zh'],
    prompts_to_use = [[0,1,2,3,4]],
    post_process_dataset=False
)

num_samples = 250
exact_match_metric = evaluate.load("exact_match")


# ===================================================================
# =======================Set up Dataset Loop=========================
# ===================================================================

# Process dataset only if required
if (args.post_process_dataset == True):
    # Set up prompt and probability info
    for lang in args.languages:

        args.lang = lang
        task = AfriMgsmTask(args)
        # task = MgsmTask(args)

        data_file_path = f"DIVERSE/logs/AFRI_MGSM-cot-dataset/{lang}/koko/5/chain_of_thought_records.jsonl"
        #data_file_path = f"DIVERSE/logs/MGSM-cot-dataset/{lang}/susu/5/chain_of_thought_records.jsonl"

        run_folder = os.path.join("results", "AFRI_MGSM-cot-test-dataset", lang)
        # run_folder = os.path.join("results", "MGSM-cot-test-dataset", lang)
        os.makedirs(run_folder, exist_ok=True)

        reasoning_with_prob_info_path = os.path.join(run_folder, "overall_experiment.jsonl")
        reasoning_with_prob_info_log = open(reasoning_with_prob_info_path, "w", encoding="utf-8")

        error_extraction_path = os.path.join(run_folder, "error_verifier.log")
        error_extraction_log = open(error_extraction_path, "w", encoding="utf-8")


        with open(data_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Loop through the entire file
        for idx, line in enumerate(lines[:num_samples], start=1):
            
            # Extract from jsonl
            record = json.loads(line)
            ground_truth = record.get("ground_truth")
            english_question = record.get("question", "No question found")
            all_reasoning_paths = record.get("reasoning_paths", [])

            answer_probs = {}

            for path in all_reasoning_paths:

                norm_ans = safe_extract_answer(task, path.get("chain", ""), error_extraction_log)
                prob = path.get("probability", 0)

                if norm_ans is not None:
                    answer_probs.setdefault(norm_ans, []).append(prob)
            
            log_entry = {
                "problem": english_question,
                "record": record,
                "verifier_probability": answer_probs
            }

            # Write the combined log entry as pretty-printed JSON.
            reasoning_with_prob_info_log.write(json.dumps(log_entry, indent=2) + "\n")

        reasoning_with_prob_info_log.close()
        error_extraction_log.close()


# ===================================================================
# ==========================Evaluation Loop==========================
# ===================================================================

# Set up evaluation loop
for lang in args.languages:
    for pr in args.prompts_to_use:

        total_count = 0

        # Counters for each method.
        verifier_correct_count = 0
        voting_verifier_correct_count = 0
        self_consistency_correct_count = 0

        # Lists to accumulate predictions and references for exact match evaluation.
        verifier_preds = []
        voting_verifier_preds = []
        voting_only_preds = []
        ground_truth_refs = []

        args.lang = lang
        args.prompt_used = pr

        data_file_path = f"results/AFRI_MGSM-cot-test-dataset/{lang}/overall_experiment.jsonl"
        #data_file_path = f"results/MGSM-cot-test-dataset/{lang}/overall_experiment.jsonl"

        run_folder = os.path.join("results", "AFRI_MGSM-cot-test-dataset", lang, str(len(pr)))
        # run_folder = os.path.join("results", "MGSM-cot-test-dataset", lang, str(len(pr)))
        os.makedirs(run_folder, exist_ok=True)


        # Combined log file that logs every instance.
        combined_log_path = os.path.join(run_folder, "overall_results.log")
        combined_log = open(combined_log_path, "w", encoding="utf-8")

        error_log_verifier_path = os.path.join(run_folder, "error_verifier.log")
        error_log_voting_verifier_path = os.path.join(run_folder, "error_voting_verifier.log")
        error_log_self_consistency_path = os.path.join(run_folder, "error_self_consistency.log")

        error_log_verifier = open(error_log_verifier_path, "w", encoding="utf-8")
        error_log_voting_verifier = open(error_log_voting_verifier_path, "w", encoding="utf-8")
        error_log_self_consistency = open(error_log_self_consistency_path, "w", encoding="utf-8")

        processing_error_log_path = os.path.join(run_folder, "reading_error.log")
        processing_error_log = open(processing_error_log_path, "w", encoding="utf-8")

        input_file_path =  f"results/AFRI_MGSM-cot-test-dataset/{lang}/overall_experiment.jsonl"   
        output_file_path =  f"results/AFRI_MGSM-cot-test-dataset/{lang}/cleaned_output.jsonl"     
        # input_file_path =  f"results/MGSM-cot-test-dataset/{lang}/overall_experiment.jsonl"   
        # output_file_path =  f"results/MGSM-cot-test-dataset/{lang}/cleaned_output.jsonl"     


        convert_pretty_printed_to_jsonl(input_file_path, output_file_path)

        print('hello')

        with open(output_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines[:num_samples], start=1):

            total_count += 1

            record = json.loads(line)
            
            question = record["problem"]
            ground_truth = record.get("record", {}).get("ground_truth")
            verifier_probability = record["verifier_probability"]

            try:
                gt_val = float(ground_truth)
            except ValueError:
                gt_val = None


            # Extract prediction
            try:
                # Verifier: key with the highest single probability
                verifier_only = max(verifier_probability, key=lambda k: max(verifier_probability[k]))
                # Voting Verifier: key with the highest sum of probabilities
                voting_verifier = max(verifier_probability, key=lambda k: sum(verifier_probability[k]))
                # Voting: key with the highest number of entries
                voting_only = max(verifier_probability, key=lambda k: len(verifier_probability[k]))

            except Exception as e:
                error_log_entry = (
                    "----------------------\n"
                    f"Exception: {e}\n"
                    f"Problem: {question}\n"
                    f"Record: {record}\n"
                    f"Verifier probability: {verifier_probability}\n"
                    "----------------------\n"

                )
                processing_error_log.write(error_log_entry)
                processing_error_log.flush()   


            # Increament count if answer matches ground truth
            if verifier_only is not None and gt_val is not None and float(verifier_only)==gt_val:
                verifier_correct_count += 1
            if voting_verifier is not None and gt_val is not None and float(voting_verifier)==gt_val:
                voting_verifier_correct_count += 1
            if voting_only is not None and gt_val is not None and float(voting_only)==gt_val:
                self_consistency_correct_count += 1

            verifier_acc = verifier_correct_count / total_count
            voting_verifier_acc = voting_verifier_correct_count / total_count
            self_consistency_acc = self_consistency_correct_count / total_count

            total_prob_sum = sum(sum(probs) for probs in verifier_probability.values())
            if str(ground_truth) in verifier_probability and total_prob_sum > 0:
                weighted_prob_correct = sum(verifier_probability[str(ground_truth)]) / total_prob_sum
            else:
                weighted_prob_correct = 0


            # ----- Exact Match Evaluation using evaluate library -----
            # Convert predictions to strings (or empty string if None)
            verifier_pred_str = str(verifier_only) if verifier_only is not None else ""
            voting_verifier_pred_str = str(voting_verifier) if voting_verifier is not None else ""
            voting_only_pred_str = str(voting_only) if voting_only is not None else ""
            gt_str = str(ground_truth)

            # Append to lists for the metric evaluation
            verifier_preds.append(verifier_pred_str)
            voting_only_preds.append(voting_only_pred_str)
            voting_verifier_preds.append(voting_verifier_pred_str)
            ground_truth_refs.append(gt_str)

            # Compute current exact match accuracy for each method
            current_em_verifier = exact_match_metric.compute(
                predictions=verifier_preds, references=ground_truth_refs
            )["exact_match"]
            current_em_voting_verifier = exact_match_metric.compute(
                predictions=voting_verifier_preds, references=ground_truth_refs
            )["exact_match"]
            current_em_voting_only = exact_match_metric.compute(
                predictions=voting_only_preds, references=ground_truth_refs
            )["exact_match"]


            log_entry = (
                "----------------------\n"
                f"Problem: {question}\n"
                f"Record: {record}\n"
                f"Verifier probability: {verifier_probability}\n"
                f"Verifier method prediction / Ground Truth: {verifier_only} / {ground_truth}\n"
                f"Voting Verifier Method Prediction / Ground Truth: {voting_verifier} / {ground_truth}\n"
                f"Voting (Self-Consistency) Method Prediction / Ground Truth: {voting_only} / {ground_truth}\n"
                f"Weighted Verifier Probability: {weighted_prob_correct:.2%}\n"
                f"Cumulative correct answer numbers: Verifier: {verifier_correct_count}/{total_count}, Voting Verifier: {voting_verifier_correct_count}/{total_count}, Self-Consistency: {self_consistency_correct_count}/{total_count}\n"
                f"Current Manual Accuracy: Verifier: {verifier_acc:.2%}, Voting Verifier: {voting_verifier_acc:.2%}, Self-Consistency: {self_consistency_acc:.2%}\n"
                
                f"Exact Match Accuracy: Verifier: {current_em_verifier:.2%}, Voting Verifier: {current_em_voting_verifier:.2%}, Self-Consistency: {current_em_voting_only:.2%}\n"
                "----------------------\n"
            )

            # Write the combined log entry.
            combined_log.write(log_entry)
            combined_log.flush()  # Force immediate write to file

            # If a method's prediction is wrong, append the entry to its corresponding error log.
            if verifier_only is None or (gt_val is not None and float(verifier_only) != gt_val):
                error_log_verifier.write(log_entry)
            if voting_verifier is None or (gt_val is not None and float(voting_verifier) != gt_val):
                error_log_voting_verifier.write(log_entry)
            if voting_only is None or (gt_val is not None and float(voting_only) != gt_val):
                error_log_self_consistency.write(log_entry)


        # Close all log files.
        combined_log.close()
        error_log_verifier.close()
        error_log_voting_verifier.close()
        error_log_self_consistency.close()









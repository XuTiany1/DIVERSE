import argparse
import os
import json
from tasks.MGSM import MgsmTask
from tasks.AFRI_MGSM import AfriMgsmTask


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



# Set up arguments.
args = argparse.Namespace(
    task='MGSM',
    lang='en', 
    prompt_used=[0, 1, 2, 3, 4],
    selection_method='voting'
)

num_samples = 250

# languages = ['bn', 'de', 'es', 'fr', 'ja', 'ru','sw', 'th', 'zh']
languages = ['ibo']
prompts_to_use = [[0],[0, 1],[0, 1, 2],[0, 1, 2, 3],[0, 1, 2, 3, 4]]


skip = True
if (skip == False):
    # Set up prompt and probability info
    for lang in languages:

        args.lang = lang
        task = AfriMgsmTask(args)

        data_file_path = f"logs/AFRI_MGSM-cot-dataset/{lang}/koko/5/chain_of_thought_records.jsonl"

        run_folder = os.path.join("debug-log", "AFRI_MGSM-cot-test-dataset", lang)
        os.makedirs(run_folder, exist_ok=True)

        reasoning_with_prob_info_path = os.path.join(run_folder, "overall_experiment.log")
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
            #every_path_answer = {}

            for path in all_reasoning_paths:

                norm_ans = safe_extract_answer(task, path.get("chain", ""), error_extraction_log)
                prob = path.get("probability", 0)

                if norm_ans is not None:
                    answer_probs.setdefault(norm_ans, []).append(prob)
                    #every_path_answer[path["chain"]] = norm_ans
            
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
# ===================================================================
# ===================================================================

# Set up evaluation loop
languages = ['ibo']
prompts_to_use = [[0, 1, 2, 3, 4]]
for lang in languages:
    for pr in prompts_to_use:

        total_count = 0

        # Counters for each method.
        verifier_correct_count = 0
        voting_verifier_correct_count = 0
        self_consistency_correct_count = 0

        args.lang = lang
        args.prompt_used = pr

        data_file_path = f"debug-log/AFRI_MGSM-cot-test-dataset/{lang}/overall_experiment.jsonl"

        run_folder = os.path.join("debug-2-log", "AFRI_MGSM-cot-test-dataset", lang, str(len(pr)))
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

        with open(data_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()


        # Prepare data for evaluation:
        # prepare data for evaluation loop
        data_file_path = "debug-log/AFRI_MGSM-cot-test-dataset/ibo/overall_experiment.jsonl"

        with open(data_file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        # Convert the concatenated objects into a JSON array.





        #def read_pretty_printed_jsonl(filepath):
        #    records = []
        #    with open(filepath, "r", encoding="utf-8") as f:
        #        current_lines = []
        #        brace_count = 0
        #        
        #        for line in f:
        #            # Skip empty lines to avoid counting them.
        #            if not line.strip():
        #                continue
        #            
        #            # Add the current line to our buffer.
        #            current_lines.append(line)
        #            # Increase count for every '{' and decrease for every '}'.
        #            # (This simple count assumes no curly braces occur within string literals.)
        #            brace_count += line.count("{")
        #            brace_count -= line.count("}")
        #            
        #            # When our counter reaches 0, we assume the object is complete.
        #            if brace_count == 0 and current_lines:
        #                try:
        #                    obj_str = "".join(current_lines)
        #                    records.append(json.loads(obj_str))
        #                except json.JSONDecodeError as e:
        #                    print("Error decoding JSON:", e)
        #                # Reset for the next object.
        #                current_lines = []
        #                
        #    return records






        input_file_path =  f"debug-log/AFRI_MGSM-cot-test-dataset/{lang}/overall_experiment.jsonl"   # file containing the pretty-printed JSON objects
        output_file_path =  f"debug-log/AFRI_MGSM-cot-test-dataset/{lang}/output.jsonl"      # destination JSONL file

        convert_pretty_printed_to_jsonl(input_file_path, output_file_path)

        print('hello')

        with open(output_file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for idx, line in enumerate(lines[:num_samples], start=1):


            total_count += 1

            record = json.loads(line)
            
            question = record["problem"]
            #ground_truth = record["ground_truth"]
            ground_truth = record.get("record", {}).get("ground_truth")
            verifier_probability = record["verifier_probability"]

            try:
                gt_val = float(ground_truth)
            except ValueError:
                gt_val = None


            # Extract prediction
            # Verifier: key with the highest single probability
            verifier_only = max(verifier_probability, key=lambda k: max(verifier_probability[k]))
                
            # Voting Verifier: key with the highest sum of probabilities
            voting_verifier = max(verifier_probability, key=lambda k: sum(verifier_probability[k]))

            # Voting: key with the highest number of entries
            voting_only = max(verifier_probability, key=lambda k: len(verifier_probability[k]))


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


            log_entry = (
                "----------------------\n"
                f"Problem: {question}\n"
                # f"Reasoning step: {combined_reasoning}\n"
                f"Record: {record}\n"
                f"Verifier probability: {verifier_probability}\n"
                f"Verifier method prediction / Ground Truth: {verifier_only} / {ground_truth}\n"
                f"Voting Verifier Method Prediction / Ground Truth: {voting_verifier} / {ground_truth}\n"
                f"Voting (Self-Consistency) Method Prediction / Ground Truth: {voting_only} / {ground_truth}\n"
                f"Weighted Verifier Probability: {weighted_prob_correct:.2%}\n"
                f"Current Accuracy: Verifier: {verifier_acc:.2%}, Voting Verifier: {voting_verifier_acc:.2%}, Self-Consistency: {self_consistency_acc:.2%}\n"
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









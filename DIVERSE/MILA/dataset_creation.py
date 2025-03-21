import argparse
import os
import json
from methods.naive_solve import naive_solve
from tasks.MATH import MATH
#from model.deberta_v3.deberta_verifier import Verifier


    
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
# ====================Main Logic STARTS HERE=========================
# ===================================================================

# Define argparse namespace
args = argparse.Namespace( 
    task='MGSM', 
    lang='en',
    naive_run=False, 
    number_generate_sample=1, 
    total_dataset_sample = 250,
    languages = ['orm', 'sna', 'sot', 'swa', 'twi', 'vai', 'wol', 'xho', 'yor', 'zul'],
    prompt_used=[0,1,2,3,4,5],
    selection_method='voting',
    checkpoint_path='/home/mila/x/xut/github/DIVERSE/DIVERSE/model/deberta_v3/checkpoint-6565',
    tokenizer_name='microsoft/deberta-v3-large',
    generator_model='gpt-4o',
    post_process_dataset=True
)

# ===================================================================
# =========================Set up Variable===========================
# ===================================================================
correct_count = 0

# Load verifier only if using chain-of-thought generation
#verifier = Verifier(args)

run_folder = os.path.join("multilingual_reasoning_dataset")
os.makedirs(run_folder, exist_ok=True)

# ===================================================================
# ===============Creating Model Reasoning Dataset====================
# ===================================================================

for lang in args.languages:

    error_questions = []

    # To calculate 
    args.lang = lang
    task = MATH(args)

    # Prepare the hyperparameter log file
    hyperparams_path = os.path.join(run_folder, f"{lang}-hyperparams.log")
    with open(hyperparams_path, "w") as hp:
        hp.write("--- Hyperparameters / Args ---\n")
        for k, v in vars(args).items():
            hp.write(f"{k}: {v}\n")

    # Instead of accumulating all records in memory, open a JSON Lines file in append mode.
    json_filename = os.path.join(run_folder, f"{lang}_dataset.jsonl")
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)

    # loop through all test instances
    for idx in range(1, args.total_dataset_sample):

        print(f"\n--- Running Test {idx} ---")

        # First, get the COT model answer
        args.generate_method ='cot'
        model_output = naive_solve(args, task, idx, to_print=False)
        model_question = task.get_input(idx)
        english_question = task.get_english_input(idx)

        try:
            # Ground truth
            ground_truth_answer = task.ground_truth_answer(idx)
            # Convert to Python int in case it's a numpy.int64
            ground_truth_answer = int(ground_truth_answer)
        except Exception as e:
            print(f"Error converting ground truth for test {idx}: {e}")
            error_questions.append(english_question)

        # Compute a dictionary mapping each reasoning path to its verifier probability.
        reasoning_path_probability = task.compute_probability(task, 
                                                              model_output, 
                                                              english_question, 
                                                              verifier)




        # Second, get the straight up model answer
        args.generate_method ='cot'
        model_output = naive_solve(args, task, idx, to_print=False)


        record = {
            "question": english_question,
            "ground_truth": ground_truth_answer,
            "reasoning_paths": []
        }





# ===================================================================
# =======================Set up Dataset Loop=========================
# ===================================================================

# Process dataset only if required
if (args.post_process_dataset == True):

    # Set up prompt and probability info
    for lang in args.languages:

        args.lang = lang
        task = MATH(args)
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
        for idx, line in enumerate(0, start=1):
            
            # Extract from jsonl
            record = json.loads(line)
            ground_truth = record.get("ground_truth")
            english_question = record.get("question", "No question found")
            all_reasoning_paths = record.get("reasoning_paths", [])

            answer_probs = {}

            for path in all_reasoning_paths:

                #norm_ans = safe_extract_answer(task, path.get("chain", ""), error_extraction_log)
                norm_ans = None
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

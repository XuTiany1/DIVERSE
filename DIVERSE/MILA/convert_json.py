import argparse
import os
import json
import sys
from methods.naive_solve import naive_solve
from tasks.MATH import MATH
from model.deberta_v3.deberta_verifier import Verifier


# ===================================================================
# ====================Main Logic STARTS HERE=========================
# ===================================================================


# Define argparse namespace
args = argparse.Namespace( 
    task='MGSM', 
    lang='ibo',
    naive_run=False, 
    number_generate_sample=1, 
    total_dataset_sample = 250,
    # 'sna', 'sot', 'swa', 'twi', 'vai', 'wol', 'xho', 'yor', 'zul', 'amh', 'ewe', 'hau', 'ibo', 
    #languages = ['sna', 'sot', 'swa', 'twi', 'vai', 'wol', 'xho', 'yor', 'zul', 'amh', 'ewe', 'hau', 'ibo', 'kin', 'lin', 'lug'],
    # need to finish spanish
    languages = ['ja', 'ru', 'sw', 'te', 'th', 'zh'],
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

# Load verifier only if using chain-of-thought generation
verifier = Verifier(args)

run_folder = os.path.join("multilingual_reasoning_dataset")
os.makedirs(run_folder, exist_ok=True)

# ===================================================================
# ===============Creating Model Reasoning Dataset====================
# ===================================================================

data_file_path = f"multilingual_reasoning_dataset/ibo_.jsonl"
jsonl_file_path = f"multilingual_reasoning_dataset/ibo_new.jsonl"

with open(data_file_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

task = MATH(args)

for idx, line in enumerate(lines):

    record = json.loads(line)

    english_question = record["question"]
    ground_truth = record["ground_truth"]
    reasoning_paths = record["reasoning_paths"]

    # Construct the list of reasoning paths
    reasoning_path_probability = {}
    for item in reasoning_paths:
        reasoning = item["chain"]
        prob = item["probability"]
        reasoning_path_probability[reasoning] = prob

    record = {
        "question": english_question,
        "ground_truth": ground_truth,
        "reasoning_paths": []
    }

    for reasoning_path, probability in reasoning_path_probability.items():
        record["reasoning_paths"].append({
            "chain": reasoning_path,
            "answer": task.extract_final_number(task, reasoning_path),
            "probability": probability
        })

    # Append the record to the JSON Lines file.
    with open(jsonl_file_path, "a") as fp:
        fp.write(json.dumps(record) + "\n")




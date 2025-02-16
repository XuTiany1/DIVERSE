
"""
Prepares the training and testing data for the verifier by processing generator outputs 
and creating labeled sequences for training a sequence-labeling model
"""



"""
QUESTION: 
1. What generator outputs? like from what model? is it just datasets downloaded from GSM8K?

These are the outputs produced by a language model (or generator) that has generated reasoning paths for each GSM8K problem. 
They aren't simply the raw GSM8K dataset; instead, they are the processed results—containing the problem text ("context"), the 
generated reasoning paths ("samples"), and additional information like the question and the correct answer ("metadata")—from a 
model that was run on the GSM8K data.


----


2. What does prompt_data_dict store? and how does it differ from prompt_data? 

prompt_data:
This is a list where each element is a dictionary holding a single sample. Each dictionary contains:
    - "context": The problem text.
    - "sample": One generated reasoning path.
    - "metadata": Information about the question and ground truth.

    

prompt_data_dict:
This is a dictionary that organizes the data by the question. Here, each key is a question, and 
the corresponding value is a list of all the processed samples (i.e., reasoning paths) associated with that question.


Essentially, while prompt_data is a flat list of all samples, prompt_data_dict groups these samples by question, making 
it easier to handle multiple reasoning paths per question.


---

"""



####################
# IMPORT LIBRARIES
####################
import os
import json
import random
import argparse
from tqdm import tqdm
import re
import utils_io
from utils import (
    GSM8KCase,
    TextEntailmentCase,
    GSM8KExample,
    TextEntailmentExample,
    compute_top1_and_recall,
    post_process_answer_clutrr_mapping,
    post_process_answer_clutrr_cutoff,
)
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)
import torch
import pdb
import logging


logger = logging.getLogger(__name__)

####################
# DATASET RELATED MAPPING
####################
case_class_map = {
    "GSM8K": GSM8KCase,
    "CLUTRR": TextEntailmentCase,
    "strategyQA": TextEntailmentCase,
}

example_class_map = {
    "GSM8K": GSM8KExample,
    "CLUTRR": TextEntailmentExample,
    "strategyQA": TextEntailmentExample,
}

# This is specifically related to CLUTRR
relation_reverse_map = {
    'sister': ['brother'],
    'son': ['father', 'mother'],
    'aunt': ['nephew', 'niece'],
    'granddaughter': ['grandfather', 'grandmother'],
    'father': ['son', 'daughter'],
    'grandfather': ['grandson', 'granddaughter'],
    'grandmother': ['grandson', 'granddaughter'],
    'mother-in-law': ['son-in-law', 'daughter-in-law'],
    'uncle': ['nephew', 'niece'],
    'niece': ['uncle', 'aunt'],
    'mother': ['son', 'daughter'],
    'brother': ['sister'],
    'daughter': ['father', 'mother'],
    'nephew': ['uncle', 'aunt'],
    'grandson': ['grandfather', 'grandmother'],
    'son-in-law': ['father-in-law', 'mother-in-law'],
    'father-in-law': ['son-in-law', 'daughter-in-law'],
    'daughter-in-law': ['father-in-law', 'mother-in-law'],
}

device = "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"


def main():

    ####################
    # Argument Parsing and Model Loading(model loaded if not GSM8k, so no worry for me)
    ####################
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator_result_file", type=str, default=None, help="generator output file in .jsonl format")
    parser.add_argument("--output_dir", type=str, default=None, help="output dir")
    parser.add_argument("--random_seed", type=int, default=233, help="random_seed")
    parser.add_argument("--split", type=str, default="train", help="split (train or test)")
    parser.add_argument("--dataset_name", type=str, default="GSM8K", help="GSM8K, CLUTRR, strategyQA")
    parser.add_argument("--text_entailment_model_name", type=str, default="roberta-large-mnli", help="roberta-large-mnli, facebook/bart-large-mnli, etc.")
    parser.add_argument("--text_entailment_batch_size", type=int, default=512, help="text entailment batch size")
    args = parser.parse_args()

    random.seed(args.random_seed)

    # Only load textual entail model if dataset is NOT GSM8K
    if args.dataset_name != "GSM8K":
        logger.info("Loading textual entailment models...")
        model = AutoModelForSequenceClassification.from_pretrained(args.text_entailment_model_name).to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(args.text_entailment_model_name)
    else:
        model = None
        tokenizer = None
    



    ####################
    # Loading and Preprocessing Generator Outputs
    ####################

    # read generator output file. 
    """
    Note: Each line corresponds to a generator output which includes:
        A context (problem text)
        A list of samples (model-generated answers)
        meta data that contains additional information like the question and the ground truth.
    """
    generator_outputs = [json.loads(line) for line in open(utils_io.get_file(args.generator_result_file))]
    question_to_ground_truth = {}

    # building prompt_data
    prompt_data = []

    # Loop through the generator outputs, and for each sample in an output, it stores a dictionary 
    # with the context, sample, and metadata
    for generator_output in generator_outputs:

        # Extract structured fields from JSONL
        context = generator_output["context"]       # Problem text
        samples = generator_output["samples"]       # Model-generated answers
        for sample in samples:
            metadata = generator_output["metadata"] # Model-generated answers
            prompt_data.append({"context": context, "sample": sample, "metadata": metadata})


    # Key: Question
    # Value: 
    prompt_data_dict = {}

    # some pre-processing about formulas and answers for GSM8K and other datasets
    for obj in tqdm(prompt_data):
        question = obj["metadata"]["question"].strip().replace("\n", "")
        
        # Find the final answere in the sample
        def extract_solution(sample):
            sample = sample.strip()
            if '####' in sample:
                stop = sample.find('\n\n', sample.index('####'))
                if stop >= 0:
                    sample = sample[:stop]
            sample = sample.replace('\n\n', '\n')
            return sample
        sample = extract_solution(obj["sample"])
        
        # In whatever code that's in this for loop, store question, ground truth, and predictions
        sample = sample.strip().replace("\n", "%%")  # for sequence labeling
        ground_truth = obj["metadata"]["ground_truth"].strip().replace("\n\n", "\n").replace("\n", "%%")  # for sequence labeling
        if args.dataset_name == "GSM8K":
            if "####" not in sample:
                reg = "<<.+>>[\d\.]+"
                eqs = re.findall(reg, sample)
                if len(eqs) > 0:
                    final_answer = eqs[-1].split(">>")[-1].strip()
                    if final_answer and len(final_answer) > 0 and final_answer[-1] == '.':
                        final_answer = final_answer[:-1]
                    if sample[-2:] == "%%":
                        sample = sample + "####" + final_answer
                    else:
                        sample = sample + "%%####" + final_answer



        elif args.dataset_name == "CLUTRR":
            pass
            if "####" not in sample:
                reg = "the.+?of"
                eqs = re.findall(reg, sample)
                if len(eqs) > 0:
                    final_answer = eqs[-1].replace("the ", "").replace(" of", "")
                    if sample[-2:] == "%%":
                        sample = sample + "####" + final_answer
                    else:
                        sample = sample + "%%####" + final_answer
        if question not in prompt_data_dict:
            prompt_data_dict[question] = []
        
        sample = sample.replace("\n", "%%")  # for sequence labeling
        ground_truth = ground_truth.replace("\n", "%%")  # for sequence labeling
        question_to_ground_truth[question] = ground_truth
        prompt_data_dict[question].append(sample)


    #########################
    # convert Data to Case objects
    #########################


    # determine the min number of samples available across questions to ensure uniformity
    min_sample_num_per_case  = 99999999
    for k in prompt_data_dict:
        min_sample_num_per_case = min(min_sample_num_per_case, len(prompt_data_dict[k]))

    # Constructing cases
    """
    For each question, a case object is created using the corresponding class (e.g., GSM8KCase), setting:
        - The question text.
        - The ground truth wrapped in an example object.
        - The list of predictions (each wrapped in an example object) limited to the minimum sample count.
    """
    prompt_cases = []
    for k in prompt_data_dict:
        case = case_class_map[args.dataset_name]("", [])
        case.question = k
        case.ground_truth = example_class_map[args.dataset_name](question_to_ground_truth[k])
        case.entailment_batch_size = args.text_entailment_batch_size
        for sample_idx, x in enumerate(prompt_data_dict[k]):
            if sample_idx >= min_sample_num_per_case:
                break
            pred = example_class_map[args.dataset_name](x)
            case.preds.append(pred)
        prompt_cases.append(case)
    print(f"Total cases: {len(prompt_cases)}".replace("\n", "\\n"))
    print(f"Case 0's question: {prompt_cases[0].question}".replace("\n", "\\n"))
    print(f"Case 0's ground truth: {prompt_cases[0].ground_truth.content}".replace("\n", "\\n"))
    print(f"Case 0's sample0: {prompt_cases[0].preds[0].content}".replace("\n", "\\n"))

    #########################
    # Compute data statistics
    #########################
    print("*********** Data statistics ***********")
    res = compute_top1_and_recall(data=prompt_cases)
    for k in res:
        print(f"{k}: {res[k]}")
    print("")

    if args.dataset_name == "CLUTRR":
        prompt_cases = post_process_answer_clutrr_cutoff(prompt_cases)
        # print the random top1 and recall of the data
        print("*********** Data statistics (after post processing for CLUTRR) ***********")
        res = compute_top1_and_recall(data=prompt_cases)
        for k in res:
            print(f"{k}: {res[k]}")
        print("")

    #########################
    # Step-wise labeling
    #########################
    for j, case in enumerate(tqdm(prompt_cases)):
        case.do_step_labeling(model=model, tokenizer=tokenizer)
        
    for case_idx, case in enumerate(tqdm(prompt_cases)):
        case.ground_truth.sequence_labels = example_class_map[args.dataset_name].get_sequence_labels(case.question, case.ground_truth)      
        for pred_idx, pred in enumerate(case.preds):
            pred.sequence_labels = example_class_map[args.dataset_name].get_sequence_labels(case.question, pred)
    
    sequence_data = []
    for case_idx, case in enumerate(tqdm(prompt_cases)):
        sequence_data.append(case.ground_truth.sequence_labels)
        for pred_idx, pred in enumerate(case.preds):
            sequence_data.append(pred.sequence_labels)

    #########################
    # Shuffle and save data
    #########################
    if args.split == "train":
        random.shuffle(sequence_data)
    
    with open(os.path.join(args.output_dir, '{}.txt'.format(args.split)), "w") as f:
        for i, arr in enumerate(tqdm(sequence_data)):
            for lhs, rhs in arr:
                f.write(f"{lhs} {rhs}\n")
            f.write("\n")

if __name__ == '__main__':
    main()
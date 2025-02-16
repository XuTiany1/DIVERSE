
"""
Question: I don't understand get_sequence_labels




"""






"""

OVERVIEW
Defines base classes and helper functions to process, label, and evaluate reasoning steps for problems

Concerning GSM8K arithmetic tasks, it contains two main object types:

1. BaseCase and GSM8KCase
        - These represent a complete question instance, including:
            the question text
            the ground truth answer
            multiple model-generated predicted answers

2. BaseExample and GSM8KExample
        - These encapsulate a single answer (or reasoning path) and extract details like individual reasoning steps and the final answer.

"""



"""

GSM8KCase
    - GSM8KCase extends BaseCase with GSM8K-specific logic for step labeling:

    - do_step_labeling:
        This method assigns correctness labels to each step in both the ground truth and predictions.
            - For the ground truth, it marks every step as correct.
            - For predictions that exactly match the final answer of the ground truth, all steps are labeled as correct.
            - For others, it calls the GSM8KExample.match() method to compare the sequence of steps against those from correct examples.
"""




"""

GSM8KExample
    - GSM8KExample extends BaseExample with methods specific to mathematical problem solving:
        
    - init_equations:
        This method uses a regular expression to extract equations from the solution text. 
        It looks for patterns like <<...>> followed by a numerical value and expects an equals sign (=) to identify valid equations.


    - get_final_answer:
        It extracts the final answer from the solution text.

        The answer is expected to appear after the marker "####".
        If the marker is missing, it returns a default value (stored in BaseExample.inf).

        
    - match:
        This static method compares a sequence of steps from a prediction with those from correct (positive) examples.

        It extracts numbers from each step (using get_answer) and uses a multiset to compare whether the set of numbers in the prediction is a subset of those in a correct example.
        If all extracted numbers from the predicted steps match a correct example’s numbers, the method returns 1 (indicating the steps are correct); otherwise, it returns 0.

        
    - get_sequence_labels:
        This function generates token-level labels for training a Named Entity Recognition (NER) model:

        1. It first labels the special [CLS] token based on whether the overall solution is correct.
        2. Then, it tokenizes each step by splitting on delimiters like >> and spaces.
        3. Tokens corresponding to the delimiters (like >>) are labeled as either "STEP-CORRECT" or "STEP-INCORRECT", depending on the step’s correctness.
        4. Other tokens are marked with "O".
        It then adds a separator token ("&&") and finally labels the question tokens with "O".


"""


"""
Helper Functions and Evaluation Metrics
clean_ans(s):
    - Cleans up the final answer by removing a trailing period and converting it to lowercase, ensuring consistency in answer matching.

Evaluation Functions (random_1_hit, recall_hit, voting_hit, weighted_voting_hit, verification_hit):
These functions compute different metrics to assess the quality of predictions:

    - Random 1-hit: Checks if a random prediction matches the ground truth.
    - Recall: Checks if at least one prediction among many matches the ground truth.
    - Voting & Weighted Voting: Simulate majority voting (or weighted voting by verifier score) among predictions.
    - Verification Hit: Uses the verifier's ranking to choose the best answer.


compute_top1_and_recall and compute_results:
These functions aggregate the evaluation metrics over all cases in the dataset, allowing you to assess how well your system is performing on GSM8K.

dedup and print_stat:
Utility functions for deduplication and printing statistics.

"""


"""

Summary for GSM8K
For GSM8K, this file:
    - Represents each question (GSM8KCase) and its answers (GSM8KExample):
        - Extracts step-by-step reasoning.
        - Marks correct steps when predictions match the ground truth.
        - Uses pattern matching on equations and final answer markers to decide correctness.

    - Provides methods for generating token-level labels (get_sequence_labels):
        - This is useful for training models (e.g., a verifier) to predict the correctness of each reasoning step.

    - Includes evaluation functions:
        - These functions help compute metrics like top-1 accuracy and recall to measure performance on GSM8K.

"""



#######################
# IMPORT LIBRARIES
#######################
import re
from tqdm import tqdm
from multiset import Multiset
from functools import lru_cache
import random
import json
import pdb
import torch
import torch.nn.functional as F
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
import time


"""
A BaseCase object represents one complete question, including:
    - The question itself
    - The correct ground-truth answer
    - A list of predicted answers (from the model)
    - Methods to assign correctness labels to predictions
"""
class BaseCase:
    def __init__(self, ground_truth, preds):
        self.question = ""
        self.ground_truth = ground_truth
        self.preds = preds
        self.correct_preds_num = 0.0


class GSM8KCase(BaseCase):
    def __init__(self, ground_truth, preds):
        super().__init__(ground_truth, preds)
        self.entailment_batch_size = 512

    #######################
    # Step Labeling for GSM8K
    #######################
    def do_step_labeling(self, model=None, tokenizer=None):
        
        # Mark ground truth label to be true
        self.ground_truth.is_correct = True
        for step in self.ground_truth.steps:
            self.ground_truth.step_labels[step] = 1

        # Store correct predictions (exact match)
        positive_preds = [self.ground_truth]
        for i, pred in enumerate(self.preds):
            if pred.get_final_answer() != BaseExample.inf and pred.get_final_answer() == self.ground_truth.get_final_answer():
                positive_preds.append(pred)

        # Compare all steps
        for i, pred in enumerate(self.preds):
            if pred.get_final_answer() != BaseExample.inf and pred.get_final_answer() == self.ground_truth.get_final_answer():
                pred.is_correct = True
                for step in pred.steps:
                    pred.step_labels[step] = 1  # Mark correct steps
            else:
                for k, step in enumerate(pred.steps):
                    ans = GSM8KExample.match(
                        pred.steps[:k+1],   # Steps so far
                        positive_preds,     # Ground truth steps
                        model=model,
                        tokenizer=tokenizer,
                    )
                    pred.step_labels[step] = ans    # Assign a correctness label


class TextEntailmentCase(BaseCase):
    def __init__(self, ground_truth, preds, entailment_batch_size=512):
        super().__init__(ground_truth, preds)
        self.entailment_results = {}
        self.entailment_batch_size = entailment_batch_size

    def do_step_labeling(self, model=None, tokenizer=None):
        # 将ground_truth标记为true
        self.ground_truth.is_correct = True
        for step in self.ground_truth.steps:
            self.ground_truth.step_labels[step] = 1

        # 先预存正样本集合
        positive_preds = [self.ground_truth]
        for i, pred in enumerate(self.preds):
            if pred.get_final_answer() != BaseExample.inf and pred.get_final_answer() == self.ground_truth.get_final_answer():
                positive_preds.append(pred)

        # 将所有待NLI的文本预存起来
        self.collect_entailment_texts(positive_preds)

        # print("Number of entailment result keys:", len(self.entailment_results.keys()))

        # 预处理所有NLI结果
        self.preprocess_entailment(model=model, tokenizer=tokenizer)

        # 再对所有样本的所有step打标签
        for i, pred in enumerate(self.preds):
            if pred.get_final_answer() != BaseExample.inf and pred.get_final_answer() == self.ground_truth.get_final_answer():
                pred.is_correct = True
                for step in pred.steps:
                    pred.step_labels[step] = 1
            else:
                for k, step in enumerate(pred.steps):
                    ans = TextEntailmentExample.match(
                        pred.steps[:k+1],
                        positive_preds,
                        model=model,
                        tokenizer=tokenizer,
                        entailment_result_dict=self.entailment_results,
                    )
                    pred.step_labels[step] = ans
    
    def collect_entailment_texts(self, positive_preds):
        for i, pred in enumerate(self.preds):
            if pred.get_final_answer() != BaseExample.inf and pred.get_final_answer() == self.ground_truth.get_final_answer():
                pass
            else:
                for pp in positive_preds:
                    for k, step in enumerate(pred.steps):
                        if k >= len(pp.steps):
                            continue
                        pp_step = pp.steps[k].strip()
                        text1 = f"premise: {pp_step} hypothesis: {step}"
                        text2 = f"premise: {step} hypothesis: {pp_step}"
                        self.entailment_results[text1] = -1
                        self.entailment_results[text2] = -1
    
    def preprocess_entailment(self, model, tokenizer):
        text_all = list(self.entailment_results.keys())
        text_batch, results_batch = [], []
        for i in range(0, len(text_all), self.entailment_batch_size):
            text_batch = text_all[i : min(len(text_all), i + self.entailment_batch_size)]
            batch_results = entailment_batch(text_batch, model, tokenizer)
            for sc in batch_results:
                results_batch.append(sc)
        for text, result in zip(text_batch, results_batch):
            self.entailment_results[text] = 1 if result else 0


"""
A BaseExample object represents one possible answer (either correct or predicted) to a given question.
It breaks the answer into steps and assigns correctness labels.

- Stores the solution content (content) and extracts step-by-step reasoning (steps).
- Keeps labels (step_labels) to indicate which steps are correct (1) or incorrect (0).
- Extracts the final answer (get_final_answer()) from a solution.
"""
class BaseExample:
    inf = "-99999999"
    
    def __init__(self, content):
        self.content = content.strip()      # Stores the solution text
        self.steps = self.get_steps()       # Extracts step-by-step reasoning
        self.step_labels = {}               # Stores labels for each step (1 = correct, 0 = incorrect)
        self.sequence_labels = []           # Stores labeled tokens for sequence labeling
        self.is_correct= False              # Whether the solution is correct

    # Only for GSM8K dataset use
    def init_equations(self):
        raise NotImplementedError

    # Extract steps from a solution
    def get_steps(self):
        return [x+"%%" if x != self.content.split("%%")[-1] else x for i, x in enumerate(self.content.split("%%"))]

    # Extract final answer from the solution
    def get_final_answer(self):
        ans = ""
        if "####" in self.content:
            ans = self.content.split("####")[-1].strip().replace("%%", "").replace(" ", "")
        else:
            ans = BaseExample.inf            # If no answer is found, return "-999999.."
        return clean_ans(ans)

    def label_to_string(self):
        return "".join(str(self.labels[k]) for k in self.labels.keys())


"""
Extend BaseExample to support mathematical problem-solving

- Extracts equations (init_equations()) from the solution.
- Checks correctness of individual steps (match()) by comparing against correct examples.
- Extracts final answers (get_answer()) using regex patterns.
- Generates labeled token sequences (get_sequence_labels()) for Named Entity Recognition (NER) tasks.
"""
class GSM8KExample(BaseExample):
    def __init__(self, content):
        super().__init__(content)
        self.equations = self.init_equations()      # Extract mathemtical equations
        self.verifier_score = 0.0 

    # Extract equations from the content
    def init_equations(self):
        return [x for x in re.findall("<<.+>>[0-9\.]+", self.content) if "=" in x]

    # Extract the answer from a solution step
    def get_step_answer(step):
        expression = re.findall("<<.+>>[0-9\.]+", step)
        if len(expression == 0):
            ans = BaseExample.inf
        else:
            ans = expression[-1].split(">>")[-1].strip()
        return clean_ans(ans)
    
    # Extract the final answer from a solution
    @staticmethod
    @lru_cache(maxsize=4096)
    def get_answer(s):
        ans = ""
        if "####" in s:
            ans = s.split("####")[-1].replace("%%", "").replace(" ", "").strip()
        else:
            expression = re.findall("<<.+>>[0-9\.]+", s)
            if len(expression) == 0:
                ans = GSM8KExample.inf
            else:
                ans = expression[-1].split(">>")[-1].strip()
        return clean_ans(ans)
    
    # Checks if a predicted step sequence matches any known correct example
        # If all numbers extracted from the predicted steps are found in a correct example, it is marked correct (returns 1).
        # If not, it is incorrect (returns 0).
    @staticmethod
    def match(steps, positive_examples, model=None, tokenizer=None):
        curr_set = Multiset([GSM8KExample.get_answer(x) for x in steps])
        for positive_example in positive_examples:
            golden_set = Multiset([GSM8KExample.get_answer(x) for x in positive_example.steps])
            
            # Remove invalude values
            if GSM8KExample.inf in curr_set:
                curr_set.remove(GSM8KExample.inf)
            if GSM8KExample.inf in golden_set:
                golden_set.remove(GSM8KExample.inf)

            # If there are no valid steps, assume incorrect
            if len(curr_set) == 0:
                return 0

            # If all steps are in the correct set, mark as correct
            if curr_set.issubset(golden_set):
                return 1
        # 0 if no match is found
        return 0
    """
    EXAMPLE to make the above function more clear

    Question:
    “Lisa has 3 apples. She buys 2 more. How many does she have now?”


    Correct Solution (ground truth):
    Step 1: Lisa has 3 apples.  # No computation yet
    Step 2: She buys 2 more.  # Still no computation
    Step 3: <<3 + 2>> 5  # Computation + final answer
    Final Answer: #### 5


    Incorrect Student Attempt:
    Step 1: Lisa starts with 3 apples.  
    Step 2: She buys <<2>> more.  
    Step 3: She now has <<3+2>> 4.  
    Final Answer: #### 4

    - Extracted numbers: {3, 2, 4}
    - Correct numbers: {3, 2, 5}
    - Since {3, 2, 4} is not a subset of {3, 2, 5}, match() returns 0.

    
    Correct Student Attempt:
    Step 1: She starts with 3 apples.  
    Step 2: She gets <<2>> more.  
    Step 3: She now has <<3 + 2>> 5.  
    Final Answer: #### 5

    - Extracted numbers: {3, 2, 5}
    - Correct numbers: {3, 2, 5}
    - {3, 2, 5} matches the correct answer, so match() returns 1.
    """


    
    # Create token-level sequence labels for NER training
    # Labels whether a step or answer is correct or not
    def get_sequence_labels(question, pred):
        sequence_labels = []

        # Label the CLS token (solution correctness)
        if pred.is_correct:
            sequence_labels.append(("[CLS]", "SOLUTION-CORRECT"))
        else:
            sequence_labels.append(("[CLS]", "SOLUTION-INCORRECT"))

        # Label step tokens
        for s in pred.steps:
            token_list = [x for x in re.split("(>>| )", s) if x != ' ']
            for token in token_list:
                if token == ">>":
                    if pred.step_labels[s] == 1:
                        sequence_labels.append((token, "STEP-CORRECT"))
                    else:
                        sequence_labels.append((token, "STEP-INCORRECT"))
                else:
                    sequence_labels.append((token, "O"))

        # add a split symbol
        sequence_labels.append(("&&", "O"))

        # Label question tokens
        for token in question.split(" "):
            sequence_labels.append((token, "O"))

        return sequence_labels
    
    """
    EXAMPLE TO UNDERSTAND BETTER the function above

    Let’s say we are training a machine learning model to learn which steps are correct or incorrect in problem-solving.

    Correct Solution:
    Step 1: Lisa has 3 apples.  
    Step 2: She buys <<2>> more.  
    Step 3: She now has <<3 + 2>> 5.  
    Final Answer: #### 5
    
    Labeled Output (for NER training):
    [CLS] SOLUTION-CORRECT
    Lisa O
    has O
    3 O
    apples O
    She O
    buys O
    << O
    2 STEP-CORRECT
    >> STEP-CORRECT
    more O
    She O
    now O
    has O
    << O
    3 STEP-CORRECT
    + STEP-CORRECT
    2 STEP-CORRECT
    >> STEP-CORRECT
    5 STEP-CORRECT
    #### SOLUTION-CORRECT
    && O
    Lisa O
    has O
    how O
    many O
    ? O

    

    INCORRECT SOLUTION
    Step 1: Lisa has 3 apples.  
    Step 2: She buys <<2>> more.  
    Step 3: She now has <<3 + 2>> 4.  
    Final Answer: #### 4

    Labeled Output:
    [CLS] SOLUTION-INCORRECT
    Lisa O
    has O
    3 O
    apples O
    She O
    buys O
    << O
    2 STEP-CORRECT
    >> STEP-CORRECT
    more O
    She O
    now O
    has O
    << O
    3 STEP-CORRECT
    + STEP-CORRECT
    2 STEP-CORRECT
    >> STEP-CORRECT
    4 STEP-INCORRECT
    #### SOLUTION-INCORRECT
    && O
    Lisa O
    has O
    how O
    many O
    ? O
    """



class TextEntailmentExample(BaseExample):
    def __init__(self, content):
        super().__init__(content)

    @staticmethod
    def match(steps, positive_examples, model, tokenizer, entailment_result_dict):
        for pp in positive_examples:
            if TextEntailmentExample.match_per_example(pp, steps, entailment_result_dict):
                return 1
        return 0
    
    @staticmethod
    def match_per_example(pp, steps, entailment_result_dict):
        for k, step in enumerate(steps):
            if k >= len(pp.steps):
                continue
            # print("step:", step)
            # print("pp.steps[k]:", pp.steps[k])
            pp_step = pp.steps[k].strip()
            text1 = f"premise: {step} hypothesis: {pp_step}"
            text2 = f"premise: {pp_step} hypothesis: {step}"
            if entailment_result_dict[text1] == 0 or entailment_result_dict[text2] == 0:
                # error_case = 'No, Christmas trees are not dissimilar to deciduous trees.%%Both Christmas trees and deciduous trees are types of trees.%%Both Christmas trees and deciduous trees have leaves.%%So the answer is no.#### no'
                # if error_case in text1 or error_case in text2:
                #     print("text1:", text1)
                #     print("text2:", text2)
                #     pdb.set_trace()
                return 0
        return 1

    def get_sequence_labels(question, pred):
        sequence_labels = []
        if pred.is_correct:
            sequence_labels.append(("[CLS]", "SOLUTION-CORRECT"))
        else:
            sequence_labels.append(("[CLS]", "SOLUTION-INCORRECT"))

        # add step tokens
        for s in pred.steps:
            token_list = [x for x in re.split("(%%| )", s) if x != ' ']
            for token in token_list:
                if token == "":
                    continue
                if token == "%%":
                    if pred.step_labels[s] == 1:
                        sequence_labels.append((token, "STEP-CORRECT"))
                    else:
                        sequence_labels.append((token, "STEP-INCORRECT"))
                else:
                    sequence_labels.append((token, "O"))

        # add a split symbol
        sequence_labels.append(("&&", "O"))

        # add question tokens
        for token in question.split(" "):
            sequence_labels.append((token, "O"))

        return sequence_labels


@torch.no_grad()
def entailment_batch(text, model, tokenizer):
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to("cuda")
    labels = torch.tensor([1] * len(text)).to("cuda")
    outputs = model(**inputs, labels=labels)
    logits = outputs.logits
    ans_list = torch.argmax(F.softmax(logits, dim=-1), dim=-1).tolist()
    ans_list = [x == model.config.label2id["ENTAILMENT"] for x in ans_list]
    return ans_list


@torch.no_grad()
def entailment(premise, hypothesis, model, tokenizer):
    text = f"premise: {premise} hypothesis: {hypothesis}"
    inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt").to(model.device)
    labels = torch.tensor([1]).to(model.device)
    outputs = model(**inputs, labels=labels)
    logits = outputs.logits
    ans = torch.argmax(F.softmax(logits, dim=-1)).item() == model.config.label2id["ENTAILMENT"]
    return ans


def convert_eval_sequences_to_cases(eval_sequences, pred_num_per_case, case_class, example_class):
    cases = []
    for i in range(0, len(eval_sequences), pred_num_per_case + 1):
        case = case_class("", [])
        # question, grount_truth = eval_sequences[i].split("&&")[0], eval_sequences[i].split("&&")[1]
        question, grount_truth = eval_sequences[i].split("&&")[1], eval_sequences[i].split("&&")[0]
        case.ground_truth = example_class(grount_truth)
        case.question = question
        for j in range(i+1, i+pred_num_per_case+1):
            # case.preds.append(GSM8KExample(eval_sequences[j].split("&&")[1]))
            case.preds.append(example_class(eval_sequences[j].split("&&")[0]))
        cases.append(case)
    # if example_class.__name__ == "TextEntailmentExample":
    #     cases = post_process_answer_clutrr(cases)
    return cases


def post_process_answer_clutrr_mapping(cases):
    print("before loading pipeline")
    classifier = pipeline("zero-shot-classification", device=0)
    print("after loading pipeline")
    print("post processing")
    candidate_labels = ['sister', 'son', 'aunt', 'granddaughter', 'father', 'grandfather', 'grandmother', 'mother-in-law', 'uncle', 'niece', 'mother', 'brother', 'daughter', 'nephew', 'grandson', 'son-in-law', 'father-in-law', 'daughter-in-law']
    for case_idx, case in tqdm(enumerate(cases)):
        gt_ans = case.ground_truth.get_final_answer()
        # skip StrategyQA task
        if gt_ans == "yes" or gt_ans == "no":
            break
        for pred in case.preds:
            pred_ans = pred.get_final_answer()
            if pred_ans != BaseExample.inf and pred_ans != gt_ans:
                outputs = classifier(pred_ans, candidate_labels)
                logits = outputs["scores"]
                labels = outputs["labels"]
                candidate_index = np.argmax(logits)
                most_similar_answer = labels[candidate_index]
                body = pred.content.split("####")[0]
                pred.content = body + "####" + most_similar_answer
                # pdb.set_trace()
    return cases
            

def post_process_answer_clutrr_cutoff(cases):
    candidate_labels = ['sister', 'son', 'aunt', 'granddaughter', 'father', 'grandfather', 'grandmother', 'mother-in-law', 'uncle', 'niece', 'mother', 'brother', 'daughter', 'nephew', 'grandson', 'son-in-law', 'father-in-law', 'daughter-in-law']
    for case_idx, case in tqdm(enumerate(cases)):
        gt_ans = case.ground_truth.get_final_answer()
        # skip StrategyQA task
        if gt_ans == "yes" or gt_ans == "no":
            break
        for pred in case.preds:
            pred_ans = pred.get_final_answer()
            if pred_ans not in candidate_labels:
                body = pred.content.split("####")[0]
                pred.content = body + "####" + BaseExample.inf
    return cases


def random_1_hit(gt_ans, preds):
    idx = random.randint(0, len(preds)-1)
    # random 1 acc
    pred0_ans = preds[idx].get_final_answer()
    return 1 if pred0_ans == gt_ans else 0


def recall_hit(gt_ans, preds):
    for pred in preds:
        if pred.get_final_answer() == gt_ans:
            return 1
    return 0


def voting_hit(gt_ans, preds):
    # voting acc
    answers = {}
    for pred in preds:
        if pred.get_final_answer() not in answers:
            answers[pred.get_final_answer()] = 0
        answers[pred.get_final_answer()] += 1
    answers = sorted(answers.items(), key=lambda x : x[1], reverse=True)
    for i in range(len(answers)):
        ans, ans_cnt = answers[i][0], answers[i][1]
        if ans != GSM8KExample.inf:
            return 1 if ans == gt_ans else 0
    return 0


def weighted_voting_hit(gt_ans, preds):
    # voting acc
    answers = {}
    for pred in preds:
        if pred.get_final_answer() not in answers:
            answers[pred.get_final_answer()] = 0
        answers[pred.get_final_answer()] += pred.verifier_score
    answers = sorted(answers.items(), key=lambda x : x[1], reverse=True)
    for i in range(len(answers)):
        ans, ans_cnt = answers[i][0], answers[i][1]
        if ans != GSM8KExample.inf:
            return 1 if ans == gt_ans else 0
    return 0


def verification_hit(gt_ans, preds):
    preds = sorted(preds, key=lambda x : x.verifier_score, reverse=True)
    for pred in preds:
        ans = pred.get_final_answer()
        if ans != GSM8KExample.inf:
            return 1 if ans == gt_ans else 0
    return 0


def compute_top1_and_recall(data, rand_k=100):
    total_random_hit_cnt = 0
    total_vote_cnt = 0
    total_recall_cnt = 0
    for i, x in enumerate(data):
        gt_ans = x.ground_truth.get_final_answer()
        slice = x.preds if rand_k >= len(x.preds) else random.sample(x.preds, rand_k)
        
        total_random_hit_cnt += random_1_hit(gt_ans, slice)
        total_vote_cnt += voting_hit(gt_ans, slice)
        total_recall_cnt += recall_hit(gt_ans, slice)
    result = {
        "random_top1": total_random_hit_cnt / len(data), 
        "voting_top1_accuracy": total_vote_cnt / len(data),
        "recall": total_recall_cnt / len(data),
    }
    return result


def compute_results(data, rand_k=100):
    total_random_hit_cnt = 0
    total_recall_cnt = 0
    total_vote_cnt = 0
    total_weighted_vote_cnt = 0
    total_verification_cnt = 0
    for i, x in enumerate(data):
        gt_ans = x.ground_truth.get_final_answer()
        slice = x.preds if rand_k == len(x.preds) else random.sample(x.preds, rand_k)
        
        total_random_hit_cnt += random_1_hit(gt_ans, slice)
        total_vote_cnt += voting_hit(gt_ans, slice)
        total_recall_cnt += recall_hit(gt_ans, slice)
        total_weighted_vote_cnt += weighted_voting_hit(gt_ans, slice)
        total_verification_cnt += verification_hit(gt_ans, slice)
    result = {
        "random_top1": total_random_hit_cnt / len(data), 
        f"recall@{rand_k}": total_recall_cnt / len(data),
        f"verifier_top1_accuracy@{rand_k}": total_verification_cnt / len(data),
        f"voting_top1_accuracy@{rand_k}": total_vote_cnt / len(data),
        f"weighted_voting_top1_accuracy@{rand_k}": total_weighted_vote_cnt / len(data),
    }
    return result


def compute_results_avg(data, rand_k=100, repeat_time=5):
    sum_result_dict = {
        "random_top1": 0, 
        f"recall@{rand_k}": 0,
        f"verifier_top1_accuracy@{rand_k}": 0,
        f"voting_top1_accuracy@{rand_k}": 0,
        f"weighted_voting_top1_accuracy@{rand_k}": 0,
    }
    for i in tqdm(range(repeat_time)):
        for k in sum_result_dict:
            result_dict = compute_results(data, rand_k=rand_k)
            sum_result_dict[k] += result_dict[k]
    for k in sum_result_dict:
        sum_result_dict[k] = sum_result_dict[k] / repeat_time if repeat_time != 1 else sum_result_dict[k]
        sum_result_dict[k] = round(sum_result_dict[k], 8)
    return sum_result_dict
    

def dedup(li):
    s = set()
    new_li = []
    for x in li:
        if str(x) not in s:
            new_li.append(x)
            s.add(str(x))
    return new_li


def print_stat(data):
    cnt = 0
    for x in data:
        if x["output"] == "correct":
            cnt += 1
    print(cnt, len(data) - cnt, len(data))


def clean_ans(s):
    s = str(s)
    if s and len(s) > 0 and s[-1] == '.':
        s = s[:-1]
    return s.lower()  # for CLUTRR and strategyQA use
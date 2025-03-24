import argparse
import os
import json
from tasks.MATH import MATH
import evaluate


# Set up arguments
args = argparse.Namespace(
    #languages = ['sna', 'sot', 'swa', 'twi', 'vai', 'wol', 'xho', 'yor', 'zul', 'amh', 'ewe', 'hau', 'ibo', 'kin', 'lin', 'lug']
    #languages = ['bn', 'de', 'en', 'es', 'ja', 'ru', 'sw', 'te', 'th', 'zh']
    languages = ['fr']
)


exact_match_metric = evaluate.load("exact_match")


for lang in args.languages:

    total_count = 0

    no_reasoning_correct_count = 0
    single_cot__path_one_correct_count = 0
    single_cot__path_two_correct_count = 0
    single_cot__path_three_correct_count = 0
    single_cot__path_four_correct_count = 0
    single_cot__path_five_correct_count = 0
    verifier_correct_count = 0
    voting_verifier_correct_count = 0
    self_consistency_correct_count = 0

    no_reasoning_preds = []
    single_cot__path_one_preds = []
    single_cot__path_two_preds = []
    single_cot__path_three_preds = []
    single_cot__path_four_preds = []
    single_cot__path_five_preds = []
    verifier_preds = []
    voting_verifier_preds = []
    voting_only_preds = []
    ground_truth_refs = []

    args.lang = lang

    data_file_path = f"multilingual_reasoning_dataset/{lang}_dataset.jsonl"

    # Create folders
    run_folder = os.path.join("multilingual_evaluation_result", lang)
    os.makedirs(run_folder, exist_ok=True)
    no_reasoning_folder = os.path.join(run_folder, "no_reasoning")
    os.makedirs(no_reasoning_folder, exist_ok=True)
    single_cot_folder = os.path.join(run_folder, "single_cot")
    os.makedirs(single_cot_folder, exist_ok=True)
    verifier_only_folder = os.path.join(run_folder, "verifier")
    os.makedirs(verifier_only_folder, exist_ok=True)
    voting_only_folder = os.path.join(run_folder, "voting")
    os.makedirs(voting_only_folder, exist_ok=True)
    voting_verifier_folder = os.path.join(run_folder, "voting_verifier")
    os.makedirs(voting_verifier_folder, exist_ok=True)


    # Create logs
    overall_result_log_path = os.path.join(run_folder, "overall_results.log")
    verifier_error_log_path = os.path.join(verifier_only_folder, "verifier_error.log")
    voting_error_log_path = os.path.join(voting_only_folder, "voting_error.log")
    voting_verifier_log_path = os.path.join(voting_verifier_folder, "voting_verifier_error.log")
    chain_of_thought_log_path = os.path.join(single_cot_folder, "single_cot.log")
    processing_error_log_path = os.path.join(run_folder, "reading_error.log")

    overall_result_log = open(overall_result_log_path, "w", encoding="utf-8")
    verifier_error_log = open(verifier_error_log_path, "w", encoding="utf-8")
    voting_error_log = open(voting_error_log_path, "w", encoding="utf-8")
    voting_verifier_error_log = open(voting_verifier_log_path, "w", encoding="utf-8")
    single_cot_error_log = open(chain_of_thought_log_path, "w", encoding="utf-8")
    processing_error_log = open(processing_error_log_path, "w", encoding="utf-8")



    with open(data_file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for idx, line in enumerate(lines):

        total_count+= 1

        record = json.loads(line)

        question = record["question"]
        ground_truth = record["ground_truth"]
        direct_answer = record["direct_answer"]
        reasoning_paths = record["reasoning_paths"]

        try:
            gt_val = float(ground_truth)
        except ValueError:
            gt_val = None


        ans_prob_dict = {}
        list_of_cot_ans = []

        for item in reasoning_paths:
            ans = item["answer"]
            prob = item["probability"]
            list_of_cot_ans.append(ans)

            if ans not in ans_prob_dict:
                ans_prob_dict[ans] = []
            ans_prob_dict[ans].append(prob)

        try:
            # Verifier: key with the highest single probability
            verifier_only_ans = max(ans_prob_dict, key=lambda k: max(ans_prob_dict[k]))
            # Voting Verifier: key with the highest sum of probabilities
            voting_verifier_ans = max(ans_prob_dict, key=lambda k: sum(ans_prob_dict[k]))
            # Voting: key with the highest number of entries
            voting_only_ans = max(ans_prob_dict, key=lambda k: len(ans_prob_dict[k]))

        except Exception as e:
            error_log_entry = (
                "----------------------\n"
                f"Exception: {e}\n"
                f"Problem: {question}\n"
                f"Record: {record}\n"
                f"Verifier probability: {ans_prob_dict}\n"
                f"List of answers: {list_of_cot_ans}\n"
                "----------------------\n"
            )
            processing_error_log.write(error_log_entry)
            processing_error_log.flush()   

        # Increament count if answer matches ground truth
        if verifier_only_ans is not None and gt_val is not None and float(verifier_only_ans)==gt_val:
            verifier_correct_count += 1
        if voting_verifier_ans is not None and gt_val is not None and float(voting_verifier_ans)==gt_val:
            voting_verifier_correct_count += 1
        if voting_only_ans is not None and gt_val is not None and float(voting_only_ans)==gt_val:
            self_consistency_correct_count += 1
        if list_of_cot_ans[0] is not None and gt_val is not None and float(list_of_cot_ans[0])==gt_val:
            single_cot__path_one_correct_count += 1
        if list_of_cot_ans[1] is not None and gt_val is not None and float(list_of_cot_ans[1])==gt_val:
            single_cot__path_two_correct_count += 1
        if list_of_cot_ans[2] is not None and gt_val is not None and float(list_of_cot_ans[2])==gt_val:
            single_cot__path_three_correct_count += 1
        if list_of_cot_ans[3] is not None and gt_val is not None and float(list_of_cot_ans[3])==gt_val:
            single_cot__path_four_correct_count += 1
        if list_of_cot_ans[4] is not None and gt_val is not None and float(list_of_cot_ans[4])==gt_val:
            single_cot__path_five_correct_count += 1

        verifier_acc = verifier_correct_count / total_count
        voting_verifier_acc = voting_verifier_correct_count / total_count
        self_consistency_acc = self_consistency_correct_count / total_count
        single_cot__path_one_acc = single_cot__path_one_correct_count / total_count
        single_cot__path_two_acc = single_cot__path_two_correct_count / total_count
        single_cot__path_three_acc = single_cot__path_three_correct_count / total_count
        single_cot__path_four_acc = single_cot__path_four_correct_count / total_count
        single_cot__path_five_acc = single_cot__path_five_correct_count / total_count

        total_prob_sum = sum(sum(probs) for probs in ans_prob_dict.values())
        if ground_truth in ans_prob_dict and total_prob_sum > 0:
            weighted_prob_correct = sum(ans_prob_dict[ground_truth]) / total_prob_sum
        else:
            weighted_prob_correct = 0

        # ----- Exact Match Evaluation using evaluate library -----
        # Convert predictions to strings (or empty string if None)
        verifier_pred_str = str(verifier_only_ans) if verifier_only_ans is not None else ""
        voting_verifier_pred_str = str(voting_verifier_ans) if voting_verifier_ans is not None else ""
        voting_only_pred_str = str(voting_only_ans) if voting_only_ans is not None else ""
        single_cot_one_pred_str = str(list_of_cot_ans[0]) if list_of_cot_ans[0] is not None else ""
        single_cot_two_pred_str = str(list_of_cot_ans[1]) if list_of_cot_ans[1] is not None else ""
        single_cot_three_pred_str = str(list_of_cot_ans[2]) if list_of_cot_ans[2] is not None else ""
        single_cot_four_pred_str = str(list_of_cot_ans[3]) if list_of_cot_ans[3] is not None else ""
        single_cot_five_pred_str = str(list_of_cot_ans[4]) if list_of_cot_ans[4] is not None else ""
        direc_answer_pred_str = str(direct_answer) if direct_answer is not None else ""
        gt_str = str(ground_truth)

        # Append to lists for the metric evaluation
        verifier_preds.append(verifier_pred_str)
        voting_only_preds.append(voting_only_pred_str)
        voting_verifier_preds.append(voting_verifier_pred_str)
        single_cot__path_one_preds.append(single_cot_one_pred_str)
        single_cot__path_two_preds.append(single_cot_two_pred_str)
        single_cot__path_three_preds.append(single_cot_three_pred_str)
        single_cot__path_four_preds.append(single_cot_four_pred_str)
        single_cot__path_five_preds.append(single_cot_five_pred_str)
        no_reasoning_preds.append(direc_answer_pred_str)
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
        current_em_single_path_one = exact_match_metric.compute(
            predictions=single_cot__path_one_preds, references=ground_truth_refs
        )["exact_match"]
        current_em_single_path_two = exact_match_metric.compute(
            predictions=single_cot__path_two_preds, references=ground_truth_refs
        )["exact_match"]
        current_em_single_path_three = exact_match_metric.compute(
            predictions=single_cot__path_three_preds, references=ground_truth_refs
        )["exact_match"]
        current_em_single_path_four = exact_match_metric.compute(
            predictions=single_cot__path_four_preds, references=ground_truth_refs
        )["exact_match"]
        current_em_single_path_five = exact_match_metric.compute(
            predictions=single_cot__path_five_preds, references=ground_truth_refs
        )["exact_match"]
        current_em_no_reasoning = exact_match_metric.compute(
            predictions=no_reasoning_preds, references=ground_truth_refs
        )["exact_match"]


        log_entry = (
            "----------------------\n"
            f"Problem: {question}\n"
            f"Record: {record}\n"
            f"Direct answer prediction / Ground Truth: {direct_answer} / {ground_truth}\n"
            f"Single cot 1 / Ground Truth: {list_of_cot_ans[0]} / {ground_truth}\n"
            f"Single cot 2 / Ground Truth: {list_of_cot_ans[1]} / {ground_truth}\n"
            f"Single cot 3 / Ground Truth: {list_of_cot_ans[2]} / {ground_truth}\n"
            f"Single cot 4 / Ground Truth: {list_of_cot_ans[3]} / {ground_truth}\n"
            f"Single cot 5 / Ground Truth: {list_of_cot_ans[4]} / {ground_truth}\n"
            f"Verifier prediction / Ground Truth: {verifier_only_ans} / {ground_truth}\n"
            f"Voting Verifier prediction / Ground Truth: {voting_verifier_ans} / {ground_truth}\n"
            f"Voting (Self-Consistency) prediction / Ground Truth: {voting_only_ans} / {ground_truth}\n"
            f"Weighted Verifier Probability: {weighted_prob_correct:.2%}\n"
            f"Verifier Correct: {verifier_correct_count}/{total_count} ({verifier_acc:.2%})\n"
            f"Voting Verifier Correct: {voting_verifier_correct_count}/{total_count} ({voting_verifier_acc:.2%})\n"
            f"Self-Consistency Correct: {self_consistency_correct_count}/{total_count} ({self_consistency_acc:.2%})\n"
            f"Single CoT 1 Correct: {single_cot__path_one_correct_count}/{total_count} ({single_cot__path_one_acc:.2%})\n"
            f"Single CoT 2 Correct: {single_cot__path_two_correct_count}/{total_count} ({single_cot__path_two_acc:.2%})\n"
            f"Single CoT 3 Correct: {single_cot__path_three_correct_count}/{total_count} ({single_cot__path_three_acc:.2%})\n"
            f"Single CoT 4 Correct: {single_cot__path_four_correct_count}/{total_count} ({single_cot__path_four_acc:.2%})\n"
            f"Single CoT 5 Correct: {single_cot__path_five_correct_count}/{total_count} ({single_cot__path_five_acc:.2%})\n"
            f"Exact Match Accuracy - Verifier: {current_em_verifier:.2%}, "
            f"Voting Verifier: {current_em_voting_verifier:.2%}, "
            f"Voting Only: {current_em_voting_only:.2%}, "
            f"CoT 1: {current_em_single_path_one:.2%}, "
            f"CoT 2: {current_em_single_path_two:.2%}, "
            f"CoT 3: {current_em_single_path_three:.2%}, "
            f"CoT 4: {current_em_single_path_four:.2%}, "
            f"CoT 5: {current_em_single_path_five:.2%}, "
            f"No Reasoning: {current_em_no_reasoning:.2%}\n"
            "----------------------\n"
        )


        # Write the combined log entry.
        overall_result_log.write(log_entry)
        overall_result_log.flush() 


        # If a method's prediction is wrong, append the entry to its corresponding error log.
        if verifier_only_ans is None or (gt_val is not None and float(verifier_only_ans) != gt_val):
            verifier_error_log.write(log_entry)
        if voting_verifier_ans is None or (gt_val is not None and float(voting_verifier_ans) != gt_val):
            voting_verifier_error_log.write(log_entry)
        if voting_only_ans is None or (gt_val is not None and float(voting_only_ans) != gt_val):
            voting_error_log.write(log_entry)


    # Close all log files.
    overall_result_log.close()
    verifier_error_log.close()
    voting_verifier_error_log.close()
    voting_error_log.close()



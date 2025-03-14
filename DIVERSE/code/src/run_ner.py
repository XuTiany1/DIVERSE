
""" Fine-tuning the library models for named entity recognition on CoNLL-2003. """


import sys
import os

# FORCE working directory change
correct_working_directory = "/home/mila/x/xut/github/DIVERSE/DIVERSE/code/src"
if os.getcwd() != correct_working_directory:
    print(f"DEBUG: Changing working directory from {os.getcwd()} to {correct_working_directory}")
    os.chdir(correct_working_directory)

# Confirm it's working
print("DEBUG: Final working directory:", os.getcwd())
print("DEBUG: Python interpreter:", sys.executable)
import torch
from torch.utils.data import Subset  # Import Subset for selecting part of dataset



#########################
# IMPORT LIBRARIES
#########################
import sys
import logging
import os
from dataclasses import dataclass, field
from importlib import import_module
from typing import Dict, List, Optional, Tuple
import pdb
import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
import scipy
from verifier_metrics import VerifierMetrics
import utils_io
import shutil

import transformers
from transformers import (
    AutoConfig,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import is_main_process
from utils_ner import Split, TokenClassificationDataset, TokenClassificationTask
from deberta_model import DebertaV2ForTokenClassification
import pdb

logger = logging.getLogger(__name__)


#########################
# ARGUMENT PARSING: ModelArguments
#########################
@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    dataset_name: str = field(
        metadata={"help": "Name of the dataset to be run"}
    )
    previous_run_dir: Optional[str] = field(
        default=None, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    previous_run_epoch: Optional[int] = field(
        default=1, metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    task_type: Optional[str] = field(
        default="NER", metadata={"help": "Task type to fine tune in training (e.g. NER, POS, etc)"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    use_fast: bool = field(default=False, metadata={"help": "Set this flag to use fast tokenization."})
    # If you want to tweak more attributes on your tokenizer, you should do it in a distinct script,
    # or just modify its tokenizer_config.json.
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )



#########################
# ARGUMENT PARSING: DataTrainingArguments
#########################
@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    train_data: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    test_data: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    data_labels: Optional[str] = field(
        default="labels.txt",
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=512,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    alpha: Optional[float] = field(
        default=0.0, metadata={"help": "help"}
    )



'''

RECAP:
ModelArguments:
    - Purpose: Stores model/tokenizer-related settings	
    - What it does: Passed to AutoModelForTokenClassification and AutoTokenizer

DataTrainingArguments:
    - Purpose: Stores dataset paths and preprocessing settings	
    - What it does: Used to load and preprocess dataset (TokenClassificationDataset)

Trainer:
    - Handles training & evaluation	
    - Uses these arguments to set up training loops

'''



def main():

    # handle command-line arguments.
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))

    # parses arguments either json file or commandline
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()


    #import torch
    #import torch.distributed as dist
    #if training_args.local_rank != -1 and not dist.is_initialized():
    #    dist.init_process_group(backend="nccl", init_method="env://")


    # Prevents overwriting an existing model unless --overwrite_output_dir is set.
    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Dynamically imports the task module
    module = import_module("tasks")
    try:
        token_classification_task_clazz = getattr(module, model_args.task_type)
        token_classification_task: TokenClassificationTask = token_classification_task_clazz()
    except AttributeError:
        raise ValueError(
            f"Task {model_args.task_type} needs to be defined as a TokenClassificationTask subclass in {module}. "
            f"Available tasks classes are: {TokenClassificationTask.__subclasses__()}"
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    # Configure huggingface logger
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
        transformers.utils.logging.enable_default_handler()
        transformers.utils.logging.enable_explicit_format()
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Prepare CONLL-2003 task, load the dataset 
    labels = token_classification_task.get_labels(data_args.data_labels)
    label_map: Dict[int, str] = {i: label for i, label in enumerate(labels)}
    num_labels = len(labels)

    # Load pretrained model and tokenizer from checkpoints
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.

    if model_args.previous_run_dir is not None:
        ckpt_path_list = [x for x in os.listdir(model_args.previous_run_dir) if "checkpoint" in x]
        ckpt_path_list = sorted(ckpt_path_list, key=lambda x : int(x.split("-")[1]))
        load_model_dir = ckpt_path_list[model_args.previous_run_epoch - 1]  # index starts from 0
        model_args.model_name_or_path = os.path.join(model_args.previous_run_dir, load_model_dir)
    # Loading model configuration
    
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        num_labels=num_labels,
        id2label=label_map,
        label2id={label: i for i, label in enumerate(labels)},
        cache_dir=model_args.cache_dir,
    )

    
    # pdb.set_trace()

    # Configure task-specific parameters
    config.task_specific_params = {}
    config.task_specific_params["solution_correct_loss_weight"] = 1.0
    config.task_specific_params["solution_incorrect_loss_weight"] = 1.0
    config.task_specific_params["step_correct_loss_weight"] = data_args.alpha
    config.task_specific_params["step_incorrect_loss_weight"] = data_args.alpha
    config.task_specific_params["other_label_loss_weight"] = 0.0

    print("alpha:", data_args.alpha)
    print("alpha:", config.task_specific_params["step_correct_loss_weight"])

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast
        # additional_special_tokens=["&&"]  # This ensures the "&&" delimiter is preserved.
    )
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/mdeberta-v3-base", additional_special_tokens=["&&"])

    # Load pre-trained model
    model = DebertaV2ForTokenClassification.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
    )
    
    # data_dir = data_args.train_data.replace("train.txt", "")  # for debug use
    # Load dataset
    data_dir = os.path.join(training_args.output_dir, "data/")
    print("[data_dir]:", data_dir)
    os.makedirs(data_dir, exist_ok=True)

    shutil.copy(utils_io.get_file(data_args.train_data), data_dir)
    print(f"train file copied to: {data_dir}")
    shutil.copy(utils_io.get_file(data_args.test_data), data_dir + "dev.txt")
    print(f"dev file copied to: {data_dir}")
    shutil.copy(utils_io.get_file(data_args.test_data), data_dir)
    print(f"test file copied to: {data_dir}")
    shutil.copy(utils_io.get_file(data_args.data_labels), data_dir)
    print(f"labels file copied to: {data_dir}")

    # Load the datasets
    train_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.train,
        )
        if training_args.do_train
        else None
    )
    eval_dataset = (
        TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.dev,
        )
        if training_args.do_eval
        else None
    )

    eval_dataset = [eval_dataset[0]]

    train_dataset = [train_dataset[0]]

    # save the texual sequences of eval dataset
    eval_sequences = [tokenizer.decode(x.input_ids) for x in eval_dataset]

    # Check if the first sequence contains "&&"
    if "&&" not in eval_sequences[0]:
        raise ValueError("The expected '&&' separator was not found in the decoded eval sequence. "
                        "Ensure your preprocessing step produces the correct delimiter.")


    first_test_case_question = eval_sequences[0].split("&&")[-1].strip()
    pred_num_per_case = 0
    for i, seq in enumerate(eval_sequences[1:]):
        if seq.split("&&")[-1].strip() == first_test_case_question:
            pred_num_per_case += 1
        else:
            break
    print("pred_num_per_case:", pred_num_per_case)
    

    def align_predictions(predictions: np.ndarray, label_ids: np.ndarray) -> Tuple[List[int], List[int]]:
        preds = np.argmax(predictions, axis=2)

        batch_size, seq_len = preds.shape

        out_label_list = [[] for _ in range(batch_size)]
        preds_list = [[] for _ in range(batch_size)]

        for i in range(batch_size):
            for j in range(seq_len):
                if j == 1: # only pick the second index
                # if label_ids[i, j] != nn.CrossEntropyLoss().ignore_index:
                    out_label_list[i].append(label_map[label_ids[i][j]])
                    preds_list[i].append(label_map[preds[i][j]])

        # preds_list: Model-predicted labels
        # out_label_list: True labels
        return preds_list, out_label_list

    def get_solution_logits(predictions: np.ndarray):
        scores = []
        for i in range(predictions.shape[0]):
            solution_correct_index = config.label2id["SOLUTION-CORRECT"]
            score = scipy.special.softmax(predictions[i][1])[solution_correct_index].item()

            scores.append(score)
        return scores

    # gsm8k_metric = datasets.load_metric("./gsm8k_verifier_metrics")
    metric = VerifierMetrics(
        eval_sequences=eval_sequences,
        pred_num_per_case=pred_num_per_case,
        dataset_name=model_args.dataset_name,
    )

    def compute_metrics(p: EvalPrediction) -> Dict:
        scores = get_solution_logits(p.predictions)
        return metric.compute(predictions=scores, references=scores)

    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8) if training_args.fp16 else None

    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=data_collator,
    )

    # Training
    if training_args.do_train:
        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        result = trainer.evaluate()

        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        if trainer.is_world_process_zero():
            with open(output_eval_file, "w") as writer:
                logger.info("***** Eval results *****")
                for key, value in result.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

            results.update(result)

    # Predict
    if training_args.do_predict:
        test_dataset = TokenClassificationDataset(
            token_classification_task=token_classification_task,
            data_dir=data_dir,
            tokenizer=tokenizer,
            labels=labels,
            model_type=config.model_type,
            max_seq_length=data_args.max_seq_length,
            overwrite_cache=data_args.overwrite_cache,
            mode=Split.test,
        )

        predictions, label_ids, metrics = trainer.predict(test_dataset)
        preds_list, _ = align_predictions(predictions, label_ids)

        output_test_results_file = os.path.join(training_args.output_dir, "test_results.txt")
        if trainer.is_world_process_zero():
            with open(output_test_results_file, "w") as writer:
                for key, value in metrics.items():
                    logger.info("  %s = %s", key, value)
                    writer.write("%s = %s\n" % (key, value))

        # Save predictions
        output_test_predictions_file = os.path.join(training_args.output_dir, "test_predictions.txt")
        if trainer.is_world_process_zero():
            with open(output_test_predictions_file, "w") as writer:
                with open(os.path.join(data_dir, "test.txt"), "r") as f:
                    token_classification_task.write_predictions_to_file(writer, f, preds_list)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()

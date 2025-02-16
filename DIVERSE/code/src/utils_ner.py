""" Named entity recognition fine-tuning: utilities to work with CoNLL-2003 task. """


"""

Question:

1. I don't understand what the InputFeatures dataclass is?

The InputFeatures dataclass is essentially a container for all the numerical data that a transformer model needs to process an example.
 When you work with textual data, models like BERT or RoBERTa cannot operate directly on strings—they need numbers. 
 InputFeatures stores these numbers along with other necessary information.

For each example (or reasoning path), InputFeatures typically contains:
    - input_ids: A list of integers that represent the tokens of your input text, 
    obtained by converting each token (like a word or subword) to its corresponding 
    numerical ID from the tokenizer’s vocabulary.

    - attention_mask: A list (of the same length as input_ids) where each element is 1 
    if the token is a real token and 0 if it is padding. This tells the model which tokens 
    to pay attention to.

    - token_type_ids (optional): For models that use them (like BERT), this indicates the 
    segment of the input (e.g., first sentence vs. second sentence). It’s not always needed.

    - label_ids (optional): A list of integers representing the label assigned to each token 
    (for token classification tasks). 
    In our GSM8K context, these labels indicate whether a token 
    (or more precisely, its associated reasoning step) is correct (e.g., "STEP-CORRECT") or incorrect (e.g., "STEP-INCORRECT") or not applicable ("O").



    

 2. What Does convert_examples_to_features Do?

 This function transforms raw input examples (which are instances of InputExample) 
 into model-ready numerical representations (instances of InputFeatures). 
 Let’s break it down with a GSM8K example.

 

    Step 1: Start with an InputExample:

        # Example content produced for a GSM8K reasoning path:
        example = InputExample(
            guid="1",
            words=["[CLS]", "Lisa", "has", "3", "apples", "She", "buys", "<<", "2", ">>", "more", "####", "5", "&&", "Lisa", "has", "how", "many", "?" ],
            labels=["SOLUTION-CORRECT", "O", "O", "O", "O", "O", "O", "STEP-CORRECT", "STEP-CORRECT", "STEP-CORRECT", "O", "SOLUTION-CORRECT", "STEP-CORRECT", "O", "O", "O", "O", "O", "O"]
        )


    Step 2: Tokenization and Label Alignment:

        The function loops through each word and:
            Tokenizes the word:
                Some words might be split into multiple sub-tokens 
                (e.g., "apples" might remain as "apples", but a complex word might be split into "app" and "##les").

            Assigns labels:
                The original label is kept for the first sub-token, and all additional sub-tokens get a placeholder label 
                (usually a padding label, e.g., -100), so that only the first token of each original word contributes to the loss during training.


            Adding Special Tokens:
                The function then adds special tokens required by the model:
                    [CLS] token: Added at the beginning (or at the end for some models) to represent the whole sequence.
                    [SEP] token: Added to mark the end of a sentence (or to separate segments).

                    
            Converting Tokens to Input IDs:
                Using the tokenizer’s vocabulary, the list of tokens is converted into input_ids. 
                For example, "[CLS]" might be converted to an ID like 101, "Lisa" might become 1234, and so on.

                         
            Creating the Attention Mask:
                The function creates an attention_mask that has a value of 1 for every real token (from your text) 
                and 0 for any padding that is added to reach a fixed sequence length.
     
                                      
            Padding or Truncating:
                The function ensures that the final sequences (input_ids, attention_mask, token_type_ids, and label_ids) 
                are all exactly of length max_seq_length by padding (adding zeros or a pad token) or truncating (cutting off extra tokens).

                
            Returning InputFeatures:
                Finally, it packages these lists into an InputFeatures object.
                    
                
A Concrete Example
Suppose our GSM8K example above after tokenization (assuming no word is split further for simplicity) and adding special tokens becomes:

Tokens:
["[CLS]", "Lisa", "has", "3", "apples", "She", "buys", "<<", "2", ">>", "more", "####", "5", "&&", "Lisa", "has", "how", "many", "?", "[SEP]"]

input_ids:
A corresponding list of integers (e.g., [101, 1234, 2005, ...])

attention_mask:
A list of 1’s with a length equal to the tokens (e.g., [1, 1, 1, 1, ..., 1])

label_ids:
For each token, numerical labels might be assigned based on a mapping, for example:
    "SOLUTION-CORRECT" → 1
    "STEP-CORRECT" → 2
    "O" → 0
    Padding tokens → -100

So you might have something like [1, -100, 0, 0, 0, 0, 0, 0, 2, 2, 2, 0, 1, 2, 0, 0, 0, 0, 0, 0, -100]
                


"""




"""


Summary for GSM8K Applications
Even though utils_ner.py was designed with CoNLL-2003 in mind, its structure is directly applicable if you want to fine-tune a model on the sequence-labeling task derived from GSM8K reasoning steps:

Data Structures:
- Use InputExample and InputFeatures to represent the tokenized reasoning paths and their labels.

Conversion and Labeling:
- The abstract methods and convert_examples_to_features function help convert your GSM8K output (formatted as sequences with labels from utils.py) into the right format for model training.

Dataset Integration:
- The provided dataset classes allow you to integrate this processed data into your model training pipeline.

In short, these utilities help transform your GSM8K reasoning step labels into a form suitable for training NER models, which in turn can be used to further refine or evaluate the step-level correctness of your reasoning paths.


"""







####################
# IMPORT Libraries
####################
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

from filelock import FileLock
from transformers import PreTrainedTokenizer, is_tf_available, is_torch_available
import pdb


logger = logging.getLogger(__name__)


@dataclass
class InputExample:
    """
    A single training/test example for token classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        labels: (Optional) list. The labels for each word of the sequence. This should be
        specified for train and dev examples, but not for test examples.
    """

    guid: str
    words: List[str]
    labels: Optional[List[str]]


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None


class Split(Enum):
    train = "train"
    dev = "dev"
    test = "test"


"""
This class acts as a blueprint (abstract class) for token classification tasks, defining methods for:
	1.	Reading data from files.
	2.	Mapping labels to tokens.
	3.	Converting examples into features for training.
	4.	Filtering out placeholder data.

It does not provide specific implementations but sets the expected structure for subclasses.

"""
class TokenClassificationTask:

    # Reads dataset files from data_dir and returns a list of InputExample objects (word-label pairs).
    @staticmethod
    def read_examples_from_file(data_dir, mode: Union[Split, str]) -> List[InputExample]:
        raise NotImplementedError

    # Reads the possible entity labels from a file.
    @staticmethod
    def get_labels(path: str) -> List[str]:
        raise NotImplementedError

    # Filters out invalid/placeholder examples from the training dataset.
    @staticmethod
    def check_placeholder_pattern(example):
        placeholder_patterns = [
            "This is a placeholder ####",
            "No chain-of-thought provided"
        ]
        for patt in placeholder_patterns:
            if patt in example:
                return True
        return False

    # Converts raw words + labels into model-friendly tokenized inputs.
    @staticmethod
    def convert_examples_to_features(
        examples: List[InputExample],
        label_list: List[str],
        max_seq_length: int,
        tokenizer: PreTrainedTokenizer,
        cls_token_at_end=False,
        cls_token="[CLS]",
        cls_token_segment_id=1,
        sep_token="[SEP]",
        sep_token_extra=False,
        pad_on_left=False,
        pad_token=0,
        pad_token_segment_id=0,
        pad_token_label_id=-100,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True,
        mode: Split = Split.dev
    ) -> List[InputFeatures]:
        """Loads a data file into a list of `InputFeatures`
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
        """
        # TODO clean up all this to leverage built-in features of tokenizers

        label_map = {label: i for i, label in enumerate(label_list)}
        label_map["O"] = pad_token_label_id

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 10_000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))

            # skip the placeholder examples
            example_str = " ".join(example.words)
            if ex_index == 0:
                print(example_str)
            if mode == Split.train and TokenClassificationTask.check_placeholder_pattern(example_str):
                continue

            tokens = []
            label_ids = []
            for word, label in zip(example.words, example.labels):
                word_tokens = tokenizer.tokenize(word)

                # bert-base-multilingual-cased sometimes output "nothing ([]) when calling tokenize with just a space.
                if len(word_tokens) > 0:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = tokenizer.num_special_tokens_to_add()
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[: (max_seq_length - special_tokens_count)]
                label_ids = label_ids[: (max_seq_length - special_tokens_count)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens += [sep_token]
            label_ids += [pad_token_label_id]
            if sep_token_extra:
                # roberta uses an extra separator b/w pairs of sentences
                tokens += [sep_token]
                label_ids += [pad_token_label_id]
            segment_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                label_ids += [pad_token_label_id]
                segment_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                label_ids = [pad_token_label_id] + label_ids
                segment_ids = [cls_token_segment_id] + segment_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding_length = max_seq_length - len(input_ids)
            if pad_on_left:
                input_ids = ([pad_token] * padding_length) + input_ids
                input_mask = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
                segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
                label_ids = ([pad_token_label_id] * padding_length) + label_ids
            else:
                input_ids += [pad_token] * padding_length
                input_mask += [0 if mask_padding_with_zero else 1] * padding_length
                segment_ids += [pad_token_segment_id] * padding_length
                label_ids += [pad_token_label_id] * padding_length

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
                logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
                logger.info("label_ids: %s", " ".join([str(x) for x in label_ids]))

            if "token_type_ids" not in tokenizer.model_input_names:
                segment_ids = None

            features.append(
                InputFeatures(
                    input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids, label_ids=label_ids
                )
            )
        return features


if is_torch_available():
    import torch
    from torch import nn
    from torch.utils.data import Dataset

    class TokenClassificationDataset(Dataset):
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            token_classification_task: TokenClassificationTask,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            # Load data features from cache or dataset file
            cached_features_file = os.path.join(
                data_dir,
                "cached_{}_{}_{}".format(mode.value, tokenizer.__class__.__name__, str(max_seq_length)),
            )

            # Make sure only the first process in distributed training processes the dataset,
            # and the others will use the cache.
            lock_path = cached_features_file + ".lock"
            with FileLock(lock_path):

                if os.path.exists(cached_features_file) and not overwrite_cache:
                    logger.error(f"Loading features from cached file {cached_features_file}")
                    self.features = torch.load(cached_features_file)
                else:
                    logger.error(f"Creating features from dataset file at {data_dir}")
                    examples = token_classification_task.read_examples_from_file(data_dir, mode)
                    # TODO clean up all this to leverage built-in features of tokenizers
                    self.features = token_classification_task.convert_examples_to_features(
                        examples,
                        labels,
                        max_seq_length,
                        tokenizer,
                        cls_token_at_end=bool(model_type in ["xlnet"]),
                        # xlnet has a cls token at the end
                        cls_token=tokenizer.cls_token,
                        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                        sep_token=tokenizer.sep_token,
                        sep_token_extra=False,
                        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                        pad_on_left=bool(tokenizer.padding_side == "left"),
                        pad_token=tokenizer.pad_token_id,
                        pad_token_segment_id=tokenizer.pad_token_type_id,
                        pad_token_label_id=self.pad_token_label_id,
                        mode = mode
                    )
                    logger.error(f"Saving features into cached file {cached_features_file}")
                    torch.save(self.features, cached_features_file)

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]


if is_tf_available():
    import tensorflow as tf

    class TFTokenClassificationDataset:
        """
        This will be superseded by a framework-agnostic approach
        soon.
        """

        features: List[InputFeatures]
        pad_token_label_id: int = -100
        # Use cross entropy ignore_index as padding label id so that only
        # real label ids contribute to the loss later.

        def __init__(
            self,
            token_classification_task: TokenClassificationTask,
            data_dir: str,
            tokenizer: PreTrainedTokenizer,
            labels: List[str],
            model_type: str,
            max_seq_length: Optional[int] = None,
            overwrite_cache=False,
            mode: Split = Split.train,
        ):
            examples = token_classification_task.read_examples_from_file(data_dir, mode)
            # TODO clean up all this to leverage built-in features of tokenizers
            self.features = token_classification_task.convert_examples_to_features(
                examples,
                labels,
                max_seq_length,
                tokenizer,
                cls_token_at_end=bool(model_type in ["xlnet"]),
                # xlnet has a cls token at the end
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
                sep_token=tokenizer.sep_token,
                sep_token_extra=False,
                # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                pad_on_left=bool(tokenizer.padding_side == "left"),
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,
                pad_token_label_id=self.pad_token_label_id,
                mode = mode
            )

            def gen():
                for ex in self.features:
                    if ex.token_type_ids is None:
                        yield (
                            {"input_ids": ex.input_ids, "attention_mask": ex.attention_mask},
                            ex.label_ids,
                        )
                    else:
                        yield (
                            {
                                "input_ids": ex.input_ids,
                                "attention_mask": ex.attention_mask,
                                "token_type_ids": ex.token_type_ids,
                            },
                            ex.label_ids,
                        )

            if "token_type_ids" not in tokenizer.model_input_names:
                self.dataset = tf.data.Dataset.from_generator(
                    gen,
                    ({"input_ids": tf.int32, "attention_mask": tf.int32}, tf.int64),
                    (
                        {"input_ids": tf.TensorShape([None]), "attention_mask": tf.TensorShape([None])},
                        tf.TensorShape([None]),
                    ),
                )
            else:
                self.dataset = tf.data.Dataset.from_generator(
                    gen,
                    ({"input_ids": tf.int32, "attention_mask": tf.int32, "token_type_ids": tf.int32}, tf.int64),
                    (
                        {
                            "input_ids": tf.TensorShape([None]),
                            "attention_mask": tf.TensorShape([None]),
                            "token_type_ids": tf.TensorShape([None]),
                        },
                        tf.TensorShape([None]),
                    ),
                )

        def get_dataset(self):
            self.dataset = self.dataset.apply(tf.data.experimental.assert_cardinality(len(self.features)))

            return self.dataset

        def __len__(self):
            return len(self.features)

        def __getitem__(self, i) -> InputFeatures:
            return self.features[i]

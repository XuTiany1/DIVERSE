import os
import sys
import torch
from transformers import AutoTokenizer, AutoConfig
import argparse


# Ensure the src directory is in the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "../../code/src")
if src_dir not in sys.path:
    sys.path.append(src_dir)

from deberta_model import DebertaV2ForTokenClassification  # Ensure the path is added

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Verifier:
    def __init__(self, model_args):
        """
        Initializes the verifier by loading the fine-tuned DeBERTa-v3 model from the given checkpoint.
        
        Args:
            model_args: An argparse.Namespace or similar object that must contain:
                - checkpoint_path: Path to the saved checkpoint.
                - tokenizer_name (optional): Path or name for the tokenizer. If not provided, checkpoint_path is used.
        """
        # Load the configuration from the checkpoint.
        self.config = AutoConfig.from_pretrained(model_args.checkpoint_path)
        
        # Load the tokenizer. Use the tokenizer_name if provided; otherwise, use the checkpoint path.
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_args.tokenizer_name if hasattr(model_args, "tokenizer_name") and model_args.tokenizer_name
            else model_args.checkpoint_path,
            use_fast=False,
        )
        
        # Load the model from the checkpoint.
        self.model = DebertaV2ForTokenClassification.from_pretrained(
            model_args.checkpoint_path, config=self.config
        )
        self.model.to(device)
        self.model.eval()  # Set the model to evaluation mode

    def get_verifier_probability(self, text):
        """
        Given a GPT-generated reasoning path text, compute the probability that the reasoning path is correct.
        
        Args:
            text (str): The chain-of-thought reasoning text (including markers like "####").
        
        Returns:
            float: The probability corresponding to the "SOLUTION-CORRECT" label.
        
        Example:
            Input: "Step 1: ... Final answer: ####3."
            Output: 0.87 (example probability)
        """
        # Tokenize the input text; adjust max_length to match training settings.
            # Converts the text input into tokenized numerical representations that the model can process.
        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # shape: [batch_size, sequence_length, num_labels]
        
        # Assume the second token (index 1) represents the [CLS] token output for overall classification.
        cls_logits = logits[0][1]  # shape: [num_labels]
        probabilities = torch.softmax(cls_logits, dim=-1)

        # Retrieve the index corresponding to "SOLUTION-CORRECT". 
        solution_correct_index = self.config.label2id.get("SOLUTION-CORRECT", 0)
        prob = probabilities[solution_correct_index].item()
        return prob

# Optional convenience function:
def get_verifier_probability_from_checkpoint(checkpoint_path, text, tokenizer_name=None):
   
    from argparse import Namespace
    args = Namespace(checkpoint_path=checkpoint_path, tokenizer_name=tokenizer_name)
    verifier = Verifier(args)
    return verifier.get_verifier_probability(text)

# Example usage:
if __name__ == "__main__":
    # Replace with the actual checkpoint path and tokenizer if needed.
    checkpoint = "/home/mila/x/xut/github/DIVERSE/DIVERSE/model/deberta_v3/checkpoint-6565"
    sample_text = ( "Janet sells 16 - 3 - 4 = <<16-3-4=9 >> 9 duck eggs a day.\n She makes 9 * 2 = $<<9*2=18 >> 18 every day at the farmer’s market.\n #### 18 \n Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"
    )
    verifier = Verifier(argparse.Namespace(checkpoint_path=checkpoint, tokenizer_name='microsoft/deberta-v3-large'))
    probability = verifier.get_verifier_probability(sample_text)
    print("Verifier probability for SOLUTION-CORRECT:", probability)

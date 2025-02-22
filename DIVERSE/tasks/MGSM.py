import re
import os
import sympy
import pandas as pd
from tasks.task import Task, DATA_PATH
from prompts.MGSM_EN import *
import importlib
from datasets import load_dataset


    # Dynamically import the correct prompt file based on chosen_lang
def load_prompt_module(lang):
    try:
        module_name = f"prompts.MGSM_{lang.upper()}"  # Ensure correct casing
        prompt_module = importlib.import_module(module_name)  # Dynamically import
        return prompt_module
    except ModuleNotFoundError:
        raise ValueError(f"Prompt module for language '{lang}' not found.")

class MgsmTask(Task):

    def __init__(self, args):

        super().__init__()

        ################################
        # Download dataset
        data_card = "juletxara/mgsm"
        data_dir = "data/MGSM/"
        os.makedirs(data_dir, exist_ok=True)
        languages = ['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te']

        # Download and save datasets for each language
        for curr_lang in languages:

            lang_dir = os.path.join(data_dir, curr_lang)
            train_file = os.path.join(lang_dir, "train.tsv")
            test_file = os.path.join(lang_dir, "test.tsv")

            # Skip downloading if files already exist
            if os.path.exists(train_file) and os.path.exists(test_file):
                print(f"Dataset for {curr_lang} already exists. Skipping download.")
                continue  # Skip to next language

            print(f"Downloading dataset for language: {curr_lang}")
            dataset = load_dataset(data_card, curr_lang)
            os.makedirs(lang_dir, exist_ok=True)

            # Save train and test splits as TSV files
            dataset["train"].to_pandas().to_csv(os.path.join(lang_dir, "train.tsv"), sep="\t", index=False)
            dataset["test"].to_pandas().to_csv(os.path.join(lang_dir, "test.tsv"), sep="\t", index=False)

        # Load the chosen langauge dataset
        chosen_lang = args.lang
        lang_dir = os.path.join(data_dir, chosen_lang)
        train_file = os.path.join(lang_dir, "train.tsv")
        test_file = os.path.join(lang_dir, "test.tsv")

        self.train_data = pd.read_csv(train_file, sep="\t", quoting=3)
        self.test_data = pd.read_csv(test_file, sep="\t", quoting=3)

        # Set current data into either train or test
        self.data = self.test_data

        self.prompt_module = load_prompt_module(args.lang)


        ################################
        # Other variable initialization
        self.stops = ['\n'] * 4
        self.steps = 7
        self.value_cache = {}



    ##################
    # HELPER FUNC.
    ##################
    def set_data_split(self, split: str):
        """
        Set the current data split (train or test).
        """
        if split == "train":
            self.data = self.train_data
        elif split == "test":
            self.data = self.test_data
        else:
            raise ValueError("Invalid split. Use 'train' or 'test'.")


    def __len__(self) -> int:
        """
        Returns the number of data instances in the current split.
        """
        return len(self.data)


    def get_input(self, idx: int) -> str:
        """
        Returns the question (input) at the given index in the current split.
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range.")
        
        return self.data.iloc[idx]["question"]



    ##################
    # TO BE TESTED
    ##################
    def ground_truth_answer(self, idx: int):
        """
        output answer
        """
        #print(self.data.iloc[idx])
        #print(self.data.iloc[idx]["answer_number"])
        return self.data.iloc[idx]["answer_number"]
    

    def extract_steps(self, response: str):
        """
        Extracts the full step-by-step reasoning from the response, including the final answer.
        """
        if isinstance(response, list):
            response = response[0]
        
        return response.strip()


    def extract_final_answer(self, response: str):
        """
        Extracts only the final numerical answer from the response.
        """
        if isinstance(response, list):
            response = response[0]

        match = re.search(r"####\s*(-?\d+(\.\d+)?)", response)
        return match.group(1) if match else None
        

    # Normalize the outputs by removing non-numeric characters and extra spaces
    def model_answer(self, answer):
        answer = str(answer).strip().lower()  # Ensure it's a string and normalize case
        answer = re.sub(r"[^\d.]", "", answer)  # Remove non-numeric characters except '.'


        try:
            return int(float(answer))  # Converts to float first, then int to remove decimals
        except ValueError:
            return None  # Return None if no valid number is found


    ##################
    # PROMPT
    ##################

    @staticmethod
    def standard_prompt_wrap(self, x: str) -> str:

        prompt = self.prompt_module.standard_prompt.format(
            question = x
        )
        return prompt
    

    @staticmethod
    def cot_prompt_wrap(self, x: str, prompt: list) -> list:
        list_of_prompt = []

        for prompt_num in prompt:
            # Dynamically get the string attribute instead of calling it
            curr_prompt_template = getattr(self.prompt_module, f"cot_prompt_{prompt_num}")

            # If it's a string, use `.format()` for substitution
            curr_prompt = curr_prompt_template.format(question=x)

            list_of_prompt.append(curr_prompt)

        return list_of_prompt




    @staticmethod
    def compute_final_answer(self, model_output, model_question, verifier):
        # Dictionary to accumulate probability for each unique final answer.
        answer_probabilities = {}
        
        for curr_answer in model_output:
            # Concatenate answer with question if needed for verification.


            answer = "\n".join(curr_answer)

            model_answer_question = answer + "\n" + model_question
            
            # Get the probability that the reasoning is correct.
            probability = verifier.get_verifier_probability(model_answer_question)
            
            # Extract the final answer using a regex that looks for '####'
            # e.g., if answer string ends with "####3", it extracts "3"
            match = re.search(r'####\s*(.*)', answer)
            if match:
                final_ans = match.group(1).strip()
            else:
                # Skip if no final answer marker is found.
                continue
            
            # Sum the probability for each unique final answer.
            answer_probabilities[final_ans] = answer_probabilities.get(final_ans, 0) + probability

        # Choose the final answer with the highest cumulative probability.
        if answer_probabilities:
            selected_answer = max(answer_probabilities.items(), key=lambda x: x[1])[0]
            return selected_answer, answer_probabilities
        else:
            return None, {}


































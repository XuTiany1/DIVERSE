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
        # module_name = f"prompts.MGSM_{lang.upper()}"  # Ensure correct casing
        module_name = f"prompts.MGSM_EN"
        prompt_module = importlib.import_module(module_name)  # Dynamically import
        return prompt_module
    except ModuleNotFoundError:
        raise ValueError(f"Prompt module for language '{lang}' not found.")

class AfriMgsmTask(Task):

    def __init__(self, args):

        super().__init__()

        ################################
        # Download dataset
        data_card = "masakhane/afrimgsm"
        data_dir = "data/AFRI_MGSM/"
        os.makedirs(data_dir, exist_ok=True)
        
        languages = ['amh', 'ewe', 'hau', 'ibo', 'kin', 'lin', 'lug', 'orm', 'sna', 'sot', 'swa', 'twi', 'vai', 'wol', 'xho', 'yor', 'zul']


        # languages = ['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te']

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
        print(lang_dir)
        train_file = os.path.join(lang_dir, "train.tsv")
        test_file = os.path.join(lang_dir, "test.tsv")

        self.train_data = pd.read_csv(train_file, sep="\t", quoting=3)
        self.test_data = pd.read_csv(test_file, sep="\t", quoting=3)

        # Set current data into either train or test
        self.data = self.test_data

        en_lang_dir = os.path.join('data/MGSM/', 'en')
        en_test_file = os.path.join(en_lang_dir, "test.tsv")
        self.english_data = pd.read_csv(en_test_file, sep="\t", quoting=3)

        self.prompt_module = load_prompt_module(args.lang)


        ################################
        # Other variable initialization
        self.stops = ['\n'] * 4

        self. LANGUAGE_PATTERNS = {
            "en": {
                "primary": r"####\s*([-+]?\d+(?:\.\d+)?)",
                "fallback": r"The answer is ([-+]?\d+(?:\.\d+)?)",
            },
            "de": {
                "primary": r"####\s*([-+]?\d+(?:[\.,]\d+)?)",
                "fallback": r"Die Antwort lautet ([-+]?\d+(?:[\.,]\d+)?)",
            }
            # ... add other languages as needed.
        }



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
    

    def get_english_input(self, idx: int) -> str:
        """
        Returns the question (input) at the given index in the current split.
        """
        if idx < 0 or idx >= len(self.data):
            raise IndexError("Index out of range.")
        
        return self.english_data.iloc[idx]["question"]



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



    ############
    # Extract the final answer
    ############
    @staticmethod
    def extract_final_answer(self, response: str, lang: str = "en"):
        """
        Extracts the final numerical answer from the response. It first looks for a marker (e.g. "####").
        If not found, it uses a language-specific fallback regex. Finally, if that fails, it falls back to 
        extracting the last number found.
        """
        # Define a regex pattern that supports commas in numbers.
        number_pattern = r"[-+]?\d{1,3}(?:,\d{3})*(?:\.\d+)?"

        # 1. Try the primary regex pattern (expected marker).
        pattern_primary = self.LANGUAGE_PATTERNS.get(lang, {}).get("primary", r"####\s*(" + number_pattern + r")")
        match = re.search(pattern_primary, response)
        if match:
            return (match.group(1) if match.lastindex else match.group(0)).strip()
        
        # 2. Try a fallback language-specific pattern.
        pattern_fallback = self.LANGUAGE_PATTERNS.get(lang, {}).get("fallback", number_pattern)
        if pattern_fallback:
            match = re.search(pattern_fallback, response)
            if match:
                return match.group(1).strip()

        # 3. As a final fallback, simply extract the last number in the text.
        numbers = re.findall(number_pattern, response)
        if numbers:
            return numbers[-1].strip()

        return None


    @staticmethod
    def compute_final_answer(self, model_output, model_question, verifier, selection_method, error_log_path=None):
        """
        For each generated answer (which may be a multi-line chain-of-thought),
        compute its verifier probability and then choose the final answer that has the highest
        cumulative probability. If no answer can be extracted via the primary marker, a fallback
        mechanism is used.
        """
        answer_probabilities = {}
        raw_probability = {}
        highest_probability = 0
        highest_probability_answer = 0
        most_popular_count = 0
        most_popular_answer = 0

        # Loop over each group of answers in model_output
        for answer_group in model_output:
            # Loop over every answer variant in the group
            for answer_text in answer_group:
                # Combine with the question for the verifier.
                model_answer_question = answer_text + "\n" + model_question
                
                # Get the probability that the reasoning is correct.
                probability = verifier.get_verifier_probability(model_answer_question)
       
                # Extract the final answer using our extraction function.
                extracted = self.extract_final_answer(self, answer_text, lang='en')
                if extracted is None:
                    error_message = (
                        "-------------------\n ERROR: Failed to extract final answer from response:\n"
                        f"{answer_text}\n -------------------\n"
                    )
                    if error_log_path is not None:
                        with open(error_log_path, "a") as ef:
                            ef.write(error_message)
                    continue  # Skip this answer

                # Remove commas from the extracted answer before conversion.
                extracted = extracted.replace(',', '')
                try:
                    rounded_answer = str(int(round(float(extracted))))
                except ValueError:
                    error_message = (
                        "-------------------\n ERROR: Failed to round the answer from extracted answer:\n"
                        f"{extracted}\n Response Text is response:\n{answer_text}\n -------------------\n"
                    )
                    if error_log_path is not None:
                        with open(error_log_path, "a") as ef:
                            ef.write(error_message)
                    continue  # Skip this answer
                    
                final_ans = rounded_answer

                if highest_probability < probability:
                    highest_probability = probability
                    highest_probability_answer = final_ans
                    
                # Record the probability values for this final answer.
                if final_ans not in raw_probability:
                    raw_probability[final_ans] = []
                raw_probability[final_ans].append(probability)
                answer_probabilities[final_ans] = answer_probabilities.get(final_ans, 0) + probability

        if answer_probabilities:

            if selection_method == 'verifier':
                selected_answer = highest_probability_answer
            elif selection_method == 'voting_verifier':
                selected_answer = max(answer_probabilities.items(), key=lambda x: x[1])[0]
            elif selection_method == 'voting':
                selected_answer = max(raw_probability, key=lambda k: len(raw_probability[k]))




            # selected_answer = max(answer_probabilities.items(), key=lambda x: x[1])[0]
            # selected_answer = highest_probability_answer
            return selected_answer, answer_probabilities, raw_probability
        else:
            return None, None, {}


    @staticmethod
    def compute_probability(self, model_output, model_question, verifier):

        reasoning_path_probability = {}

        # Loop over each group of answers in model_output
        for answer_group in model_output:
            # Loop over every answer variant in the group
            for answer_text in answer_group:
                # Combine with the question for the verifier.
                model_answer_question = answer_text + "\n" + model_question
                
                # Get the probability that the reasoning is correct.
                reasoning_probability = verifier.get_verifier_probability(model_answer_question)

                reasoning_path_probability[answer_text] = reasoning_probability
       
        return reasoning_path_probability
















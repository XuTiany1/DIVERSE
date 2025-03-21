import re
import os
import sympy
import pandas as pd
from tasks.task import Task, DATA_PATH
from prompts.PROMPTS import *
import importlib
from datasets import load_dataset

class MATH(Task):

    def __init__(self, args):

        super().__init__()

        ################################
        # Download dataset 
        # First, downlaoad AFRI_MGSM
        afri_data_card = "masakhane/afrimgsm"
        afri_data_dir = "data/AFRI_MGSM/"
        os.makedirs(afri_data_dir, exist_ok=True)
        afri_mgsm_lang = ['amh', 'ewe', 'hau', 'ibo', 'kin', 'lin', 'lug', 'orm', 'sna', 'sot', 'swa', 'twi', 'vai', 'wol', 'xho', 'yor', 'zul']
        self.download_dataset(afri_data_card, afri_data_dir, afri_mgsm_lang)


        mgsm_data_card = "juletxara/mgsm"
        mgsm_data_dir = "data/MGSM/"
        os.makedirs(mgsm_data_dir, exist_ok=True)
        mgsm_lang = ['en', 'es', 'fr', 'de', 'ru', 'zh', 'ja', 'th', 'sw', 'bn', 'te']
        self.download_dataset(mgsm_data_card, mgsm_data_dir, mgsm_lang)

        # Load the chosen langauge dataset
        chosen_lang = args.lang
        
        if chosen_lang in afri_mgsm_lang:
            lang_dir = os.path.join(afri_data_dir, chosen_lang)
        else:
            lang_dir = os.path.join(mgsm_data_dir, chosen_lang)
        print(lang_dir)
        train_file = os.path.join(lang_dir, "train.tsv")
        test_file = os.path.join(lang_dir, "test.tsv")

        self.train_data = pd.read_csv(train_file, sep="\t", quoting=3)
        self.test_data = pd.read_csv(test_file, sep="\t", quoting=3)

        # Set current data into either train or test
        self.data = self.test_data

        ################################
        # English dataset

        en_lang_dir = os.path.join(mgsm_data_dir, 'en')
        en_test_file = os.path.join(en_lang_dir, "test.tsv")
        self.english_data = pd.read_csv(en_test_file, sep="\t", quoting=3)
        
        PROMPTS = {
            0: cot_prompt_0,
            1: cot_prompt_1,
            2: cot_prompt_2,
            3: cot_prompt_3,
            4: cot_prompt_4,
            5: cot_prompt_5,
        }



    #################
    # Dataset Download
    #################
    def download_dataset(self, data_card, data_dir, list_of_languages):
        # Download and save datasets for each language
        for lang in list_of_languages:
            lang_dir = os.path.join(data_dir, lang)
            train_file = os.path.join(lang_dir, "train.tsv")
            test_file = os.path.join(lang_dir, "test.tsv")

            # Skip downloading if files already exist
            if os.path.exists(train_file) and os.path.exists(test_file):
                print(f"Dataset for {lang} already exists. Skipping download.")
                continue  # Skip to next language

            print(f"Downloading dataset for language: {lang}")
            dataset = load_dataset(data_card, lang)
            os.makedirs(lang_dir, exist_ok=True)

            # Save train and test splits as TSV files
            dataset["train"].to_pandas().to_csv(os.path.join(lang_dir, "train.tsv"), sep="\t", index=False)
            dataset["test"].to_pandas().to_csv(os.path.join(lang_dir, "test.tsv"), sep="\t", index=False)



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
        return self.data.iloc[idx]["answer_number"]


    def extract_final_answer(self, response: str):
        """
        Extracts only the final numerical answer from the response.
        """

        return 
        

    ##################
    # PROMPT CREATION
    ##################

    #@staticmethod
    #def standard_prompt_wrap(self, x: str) -> str:
#
    #    prompt = standard_prompt.format(
    #        question = x
    #    )
    #    return prompt
    #
#
    #@staticmethod
    #def cot_prompt_wrap(self, x: str, prompt: list) -> list:
    #    list_of_prompt = []
#
    #    for prompt_num in prompt:
    #        # Dynamically get the string attribute instead of calling it
    #        curr_prompt_template = f"cot_prompt_{prompt_num}"
#
    #        # If it's a string, use `.format()` for substitution
    #        curr_prompt = curr_prompt_template.format(question=x)
#
    #        list_of_prompt.append(curr_prompt)
#
    #    return list_of_prompt

    @staticmethod
    def prompt_wrap(self,
                    question: str,
                    list_of_prompt: list) -> list:
        final_list_of_prompt = []
        
        for prompt_num in list_of_prompt:
            var_name = f"cot_prompt_{prompt_num}"  # Correct variable name generation
            
            if var_name in globals():  # Check if variable exists
                prompt_template = globals()[var_name]  # Retrieve actual prompt
                curr_prompt = prompt_template.format(question=question)
                final_list_of_prompt.append(curr_prompt)
            else:
                raise ValueError(f"Prompt {var_name} not found in global scope.")

        return final_list_of_prompt

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

        return None
    


    @staticmethod
    def extract_final_number(self, response: str):
        """
        Extracts the final numerical value from the response string.
        This function looks for numbers that may include commas, a dollar sign,
        a sign (+/-), and an optional decimal part.
        
        If a number is found, it returns it as an int or float.
        If no number is found, it returns 0.
        """
        # This regex pattern matches:
        # - An optional sign (- or +)
        # - An optional dollar sign ($)
        # - A digit (with possible commas in the middle)
        # - An optional decimal part
        pattern = re.compile(r'[-+]?\$?\d[\d,]*(?:\.\d+)?')
        matches = pattern.findall(response)
        
        if not matches:
            return 0

        # Grab the last match
        final_number_str = matches[-1]
        
        # Remove common extraneous symbols like '$' and commas
        final_number_str = final_number_str.replace('$', '').replace(',', '')
        
        try:
            # Convert to float first, then to int (truncates any decimals)
            return int(float(final_number_str))
        except ValueError:
            return 0




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
















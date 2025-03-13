standard_prompt = '''
নিম্নলিখিত গণিতের প্রশ্নের উত্তর দিন। শুধুমাত্র চূড়ান্ত উত্তরটি একটি সংখ্যা হিসাবে ইনপুট করুন এবং অন্য কিছু নয়।
প্রশ্ন:
{question}
উত্তর:
'''

cot_prompt_0='''
The last number in your response should be the final answer without units.
Question: {question}\n Step-by-Step Answer using English:
'''

cot_prompt_1 = '''
Provide a detailed, step-by-step explanation in English to arrive at the correct solution.  
Ensure that the final number in your response is the final answer without units.  
Question: {question}
'''

cot_prompt_2 = '''
Explain your reasoning step by step in clear English to solve the problem.  
Your response should end with the final numerical answer, without including units.  
Question: {question}
'''

cot_prompt_3 = '''
Break down the solution into logical steps, explaining each one in English.  
The last number you write should be the final computed answer, excluding any units.  
Question: {question}
'''

cot_prompt_4 = '''
Walk through the solution in a structured manner, describing each step in English.  
The final answer should be the last number in your response, without any units.  
Question: {question}
'''
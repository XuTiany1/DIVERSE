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
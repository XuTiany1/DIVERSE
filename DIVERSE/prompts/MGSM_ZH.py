standard_prompt = '''
请回答以下数学问题。只需输入最终答案，且答案必须是一个数字，不要输入其他内容。
问题：
{question}
答案：
'''


cot_prompt_0='''
The last number in your response should be the final answer without units.
Question: {question}\n Step-by-Step Answer using English:
'''
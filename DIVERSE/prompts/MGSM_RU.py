standard_prompt = '''
Ответьте на следующий математический вопрос. Введите только окончательный ответ в виде числа и ничего больше.
Вопрос:
{question}
Ответ:
'''

cot_prompt_0='''
The last number in your response should be the final answer without units.
Question: {question}\n Step-by-Step Answer using English:
'''
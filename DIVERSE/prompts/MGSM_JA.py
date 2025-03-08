standard_prompt = '''
次の数学の問題に答えてください。最終的な答えを数字のみで入力し、それ以外は入力しないでください。
質問:
{question}
答え:
'''

cot_prompt_0='''
The last number in your response should be the final answer without units.
Question: {question}\n Step-by-Step Answer using English:
'''
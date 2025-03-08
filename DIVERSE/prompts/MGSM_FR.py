standard_prompt = '''
Répondez à la question mathématique suivante. Saisissez uniquement la réponse finale sous forme de nombre et rien d'autre.
Question :
{question}
Réponse :
'''

cot_prompt_0='''
The last number in your response should be the final answer without units.
Question: {question}\n Step-by-Step Answer using English:
'''
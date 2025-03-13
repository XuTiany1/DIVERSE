standard_prompt = '''
Responde la siguiente pregunta matemática. Ingresa solo la respuesta final como un número y nada más.
Pregunta:
{question}
Respuesta:
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
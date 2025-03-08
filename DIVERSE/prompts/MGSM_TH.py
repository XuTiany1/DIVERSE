standard_prompt = '''
ตอบคำถามคณิตศาสตร์ต่อไปนี้ ให้ป้อนเฉพาะคำตอบสุดท้ายเป็นตัวเลขเท่านั้น และห้ามป้อนอย่างอื่น
คำถาม:
{question}
คำตอบ:
'''


cot_prompt_0='''
The last number in your response should be the final answer without units.
Question: {question}\n Step-by-Step Answer using English:
'''
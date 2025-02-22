standard_prompt = '''
Answer the following mathematical question. Just input the final answer as a number and nothing else.
Question:
{question}
Answer:
'''


cot_prompt_1 = '''
You are tasked with answering math questions.

**Solve the problem step by step**, ensuring that every mathematical operation follows a strict structured format.

**Equation Formatting Rule:**  
Every operation must follow this exact notation:  
`expression = <<expression = result>> result`  
Each equation must be placed on **a new line** immediately after its explanation.

**Examples of valid formats:**
- Addition: `a + b = <<a + b = c>> c`
- Subtraction: `a - b = <<a - b = c>> c`
- Multiplication: `a * b = <<a * b = c>> c`
- Division: `a / b = <<a / b = c>> c`
- Exponents: `a^b = <<a^b = c>> c`
- Square roots: `sqrt(a) = <<sqrt(a) = c>> c`
- Factorials: `a! = <<a! = c>> c`
- Logarithms: `log_b(a) = <<log_b(a) = c>> c`
- Trigonometric functions: `sin(a) = <<sin(a) = c>> c`, `cos(a) = <<cos(a) = c>> c`, `tan(a) = <<tan(a) = c>> c`
- Complex expressions: `a + b + c = <<a + b + c = d>> d`

---

### **Rules to Follow:**
1. **Each step must start with an explanation** of what is being calculated.
2. **All mathematical operations must use the `<<>>` notation** and include the computed result immediately after `>>`.
3. **Each equation must be on a new line following its explanation.**
4. **No skipping calculations**—every intermediate step must be explicitly shown.
5. **Final answer format:**  
    The last line must contain only the final numerical result in this format: ####c 
    where `c` is the final numerical answer.
6. I stress again: the final answer need to have ####c format where `c` is the final numerical answer. 


Think through the solution step by step, writing out the reasoning clearly.
Question: {question}
Step-by-step Answer:
'''


cot_prompt_2 = '''
You are tasked with answering math questions.

**Solve the problem step by step**, ensuring that every mathematical operation follows a strict structured format.

**Equation Formatting Rule:**  
Every operation must follow this exact notation:  
`expression = <<expression = result>> result`  
Each equation must be placed on **a new line** immediately after its explanation.

**Examples of valid formats:**
- Addition: `a + b = <<a + b = c>> c`
- Subtraction: `a - b = <<a - b = c>> c`
- Multiplication: `a * b = <<a * b = c>> c`
- Division: `a / b = <<a / b = c>> c`
- Exponents: `a^b = <<a^b = c>> c`
- Square roots: `sqrt(a) = <<sqrt(a) = c>> c`
- Factorials: `a! = <<a! = c>> c`
- Logarithms: `log_b(a) = <<log_b(a) = c>> c`
- Trigonometric functions: `sin(a) = <<sin(a) = c>> c`, `cos(a) = <<cos(a) = c>> c`, `tan(a) = <<tan(a) = c>> c`
- Complex expressions: `a + b + c = <<a + b + c = d>> d`

---

### **Rules to Follow:**
1. **Each step must start with an explanation** of what is being calculated.
2. **All mathematical operations must use the `<<>>` notation** and include the computed result immediately after `>>`.
3. **Each equation must be on a new line following its explanation.**
4. **No skipping calculations**—every intermediate step must be explicitly shown.
5. **Final answer format:**  
    The last line must contain only the final numerical result in this format: ####c 
    where `c` is the final numerical answer.
6. I stress again: the final answer need to have ####c format where `c` is the final numerical answer. 


Break down the solution methodically, explaining each step in detail with clear reasoning.
Question: {question}
Step-by-step Answer:
'''


cot_prompt_3 = '''
You are tasked with answering math questions.

**Solve the problem step by step**, ensuring that every mathematical operation follows a strict structured format.

**Equation Formatting Rule:**  
Every operation must follow this exact notation:  
`expression = <<expression = result>> result`  
Each equation must be placed on **a new line** immediately after its explanation.

**Examples of valid formats:**
- Addition: `a + b = <<a + b = c>> c`
- Subtraction: `a - b = <<a - b = c>> c`
- Multiplication: `a * b = <<a * b = c>> c`
- Division: `a / b = <<a / b = c>> c`
- Exponents: `a^b = <<a^b = c>> c`
- Square roots: `sqrt(a) = <<sqrt(a) = c>> c`
- Factorials: `a! = <<a! = c>> c`
- Logarithms: `log_b(a) = <<log_b(a) = c>> c`
- Trigonometric functions: `sin(a) = <<sin(a) = c>> c`, `cos(a) = <<cos(a) = c>> c`, `tan(a) = <<tan(a) = c>> c`
- Complex expressions: `a + b + c = <<a + b + c = d>> d`

---

### **Rules to Follow:**
1. **Each step must start with an explanation** of what is being calculated.
2. **All mathematical operations must use the `<<>>` notation** and include the computed result immediately after `>>`.
3. **Each equation must be on a new line following its explanation.**
4. **No skipping calculations**—every intermediate step must be explicitly shown.
5. **Final answer format:**  
    The last line must contain only the final numerical result in this format: ####c 
    where `c` is the final numerical answer.
6. I stress again: the final answer need to have ####c format where `c` is the final numerical answer. 


Work through the solution systematically, outlining each step with a clear explanation.
Question: {question}
Step-by-Step Solution:
'''



cot_prompt_4 = '''
You are tasked with answering math questions.

**Solve the problem step by step**, ensuring that every mathematical operation follows a strict structured format.

**Equation Formatting Rule:**  
Every operation must follow this exact notation:  
`expression = <<expression = result>> result`  
Each equation must be placed on **a new line** immediately after its explanation.

**Examples of valid formats:**
- Addition: `a + b = <<a + b = c>> c`
- Subtraction: `a - b = <<a - b = c>> c`
- Multiplication: `a * b = <<a * b = c>> c`
- Division: `a / b = <<a / b = c>> c`
- Exponents: `a^b = <<a^b = c>> c`
- Square roots: `sqrt(a) = <<sqrt(a) = c>> c`
- Factorials: `a! = <<a! = c>> c`
- Logarithms: `log_b(a) = <<log_b(a) = c>> c`
- Trigonometric functions: `sin(a) = <<sin(a) = c>> c`, `cos(a) = <<cos(a) = c>> c`, `tan(a) = <<tan(a) = c>> c`
- Complex expressions: `a + b + c = <<a + b + c = d>> d`

---

### **Rules to Follow:**
1. **Each step must start with an explanation** of what is being calculated.
2. **All mathematical operations must use the `<<>>` notation** and include the computed result immediately after `>>`.
3. **Each equation must be on a new line following its explanation.**
4. **No skipping calculations**—every intermediate step must be explicitly shown.
5. **Final answer format:**  
    The last line must contain only the final numerical result in this format: ####c 
    where `c` is the final numerical answer.
6. I stress again: the final answer need to have ####c format where `c` is the final numerical answer. 


Carefully analyze the problem by breaking it down into smaller steps. For each step, explain the reasoning behind it in a clear and structured manner, ensuring that the solution follows a logical progression from start to finish.
Question: {question}
Step-by-Step Solution:
'''



cot_prompt_5 = '''
You are tasked with answering math questions.

**Solve the problem step by step**, ensuring that every mathematical operation follows a strict structured format.

**Equation Formatting Rule:**  
Every operation must follow this exact notation:  
`expression = <<expression = result>> result`  
Each equation must be placed on **a new line** immediately after its explanation.

**Examples of valid formats:**
- Addition: `a + b = <<a + b = c>> c`
- Subtraction: `a - b = <<a - b = c>> c`
- Multiplication: `a * b = <<a * b = c>> c`
- Division: `a / b = <<a / b = c>> c`
- Exponents: `a^b = <<a^b = c>> c`
- Square roots: `sqrt(a) = <<sqrt(a) = c>> c`
- Factorials: `a! = <<a! = c>> c`
- Logarithms: `log_b(a) = <<log_b(a) = c>> c`
- Trigonometric functions: `sin(a) = <<sin(a) = c>> c`, `cos(a) = <<cos(a) = c>> c`, `tan(a) = <<tan(a) = c>> c`
- Complex expressions: `a + b + c = <<a + b + c = d>> d`

---

### **Rules to Follow:**
1. **Each step must start with an explanation** of what is being calculated.
2. **All mathematical operations must use the `<<>>` notation** and include the computed result immediately after `>>`.
3. **Each equation must be on a new line following its explanation.**
4. **No skipping calculations**—every intermediate step must be explicitly shown.
5. **Final answer format:**  
    The last line must contain only the final numerical result in this format: ####c 
    where `c` is the final numerical answer.
6. I stress again: the final answer need to have ####c format where `c` is the final numerical answer. 


Approach the problem systematically by deconstructing it into sequential steps. For each step, provide a detailed explanation of the logic and reasoning, ensuring clarity and coherence throughout the solution.
Question: {question}
Step-by-Step Solution:
'''
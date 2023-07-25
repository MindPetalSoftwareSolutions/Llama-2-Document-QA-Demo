'''
===========================================
        Module: Prompts collection
===========================================
'''
# Note: Precise formating of spacing and indentation of the prompt template is important for MPT-7B-Instruct,
# as it is highly sensitive to whitespace changes. For example, it could have problems generating
# a summary from the pieces of context if the spacing is not done correctly

qa_template = """Use the following pieces of information to answer the user's question.

Context: {context}
Question: {question}

Return the answer below and nothing else. The answer should be as brief as possible and omit unnecessary context. 
The answer to the question is:
"""
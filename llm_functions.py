import torch
from transformers import pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain

# model_name = "MBZUAI/LaMini-Flan-T5-783M"
model_name = "MBZUAI/LaMini-Flan-T5-248M"

pipe = pipeline("text2text-generation",
                model=model_name,
                max_length=2048,
                device_map='cuda:0'
                )

llm = HuggingFacePipeline(pipeline=pipe)


def generate_response(input_query, context=None):

    input_documents = ""
    if context:
        prompt_template = """ 
    
        Answer the following question in more than 40 words.  
    
        Context : {context}
    
        Question : {question}
    
        Answer : """

        prompt = PromptTemplate.from_template(prompt_template)

        llm_chain = LLMChain(llm=llm, prompt=prompt)
        llm_response = llm_chain.predict(context=context, question=input_query)
    else:
        llm_response = llm(input_query)
    return llm_response

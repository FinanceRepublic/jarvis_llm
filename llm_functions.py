import torch
from transformers import LlamaTokenizer, LlamaForCausalLM, GenerationConfig, pipeline
from langchain.llms import HuggingFacePipeline

tokenizer = LlamaTokenizer.from_pretrained("eachadea/vicuna-7b-1.1")

model = LlamaForCausalLM.from_pretrained("eachadea/vicuna-7b-1.1",
                                         load_in_8bit=True,
                                         device_map='auto',
                                         torch_dtype=torch.float16,
                                         low_cpu_mem_usage=True)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_length=2048,
    temperature=0,
    top_p=0.95,
    repetition_penalty=1.15
)

llm = HuggingFacePipeline(pipeline=pipe)


def generate_response(input_query):
    llm_response = llm(input_query)
    llm_response = str(llm_response)
    return llm_response

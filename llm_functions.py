import ctranslate2
import transformers

# model_name = "MBZUAI/LaMini-Flan-T5-783M"
# model_name = "MBZUAI/LaMini-Flan-T5-248M"

translator = ctranslate2.Translator("models/reduced_lamini_248M")
tokenizer = transformers.AutoTokenizer.from_pretrained("MBZUAI/LaMini-Flan-T5-248M")


def ctranslated_response(question, context=None):
    if context:
        input_text = f'''Instruction: Answer the following question in detail based on the context
    
                         Context :
                        
                         {context}
                        
                         Question : {question}
                        
                         Answer : '''

        input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))

        results = translator.translate_batch([input_tokens])

        output_tokens = results[0].hypotheses[0]
        output_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))

    else:
        input_text = question
        input_tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(input_text))

        results = translator.translate_batch([input_tokens])

        output_tokens = results[0].hypotheses[0]
        output_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(output_tokens))

    return output_text

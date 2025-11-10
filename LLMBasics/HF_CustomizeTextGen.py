import sys
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
sys.path.append('../')  

HF_API_KEY=os.getenv("HUGGINGFACEHUB_API_TOKEN")
print(HF_API_KEY)


client = InferenceClient(api_key=HF_API_KEY)

completion = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {
            "role": "user",
            "content": "How LLM works"
        }
    ],


    #All below are optional parameters
    #Max Length of the response
    max_tokens=200, #Controls the maximum number of tokens to generate

    #Sampling parameters
    temperature=0.7, #Controls the randomness of the output|  Range: 0.0 to 2.0
    top_p=0.6,       #Controls diversity via nucleus sampling| Max 1.0
#    top_k=50,      #Limits the next token selection to the top_k most probable tokens. Not all models support this parameter.

    #Penalties
    frequency_penalty=0.7, #Penalizes new tokens based on existing frequency in the text so far| -2.0 to 2.0. Positive values penalize
    presence_penalty=0.9,  #Penalizes new tokens based on thier presence in the text so far| -2.0 to 2.0. Positive values penalize



#    logprobs=5,  #Include the log probabilities on the logprobs most likely tokens, as well the chosen tokens.



)
for choice in completion.choices:
    print("Finish Reason: ", choice.finish_reason)
    print("Message Role: ", choice.message.role)
    print("Message Content: ", choice.message.content)
    print("Message Reasoning: ", choice.message.reasoning)
    print("Tool Calls: ", choice.message.tool_calls)

print("Completion Tokens: ", completion.usage.completion_tokens)
print("Prompt Tokens: ", completion.usage.prompt_tokens)
print("Total Tokens: ", completion.usage.total_tokens)
print("Completion Reasoning Tokens: ", completion.usage.completion_tokens_details.get('reasoning_tokens', 0))

#print(completion)

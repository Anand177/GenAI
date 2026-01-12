import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")

HF_API_KEY=os.getenv("HUGGINGFACEHUB_API_TOKEN")

client = InferenceClient(api_key=HF_API_KEY)
model_name = "openai/gpt-oss-20b"

messages=[
    {
         "role": "system",
         "content": "You are a helpful assistant"
    }
]

def chat_with_model():

    while True:
        try:
            user_input = input("\n[User] : ")
            messages.append({"role": "user", "content": user_input})

            if(user_input.lower() in ['exit', 'quit', 'bye']):
                print("\n[System] : Exiting Chat Session")
                break
        

            completion = client.chat.completions.create(
                model=model_name,

                messages=messages,
                max_tokens=200, #Controls the maximum number of tokens to generate
                temperature=0.7
            )

            print("[AI Asst]: ", completion.choices[0].message.content)
            messages.append({"role": "assistant", "content": completion.choices[0].message.content})    
        
        except KeyboardInterrupt:
            print("\n[System] : Session Interrupted. Exiting Chat Session")
            break

#print(completion)

if __name__ == "__main__":
    chat_with_model()
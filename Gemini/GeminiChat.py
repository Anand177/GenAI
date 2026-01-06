from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

import os


load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini Chat Model
llm = ChatGoogleGenerativeAI(
    model="gemini-flash-latest",
    temperature=0.7
)
print("Gemini model initialized successfully.")

memory = ConversationBufferMemory(
    memory_key="chat_history", 
    return_messages=True  # Ensure messages are returned as Message objects
)

# Use a MessagesPlaceholder to include the chat history from the memory component.
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and friendly assistant interacting with 6 year old Indian kid named Anvith.
        Keep your answers engaging, interesting and conversational."""),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("user", "{input}")
])

# A Chain of Runnables is used to process the input through several steps:

# 3.1. Load Memory: This Runnable retrieves the chat history from the memory component.
# The `RunnablePassthrough` here ensures that the original input is also passed down.
chat_chain_with_memory = (RunnablePassthrough.assign(
        chat_history=lambda x: memory.load_memory_variables({})['chat_history']
    )
    | prompt          # 3.2. Pass context and input to the Prompt Template
    | llm             # 3.3. Invoke the Gemini LLM
    | StrOutputParser()
)

def run_chat():
    """Starts the interactive console chat loop."""
    print("\n" + "="*50)
    print("🤖 GeminiBot Interactive Chat")
    print("Type 'quit', 'exit', or 'clear' to end the session.")
    print("="*50 + "\n")

    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            # Handle exit commands
            if user_input.lower() in ["quit", "exit", "clear"]:
                print("\nSession ended. Goodbye!")
                break
            
            if not user_input:
                continue

            # 1. Invoke the Chain
            # The input dictionary only needs the 'input' key. The chat_history is loaded internally.
            response = chat_chain_with_memory.invoke({"input": user_input})

            # 2. Print the Response
            print(f"[GeminiBot]: {response}\n")

            # 3. Save Context (Crucial for memory)
            # The memory needs to be explicitly saved after each turn.
            memory.save_context(
                {"input": user_input},
                {"output": response}
            )

        except KeyboardInterrupt:
            print("\nSession interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.")
            # Optionally, you might break the loop here depending on error severity
            
if __name__ == "__main__":
    run_chat()

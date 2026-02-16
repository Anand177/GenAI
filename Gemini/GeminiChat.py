from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
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

store={}
def get_session_history(session_id: str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# Use a MessagesPlaceholder to include the chat history from the memory component.
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful and friendly assistant interacting with 6 year old Anvith.
        Keep your answers engaging, interesting and conversational."""),
    MessagesPlaceholder(variable_name="chat_history"), 
    ("user", "{input}")
])
chain =prompt | llm | StrOutputParser()

# A Chain of Runnables is used to process the input through several steps:

# 3.1. Load Memory: This Runnable retrieves the chat history from the memory component.
# The `RunnablePassthrough` here ensures that the original input is also passed down.
chat_chain_with_memory = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
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
            response = chat_chain_with_memory.invoke({"input": user_input},
                            config={"configurable": {"session_id": "anvith_session_1"}})

            # 2. Print the Response
            print(f"[GeminiBot]: {response}\n")

        except KeyboardInterrupt:
            print("\nSession interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\nAn error occurred: {e}")
            print("Please try again.")
            # Optionally, you might break the loop here depending on error severity
            
if __name__ == "__main__":
    run_chat()

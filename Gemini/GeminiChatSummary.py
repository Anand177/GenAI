from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

import os
load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Initialize the Gemini Chat Model
llm = ChatGoogleGenerativeAI( model="gemini-flash-latest", temperature=0.7)

# Define Prompt
prompt= ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and friendly assistant. Keep your answers engaging and interesting"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("user", "{input}")
])

# Create Chain
chain = prompt | llm | StrOutputParser()

# Store message history
store = {}
def get_session_history(session_id : str):
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

def summarize_chat_history(session_id: str):
    if session_id not in store:
        return "No chat history found for this session."
    
    history = get_session_history(session_id)
    messages = history.messages
    
    if not messages:
        return "No messages to summarize."
    
    # Create a summary prompt
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at summarizing conversations. 
        Provide a concise summary of the chat conversation.
        Include:
        - Main topics discussed
        - Key questions asked by the user
        - Important information or advice provided
        - Overall tone and outcome of the conversation
        
        Keep the summary brief but informative (3-5 sentences)."""),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "Please summarize our conversation above.")
    ])
    
    summary_chain= summary_prompt | llm | StrOutputParser()
    summary = summary_chain.invoke({"chat_history" : messages})
    
    return summary

# Wrap history with chain
wrapped_chain = RunnableWithMessageHistory(chain, get_session_history, 
                    input_messages_key="input", history_messages_key="chat_history")

def run_chat():

    print("\n" + "-"*50)
    print("🤖 GeminiBot Interactive Chat")
    print("Type 'quit', 'exit', or 'clear' to end the session.")
    print("-"*50 + "\n")

    config = {"configurable": {"session_id": "my_chat_1"}}

    while True:
        try:
            user_input = input("[You]: ").strip()     #   Get User Input 

            if user_input.lower() in ["quit", "exit", "clear"]:     # Close Chat on User Request
                print("\nSession ended. Goodbye!")
                print("-"*50)
                print("Generating Conversation Summary")
                print("-"*50)
                summary=summarize_chat_history("my_chat_1")
                print(summary)
                print("-"*50)
                break
            
            if not user_input:
                continue

            # Invoke wrapped Chain
            response = wrapped_chain.invoke({"input": user_input}, config=config)
            print(f"[GeminiBot]: {response}\n")


        except KeyboardInterrupt:
            print("\nSession interrupted by user. Goodbye!")
            print("-"*50)
            print("Generating Conversation Summary")
            print("-"*50)
            summary=summarize_chat_history("my_chat_1")
            print(summary)
            print("-"*50)
            break
        except Exception as e:
            print(f"\nError occurred: {e}")
            print("Session Interrupted.")
            
if __name__ == "__main__":
    run_chat()

from dotenv import load_dotenv

import requests
import json
import time
import os

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")


# --- Gemini API Configuration ---
# NOTE: The API key is left empty. In environments like Canvas, this is provided automatically.
API_KEY = os.getenv("GOOGLE_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent"
HEADERS = {'Content-Type': 'application/json'}

# Define the System Instruction outside of the main history list
SYSTEM_PROMPT = "You are a helpful assistant. Keep your responses concise and conversational."

# --- History Management Structure ---
# History will be a list of dictionaries, following the Gemini API 'contents' structure:
# [{"role": "user", "parts": [{"text": "Hello!"}]}, ...]

def exponential_backoff_fetch(url, payload, max_retries=5):
    """Handles API calls with exponential backoff for transient errors."""
    for attempt in range(max_retries):
        try:
            # We add the API key to the URL here
            full_url = f"{url}?key={API_KEY}"
            
            response = requests.post(full_url, headers=HEADERS, data=json.dumps(payload))
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            return response.json()
        except requests.exceptions.HTTPError as e:
            # Only retry on 429 (Rate Limit) or 5xx errors
            if response.status_code in [429, 500, 503] and attempt < max_retries - 1:
                wait_time = 2 ** attempt
                # print(f"Rate limit or server error ({response.status_code}). Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                # print(f"Attempt {attempt + 1}/{max_retries} failed. Error: {e}")
                # Re-raise the exception to be caught in get_gemini_response
                raise requests.exceptions.HTTPError(
                    f"Attempt {attempt + 1}/{max_retries} failed. "
                    f"{response.status_code} Client Error: {response.reason} for url: {url}", 
                    response=response
                )


def get_gemini_response(history, system_prompt):
    """
    Calls the Gemini API using the full conversation history and system instruction.

    Args:
        history (list): The list of past messages in the correct API format (user/model roles only).
        system_prompt (str): The instruction defining the model's persona/behavior.

    Returns:
        tuple: (response_text, new_history_entry)
    """
    # CORRECT: System instruction is passed in the separate 'systemInstruction' field
    payload = {
        "contents": history,
        "systemInstruction": {
            "parts": [{"text": system_prompt}]
        },
    }

    try:
        api_result = exponential_backoff_fetch(API_URL, payload)

        # Extract the generated text
        candidate = api_result.get('candidates', [{}])[0]
        text = candidate.get('content', {}).get('parts', [{}])[0].get('text', 'Error: Could not get a valid response.')

        # Construct the new assistant message to add to history
        assistant_message = {
            "role": "model",
            "parts": [{"text": text}]
        }
        
        return text, assistant_message

    except requests.exceptions.HTTPError as e:
        # Re-raise the exception with a clean message to be handled in main()
        raise Exception(f"API Error: {e}")
    except Exception as e:
        # Handle other errors (e.g., json parsing)
        raise Exception(f"An unexpected error occurred: {e}")


def main():
    print("--- Simple Gemini Chat Agent (No LangChain Memory) ---")
    print("Type 'quit' or 'exit' to end the conversation.")
    print("-" * 50)

    # Initialize the conversation history list (only user/model roles)
    conversation_history = []
    
    # We use the globally defined SYSTEM_PROMPT

    while True:
        user_input = input("You: ")

        if user_input.lower() in ['quit', 'exit']:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        # 1. Construct the user message object
        user_message = {
            "role": "user",
            "parts": [{"text": user_input}]
        }

        # 2. Add the user message to the history
        conversation_history.append(user_message)

        try:
            # 3. Get the response from the Gemini API, passing the system prompt separately
            gemini_text, assistant_message = get_gemini_response(conversation_history, SYSTEM_PROMPT)

            print(f"Gemini: {gemini_text}")

            # 4. Add the assistant's response to the history for context in the next turn
            if assistant_message:
                conversation_history.append(assistant_message)

        except Exception as e:
            print(f"\nError: {e}")
            # Remove the last user message to avoid it being retried with the same error
            if conversation_history and conversation_history[-1]['role'] == 'user':
                conversation_history.pop()
            print("Please try your input again.")

if __name__ == "__main__":
    main()
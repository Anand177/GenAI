from langchain_core.language_models import FakeListLLM, FakeStreamingListLLM

# Define fake responses to be used by the FakeLLM
fake_responses = [
    "This is a fake response 1",
    "This is a fake response 2",
    "This is a fake response 3"
]


#Create fake LLM instance
fake_llm = FakeListLLM(responses=fake_responses)

# Test the FakeLLM with a sample query
response = fake_llm.invoke("What is the capital of France?")
print("FakeLLM Response:", response)

#Test with async invoke
import asyncio

async def test_async():
    response = await fake_llm.ainvoke("What is the capital of Germany?")
    print("FakeLLM Async Response:", response)

asyncio.run(test_async())


# Create fake streaming LLM instance
fake_streaming_llm = FakeStreamingListLLM(responses=fake_responses)
streaming_object = fake_streaming_llm.stream("What is the capital of India?")
# Sync streaming
for chunk in streaming_object:
    print(chunk, end='', flush=True)

# Async streaming
async def test_async_streaming():
    streaming_object = fake_streaming_llm.astream("What is capital of USA.")
    print("FakeStreamingLLM Async Streaming Response:")
    async for chunk in streaming_object:
        print(chunk, end='', flush=True)

asyncio.run(test_async_streaming())


# Batch invoke test
requests = [
    "What is the capital of Italy?",
    "What is the capital of Spain?",
    "What is the capital of Portugal?"
]

batch_responses = fake_llm.batch(requests)
for response in batch_responses:
    print("Batch Response:", response)

# Async batch invoke test
async def test_async_batch():
    batch_responses = await fake_llm.abatch(requests)
    for response in batch_responses:
        print("Async Batch Response:", response)

asyncio.run(test_async_batch())
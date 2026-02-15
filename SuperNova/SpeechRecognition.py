from dotenv import load_dotenv
from huggingface_hub import InferenceClient
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAI
from pydantic import BaseModel, Field

from google import genai
from google.genai import types

import os, wave, time

load_dotenv("C:\\Learning\\AI\\Key\\Api-key.txt")
HF_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY=os.getenv("GOOGLE_API_KEY")

client=InferenceClient(api_key=HF_TOKEN)
model_id="openai/whisper-large-v3-turbo"
input_text=""

try:
    audio_file = "C:/Learning/Python/GenAI/SuperNova/Inputs/Anvith.flac"
    output=client.automatic_speech_recognition(audio_file, model=model_id)
    print(output.text)
    input_text=output.text


except Exception as e:
    print(f"Exception {e} occured")

instruction_template="""
You are an expect English Tutor. 
Translate below passage from broken Indian English to Simple English with proper grammar.
Provide translated passage, explain translation rationale and steps implemented to make passage better,
summarize tips to follow to construct similar passages in English.
Also translate and provide the summary of changes made and tips in Tamil language

English is a foreign language for the person providing the passage. 
Be appreciative of his efforts in your response.
Explain in interactive, very simple way and in second person narration for his to understand.
Strictly Avoid authoritarian words

Input Passage:
{input_passage}

Output Format:
{output_format}
"""

class SpeechTranslator(BaseModel):
    input_passage: str = Field(description="Exact input passage provided to the model for translation")
    translated_passage: str = Field(description="This is the passage translated to Simple English from Broken passage")
    changes_made: str = Field(description=" Summary of the changes and corrections made to make passage " \
    "better for communication in second person narration in English")
    changes_made_in_tamil: str = Field(description=" Summary of the changes and corrections made to make passage " \
    "better for communication in second person narration in Tamil")
    tips: str = Field(description="Summary of tips in second person narration to construct similar passages "
    "in proper English")
    tips_in_tamil: str = Field(description="Summary of tips in second person narration to construct similar passages "
    "in proper Tamil")

class SpeechTranslatorForSpeech(BaseModel):
    input_passage: str = Field(description="Exact input passage provided to the model for translation")
    translated_passage: str = Field(description="This is the passage translated to Simple English from Broken passage")
    changes_made: str = Field(description=" Summary of the changes and corrections made to make passage " \
    "better for communication in second person narration in English")
    tips: str = Field(description="Summary of tips in second person narration to construct similar passages "
    "in proper English")

output_parser=PydanticOutputParser(pydantic_object=SpeechTranslator)

prompt=PromptTemplate(template=instruction_template, 
                      input_variables=["input_passage"],
                      partial_variables= {"output_format" : output_parser.get_format_instructions})

llm=GoogleGenerativeAI(model="gemini-flash-latest", google_api_key=GOOGLE_API_KEY, temperature = 0.35)

chain = prompt | llm | output_parser
translated_op=chain.invoke({"input_passage" : input_text})

#Printing Output
print(f"Input Passage : {translated_op.input_passage}")
print(f"Translated Passage : {translated_op.translated_passage}")
print(f"Changes Made : {translated_op.changes_made}")
print(f"Changes Made in Tamil : {translated_op.changes_made_in_tamil}")
print(f"Tips in English : {translated_op.tips}")
print(f"Tips in Tamil : {translated_op.tips_in_tamil}")

instruction = """Read the entire following text exactly as written, including the section titles.
Use a warm, encouraging, and slow pace:"""
speech_inp = f"""Thats an amazing attempt. 
Your Passage can be made better as : {translated_op.translated_passage}
Below are the changes made to improve your passage : {translated_op.changes_made}
You can follow these tips to improve your English : {translated_op.tips}"""

print(speech_inp)
gemini_client=genai.Client(api_key=GOOGLE_API_KEY)

def save_wav(file_name, pcm_data, rate=24000):
    with wave.open(file_name, "wb") as wf:
        wf.setnchannels(1)      # Mono
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(rate)   # 24kHz (Gemini Default)
        wf.writeframes(pcm_data)

print("Generating Audio")
response = gemini_client.models.generate_content(model="gemini-2.5-flash-preview-tts",
                contents=f"{instruction} \n\nText : {speech_inp}",
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name="Puck"
                            )
                        )
                    )
                )
            )

epoch_time = time.time()
audio_bytes = response.candidates[0].content.parts[0].inline_data.data
output_file = f"C:/Learning/Python/GenAI/SuperNova/Outputs/translated_speech_{epoch_time}.wav"
save_wav(output_file, audio_bytes)
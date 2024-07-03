from utils import record_audio, save_as_wav, find_and_insert_text
from openai import OpenAI
import os
from docx import Document

def main():
    # Record and save audio
    duration = 5
    filename = "temp.wav"
    audio_data, sample_rate = record_audio(duration)
    save_as_wav(audio_data, sample_rate, filename)

    # Set API Key and initiate client
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key is None:
        raise ValueError("API key is not set in the environment variables.")
    client = OpenAI(api_key=api_key)

    # Transcribe audio
    with open("temp.wav", "rb") as audio_file:
        transcription_response = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
    transcribed_text = transcription_response.text
    print("Transcription:", transcribed_text)

    # Generate summary and keywords
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "system",
                "content": "Extract a list of keywords and provide a summary of the text."
            },
            {
                "role": "user",
                "content": transcribed_text
            }
        ],
        temperature=0.5,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )

    if response.choices:
        first_choice = response.choices[0]
        if hasattr(first_choice, 'message') and hasattr(first_choice.message, 'content'):
            keywords_and_summary = first_choice.message.content
            print("Extracted Keywords and Summary:", keywords_and_summary)
        else:
            print("No content found in the response message.")
    else:
        print("No valid response or choices found.")

    doc = Document('template.docx')

    if not find_and_insert_text(doc, "speech-to-text", transcribed_text):
        print("Speech to Text title not found, added at the end.")

    if not find_and_insert_text(doc, "keyword extraction", keywords_and_summary):
        print("Keyword Extraction title not found, added at the end.")

    doc.save('updated_template.docx')

if __name__ == "__main__":
    main()

import pdfplumber
from transformers import pipeline

# Load summarization pipeline
summarizer = pipeline("summarization")

# Function to extract text and summarize PDF
def summarize_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()

    # Summarize extracted text
    summary = summarizer(text, max_length=100, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Example use
pdf_path = 'testiranje.pdf'
summary = summarize_pdf(pdf_path)
print(summary)

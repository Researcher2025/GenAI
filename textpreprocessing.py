import fitz
import re
import nltk
import spacy
import time
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import os
nltk.download('punkt')
nltk.download('stopwords')

def read_pdf(file_path):
    pdf_text = ""
    document = fitz.open(file_path)
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        pdf_text += page.get_text()
    return pdf_text

# Initialize spacy model
nlp = spacy.load('en_core_web_sm')

def clean_text(text):
    # Remove emails
    text = re.sub(r'\S+@\S+', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove non-alphabetic characters and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def tokenize_text(text):
    return word_tokenize(text)

def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word.lower() not in stop_words]

def stem_tokens(tokens):
    stemmer = PorterStemmer()
    return [stemmer.stem(word) for word in tokens]

def lemmatize_tokens(tokens):
    doc = nlp(' '.join(tokens))
    return [token.lemma_ for token in doc]

def normalize_text(text, use_stemming=False):
    text = text.lower()  # Convert to lowercase
    text = clean_text(text)  # Clean text
    tokens = tokenize_text(text)  # Tokenize text
    tokens = remove_stopwords(tokens)  # Remove stop words
    if use_stemming:
        tokens = stem_tokens(tokens)  # Stem tokens
    else:
        tokens = lemmatize_tokens(tokens)  # Lemmatize tokens
    return ' '.join(tokens)

# Example usage
resume_text = read_pdf('TarunResume.pdf')
job_description_text = read_pdf('sample-job-description.pdf')

normalized_resume_text = normalize_text(resume_text)
normalized_job_description_text = normalize_text(job_description_text)

with open('resume.txt','w') as file:
    file.write(normalized_resume_text)
with open('jobdescription.txt','w') as file:
    file.write(normalized_job_description_text)
print("Done !")
print("Checkout the file:)")
os.startfile("resume.txt")
os.startfile("jobdescription.txt")
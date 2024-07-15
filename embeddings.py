#docker run -p 6333:6333 -v .:/qdrant/storage qdrant/qdrant
print('Checking NLTK Modules...')
import fitz 
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
import warnings
from qdrant_client import QdrantClient
import re
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import textwrap
import google.generativeai as genai
from qdrant_client.http import models
from config import OPEN_API_KEY
from openai import OpenAI

nltk.download('punkt',quiet=True)
nltk.download('stopwords',quiet=True)

def to_markdown(text):
  text = text.replace('•', '  *')
  return textwrap.indent(text, '', predicate=lambda _: True)
# Function to perform a similarity search in Qdrant
def read_pdf(file_path):
    pdf_text = ""
    document = fitz.open(file_path)
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        pdf_text += page.get_text()
    return pdf_text

# Initialize spacy model
nlp = spacy.load('en_core_web_sm')
def get_result(system,prompt):
    completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  temperature=0.7,
  n=1,
  messages=[
    {"role": "system", "content": system},
    {"role": "user", "content": prompt}
  ]
  
)

    return completion.choices[0].message.content

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
    temp= ' '.join(tokens) 
    formatted_data = "{'skills':'c,java,python,etc..','experience':'commaseperatedin string format','education':'education in string format'}"
    return get_result("You helping me in 'AI Resume Match maker' project",f"Parse the skills, education, experience into a Python dictionary for this data {temp}. Do not disturb the internal data. Convert it into a string format such as {formatted_data}. Ensure the format is ready to use with the eval() method and make it a single line.")

def retrieve_point_by_id(collection_name, point_id):
    try:
        result = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[point_id]
        )
        if result:
            for point in result:
                text=point.payload.get('text')
                return text
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# def text_response(prompt):
#   res=model.generate_content(prompt)
#   return to_markdown(res.text)
# Function to insert data into Qdrant
def insert(collection_name, texts):
        embeddings = embeddings_model.embed_documents(texts)

        # Check if the collection exists and create it if it doesn't
        if qdrant_client.collection_exists(collection_name):
            qdrant_client.delete_collection(collection_name)

        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=len(embeddings[0]), distance=models.Distance.COSINE)
        )

        # Store embeddings in Qdrant using Upsert with integer point IDs
        points = [
            models.PointStruct(id=i+1, vector=embedding, payload={"text": texts[i]})
            for i, embedding in enumerate(embeddings)
        ]
        qdrant_client.upsert(collection_name=collection_name, points=points)
if __name__=="__main__":
    qdrant_client = QdrantClient("http://localhost:6333")  # Replace with your Qdrant URL
     # Example usage
    print("Initializing OpenAI")
    client = OpenAI(api_key=OPEN_API_KEY)
    genai.configure(api_key="AIzaSyBbQuzonSHusVP2HcsvpBxAWHO7XZfrMrM")
    model = genai.GenerativeModel('gemini-pro')
    resume_text = read_pdf('./Resumes/Frontend1.pdf')
    job_description_text = read_pdf('./JD/AI_Jd.pdf')
    normalized_resume_text = normalize_text(resume_text)
    normalized_job_description_text = normalize_text(job_description_text)
    
    try:
        with open('resume.txt','w') as file:
            file.write(normalized_resume_text)
    except Exception as e:
        print(normalize_text)
    with open('jobdescription.txt','w') as file:
        file.write(normalized_job_description_text)
    print('Stored the data..')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model_name = "BAAI/bge-large-en"
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': False}
        embeddings_model = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
    print('Embedding model is ready✅')
    with open('resume.txt','r') as file:
        resume_dic=file.read()
    dict_resume=eval(resume_dic)
    with open('jobdescription.txt','r') as file:
        job_dic=file.read()
    dict_job=eval(job_dic)
    print(dict_job.keys())
    # print('inserting...')
    # insert('Skills',dict_resume['skills'],'s')
    # insert('Experience',dict_resume['experience'],'e')
    # insert('Education',dict_resume['education'],'edu')
    # print('Inserted....')
    # Example usage
    print('inserting..')
    texts=[dict_resume['skills'],dict_resume['education'],dict_resume['experience']]
    jd_text=[dict_job['skills'],dict_job['education'],dict_job['experience']]
    # insert('Resume',texts)
    # insert('JD',jd_text)
    print('Done✅')
    print('Check Here:http://localhost:6333/dashboard')
    print('-----------------------------------------------------------')
    print('Retriving Skills:')
    print(retrieve_point_by_id('Resume',1))
    print('-----------------------------------------------------------')
    print('Retriving education:')
    print(retrieve_point_by_id('Resume',2))
    print('-----------------------------------------------------------')
    print('Retriving experience:')
    print(retrieve_point_by_id('Resume',3))
    print('-------------JD:--------------')
    print('Retriving Skills:')
    print(retrieve_point_by_id('JD',1))
    print('-----------------------------------------------------------')
    print('Retriving education:')
    print(retrieve_point_by_id('JD',2))
    print('-----------------------------------------------------------')
    print('Retriving experience:')
    print(retrieve_point_by_id('JD',3))
    resume=retrieve_point_by_id('Resume',1)+retrieve_point_by_id('Resume',2)+retrieve_point_by_id('Resume',3)
    jd=retrieve_point_by_id('JD',1)+retrieve_point_by_id('JD',2)+retrieve_point_by_id('JD',3)
    print(get_result("You 'IT recruiter' help me to get the matching between the resume and job description and don't give any explanation.",f"Return the matching score in percentage and  0(If does not match)|1(if matches) for this data {resume} and {jd}"))
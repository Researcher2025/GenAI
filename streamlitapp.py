import streamlit as st
import pandas as pd
from Backend.main import clean_text,tokenize_text,remove_stopwords,stem_tokens,lemmatize_tokens
from qdrant_client import QdrantClient
from config import OPEN_API_KEY
from openai import OpenAI
import matplotlib.pyplot as plt
import fitz
import time
def get_result(system, prompt):
    try:
        
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
    except Exception as e:
        print(f"An error occurred while getting result: {e}")
        return None
def file_size_exceeds_limit(uploaded_file, limit_mb):
    # Convert the limit from MB to bytes
    limit_bytes = limit_mb * 1024 * 1024
    return uploaded_file.size > limit_bytes
def retrieve_point_by_id(collection_name, point_id):
    try:
        qdrant_client = QdrantClient("http://localhost:6333")
        result = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[point_id]
        )
        if result:
            for point in result:
                text = point.payload.get('text')
                job_title = point.payload.get('JD')
                if text and job_title:
                    return text, job_title
    except Exception as e:
        print(f"An error occurred while retrieving point: {e}")
    return None, None
def read_pdf(file_path):
    pdf_text = ""
    document = fitz.open(stream=file_path.read(),filetype='pdf')
    for page_num in range(document.page_count):
        page = document.load_page(page_num)
        pdf_text += page.get_text()
    return pdf_text
def normalize_text(text, classify=False,use_stemming=False):
    text = text.lower()  # Convert to lowercase
    text = clean_text(text)  # Clean text
    tokens = tokenize_text(text)  # Tokenize text
    tokens = remove_stopwords(tokens)  # Remove stop words
    if use_stemming:
        tokens = stem_tokens(tokens)  # Stem tokens
    else:
        tokens = lemmatize_tokens(tokens)  # Lemmatize tokens
    temp= ' '.join(tokens) 
    if classify:
        return temp
    formatted_data = "{'skills':'c,java,python,etc..','experience':'commaseperatedin string format','education':'education in string format'}"
    return get_result("You helping me in 'AI Resume Match maker' project",f"Parse the skills, education, experience into a Python dictionary for this data {temp}. Do not disturb the internal data. Convert it into a string format such as {formatted_data}. Ensure the format is ready to use with the eval() method and make it a single line.")
def extract_features(text, feature_type):
    system_prompt = f"""
    You are an AI that extracts information from text. Extract the {feature_type} section from the provided text.
    """
    user_prompt = f"""
    Extract the {feature_type} section from the following text:
    {text}
    """
    return get_result(system_prompt, user_prompt)
client = OpenAI(api_key=OPEN_API_KEY)
# Page configuration
st.set_page_config(
    page_title="AI Resume Match Maker",
    page_icon="https://raw.githubusercontent.com/tarun261003/PdfViewer/main/IS.png",
    layout="wide",
    initial_sidebar_state="auto",
)

# Custom CSS to style the app
st.markdown("""
    <style>
       .main {
            background-color: #ffffff; /* White background */
            color: #333333; /* Dark gray font color */
            padding: 2rem;
            font-family: 'Open Sans', sans-serif; /* Apply Open Sans font */
        }
       .css-1d391kg {
            background-color: #3498db; /* Blue sidebar */
            color: #ffffff;
            padding: 1rem;
            font-family: 'Open Sans', sans-serif; /* Apply Open Sans font */
        }
       .css-1d391kg.css-2trqyj {
            margin-bottom: 1rem;
        }
       .css-1d391kg.css-1x8cf1d {
            background-color: #f1c40f; /* Orange button */
            color: #ffffff;
            padding: 0.5rem 1rem;
            border: none;
            border-radius: 0.25rem;
            cursor: pointer;
            font-family: 'Open Sans', sans-serif; /* Apply Open Sans font */
        }
       .css-1d391kg.css-1x8cf1d:hover {
            background-color: #e67e00; /* Darker orange on hover */
        }
       .css-10trblm {
            font-size: 1.2rem;
            font-family: 'Open Sans', sans-serif; /* Apply Open Sans font */
        }
       .instructions-container {
            background-color: #f7f7f7; /* Light gray background for instructions */
            padding: 1rem;
            border-radius: 0.25rem;
            margin-bottom: 2rem;
            color: #333333; /* Dark gray font color for instructions */
            font-family: 'Open Sans', sans-serif; /* Apply Open Sans font */
        }
       .stTitle h1 {
            color: #ffffff!important; /* White color for the title */
            font-family: 'Open Sans', sans-serif; /* Apply Open Sans font */
        }
       .instructions-container h2 {
            color: #333333!important; /* Black color for the instructions heading */
            font-family: 'Open Sans', sans-serif; /* Apply Open Sans font */
        }
        .stAlert p {
            color: #ff0000 !important; /* Custom color for error messages */
        }
    </style>
""", unsafe_allow_html=True)

# Title of the app
st.markdown("""
  <div style="display: flex; align-items: center;">
    <img src="https://raw.githubusercontent.com/tarun261003/PdfViewer/main/IS.png" alt="Logo" style="width: 50px; height: 50px; margin-right: 10px;">
    <h1 style="text-align: center; color: #333333; margin-bottom: 0; font-family: 'Open Sans', sans-serif;">AI ResumeMatch Maker</h1>
  </div>
""", unsafe_allow_html=True)

# Create a container for the main area
main_area = st.container()

# Display instructions in the main area
with main_area:
    st.markdown('<div class="instructions-container">', unsafe_allow_html=True)
    st.markdown("<h2 style='color: #333333; font-family: 'Open Sans', sans-serif;'>Instructions</h2>", unsafe_allow_html=True)
    st.markdown("1. Upload your resume.", unsafe_allow_html=True)
    st.markdown("2. Choose an option:", unsafe_allow_html=True)
    st.markdown("   a) Upload a job description to match with.", unsafe_allow_html=True)
    st.markdown("   b) Match with existing job descriptions in the database.", unsafe_allow_html=True)
    st.markdown("3. Click the corresponding button.", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Create a sidebar for file uploaders and match buttons
sidebar = st.sidebar

# Input for resume
sidebar.header("Upload your files")
resume_upload = sidebar.file_uploader("Resume", type=["pdf", "doc", "docx"], help="Select your resume file")

# Option 1: Upload a job description to match with
job_description_upload = sidebar.file_uploader("Job Description", type=["pdf", "doc", "docx"], help="Select a job description file")
match_button_1 = sidebar.button("Match with uploaded JD", help="Click to match your resume with the uploaded job description")

# Option 2: Match with existing job descriptions in the database
match_button_2 = sidebar.button("Match with existing JDs", help="Click to match your resume with existing job descriptions in the database")

# Code to handle matching
if match_button_1:
    if resume_upload:
        if file_size_exceeds_limit(resume_upload, 2):
            st.error("The resume file exceeds the 2MB size limit. Please upload a smaller file.")
            resume_upload = None
    if job_description_upload:
        if file_size_exceeds_limit(job_description_upload, 2):
            st.error("The job description file exceeds the 2MB size limit. Please upload a smaller file.")
            job_description_upload = None
    if resume_upload and job_description_upload:
        try:
            resume_text = read_pdf(resume_upload)
            jd_text = read_pdf(job_description_upload)
            print('read')
            resume_dic=eval(normalize_text(resume_text))
            jd_dic=eval(normalize_text(jd_text))
            job_title=get_result("You 'IT recruiter' help me to get jobtile",f'Give me the Job title from {jd_text} donot give any additional information.')
            system_prompt = "You 'IT recruiter' help me to get the matching between the resume and job description and don't give any additional information."
            user_prompt = f"I want you to give only percentage matching score between resume and job description on the basis of title, skills, education, experience whose details are as follows:  with job title {job_title} resume skills which is: {resume_dic['skills']} with job description skills: {jd_dic['skills']}, resume experience: {resume_dic['experience']} with job description experience: {jd_dic['experience']} and finally resume education: {resume_dic['education']} with job description education: {jd_dic['education']}. Produce the output in '''overall percentage without including '%' and Matching skills between {resume_dic['skills']} and {jd_dic['skills']} both in tuple format''' 'Dont produce in extra additional information'."
            temp= get_result(system_prompt, user_prompt).split('\n')
            print(temp[0],temp[-1])
            score,matching_skills=temp[0],temp[-1]
            if score:
                try:
                    score = eval(score[-2:])
                    st.markdown(f'''<h2 style='color: #333333; font-family: 'Open Sans', sans-serif;>Match Score:{score}</h2>''',unsafe_allow_html=True)
                    temp[-1]=temp[-1].lower().replace('matching','').replace('skills','').replace(':','')
                    print(temp[-1])
                    st.markdown('''<h1 style='color: #333333; font-family: 'Open Sans', sans-serif;>Matching Skills:</h1>''',unsafe_allow_html=True)
                    elements = eval(temp[-1])

                    # Iterate through the elements and display them in four columns
                    columns = st.columns(4)

                    for index, element in enumerate(elements):
                        col_index = index % 4
                        columns[col_index].write(element)
                except ValueError as e:
                    print(temp)
                    st.error(f"An error occurred while converting score to integer: {e}")
        except Exception as e:
            st.error(f"An error occurred while reading or normalizing resume: {e}")
    else:
        st.error("Please upload your resume and a job description to match with.")

if match_button_2:
    if resume_upload:
        if(file_size_exceeds_limit(resume_upload,2)):
                st.error("The resume file exceeds the 2MB size limit. Please upload a smaller file.")
                resume_upload = None
        try:
            resume_text = read_pdf(resume_upload)
            normalize_resume = normalize_text(resume_text, classify=True)
            resume_skills= extract_features(normalize_resume, "skills")
            resume_experience = extract_features(normalize_resume, "experience")
            resume_education = extract_features(normalize_resume, "education")

            result = []
            for i in range(1, 11):
                text, job_title = retrieve_point_by_id('JDS', i)
                jd_text = eval(text)
                if jd_text and job_title:
                    resume = "".join(resume_skills) + "".join(resume_experience) + "".join(resume_education)
                    system_prompt = "You 'IT recruiter' help me to get the matching between the resume and job description and don't give any additional information."
                    user_prompt = f"I want you to give only percentage matching score between resume and job description on the basis of title, skills, education, experience whose details are as follows:  with job title {jd_text['Job Title']} resume skills which is: {resume_skills} with job description skills: {jd_text['Skills']}, resume experience: {resume_experience} with job description experience: {jd_text['Experience']} and finally resume education: {resume_education} with job description education: {jd_text['Experience']}. Produce the output in '''overall percentage without including '%' in the form of int type''' Dont produce in extra additional information."
                    score = get_result(system_prompt, user_prompt)
                    if score:
                        try:
                            if '%' in score:
                                score=score.replace('%','')
                                score=int(score[-1])
                            else:
                                score = int(score[-2:])
                            result.append([job_title, score])
                        except ValueError as e:
                            st.error(f"An error occurred while converting score to integer: {e}")

            result.sort(key=lambda x: x[1], reverse=True)
            top_5 = result[:5]

            st.markdown('''<h1 style="color: #333333; margin-bottom: 0; font-family: 'Open Sans', sans-serif;">Top 5 Resumes</h1>''', unsafe_allow_html=True)

            for i, (job_title, score) in enumerate(top_5):
                st.markdown(f'''<p style="color: #333333; margin-bottom: 0; font-family: 'Open Sans', sans-serif;">{i + 1}. {job_title}:{score}%</p>''',unsafe_allow_html=True)
                progress_bar = st.progress(0)
                progress_bar.progress(score / 100)
                # st.markdown(f"""<h1 style="color: #333333; margin-bottom: 0; font-family: 'Open Sans', sans-serif;"Score: {score}% </h1>""",unsafe_allow_html=True)
                time.sleep(0.5)  # Simulate some work being done

        except Exception as e:
            st.error(f"An error occurred while reading or normalizing resume: {e}")
    else:
        st.error("Please upload your resume to match with existing job descriptions.")
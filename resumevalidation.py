from main import read_pdf
from qdrant_client import QdrantClient
from main import normalize_text
from config import OPEN_API_KEY
from openai import OpenAI
def get_result(system,prompt):
    completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  temperature=0.2,
  n=1,
  messages=[
    {"role": "system", "content": system},
    {"role": "user", "content": prompt}
  ]
  
)
    return completion.choices[0].message.content
def retrieve_point_by_id(collection_name, point_id):
    try:
        result = qdrant_client.retrieve(
            collection_name=collection_name,
            ids=[point_id]
        )
        if result:
            for point in result:
                text=point.payload.get('text')
                job_title=point.payload.get('JD')
                return text,job_title
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
if __name__=="__main__":
    client = OpenAI(api_key=OPEN_API_KEY)
    qdrant_client = QdrantClient("http://localhost:6333")
    resume_text = read_pdf('./Resumes/Resume_Prasad.pdf')
    normalize_resume=normalize_text(resume_text,classify=True)
    result=[]
    for i in range(1,11):
        jd_text,job_title=retrieve_point_by_id('JDS',i)
        output_format={}
        score=int(get_result("You are an AI designed to assist an IT recruiter in evaluating resumes. Your task is to calculate the matching score out of 100 (don't add extra information) resume and job description (JD) text. The matching score should be a numerical representation of the similarity between the two texts, Return the results in a integer without any additional explanation.",f'Campare {normalize_resume} and {jd_text} give me the matching score out of 100.Dont give me any explanation.'))
        result.append([job_title,score])
    result.sort(key=lambda x : x[1],reverse=True)
    for i in result:
        print(i)
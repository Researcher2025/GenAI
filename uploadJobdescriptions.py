import os
from main import insert
from main import normalize_text,read_pdf
from config import OPEN_API_KEY
from qdrant_client import QdrantClient
from openai import OpenAI
import warnings
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from qdrant_client.http import models
def insert(collection_name, texts,type1,id):
        embeddings = embeddings_model.embed_documents(texts)

        # Check if the collection exists and create it if it doesn't
        # if qdrant_client.collection_exists(collection_name):
        #     qdrant_client.delete_collection(collection_name)

        # qdrant_client.create_collection(
        #     collection_name=collection_name,
        #     vectors_config=models.VectorParams(size=len(embeddings[0]), distance=models.Distance.COSINE)
        # )

        # Store embeddings in Qdrant using Upsert with integer point IDs
        points = [
            models.PointStruct(id=id,vector=embedding, payload={"text": texts[i],"JD":type1},)
            for i, embedding in enumerate(embeddings)
        ]
        qdrant_client.upsert(collection_name=collection_name, points=points)
if __name__=="__main__":
    qdrant_client = QdrantClient("http://localhost:6333")  # Replace with your Qdrant URL
    i=0
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
    pdfs=os.listdir('./jd')
    while i<15:
         text=normalize_text(read_pdf(f'./jd/{pdfs[i]}'),classify=True)
         insert('JDS',[text],pdfs[i].rstrip('.pdf'),i+1)
         i+=1
    print('Check Here:http://localhost:6333/dashboard')
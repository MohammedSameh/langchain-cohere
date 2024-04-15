# uvicorn main:app --reload
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel
from chromadb import EmbeddingFunction
from langchain.llms import GPT4All, LlamaCpp, Cohere
from langchain.chains import RetrievalQA
import os
from langchain.vectorstores import Chroma
from langchain.embeddings.cohere import CohereEmbeddings
import argparse
import time

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

class Item(BaseModel):
    text: str

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
#                                                  'using the power of LLMs.')
#     parser.add_argument("--hide-source", "-S", action='store_true',
#                         help='Use this flag to disable printing of source documents used for answers.')

#     parser.add_argument("--mute-stream", "-M",
#                         action='store_true',
#                         help='Use this flag to disable the streaming StdOut callback for LLMs.')

#     return parser.parse_args()

app = FastAPI()

# args = parse_arguments()

target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

# Replace 'your_api_key' with your actual API key
cohere_api_key = os.environ.get('COHERE_API_KEY') 
print(cohere_api_key)

embeddings = CohereEmbeddings(cohere_api_key=cohere_api_key, model = "multilingual-22-12")

# Replace 'your_persist_directory' with the path to your persist directory
persist_directory = os.environ.get('PERSIST_DIRECTORY')
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
llm = Cohere(model="command-nightly", temperature=0.9)

# Create a RetrievalQA instance
print("Creating RetrievalQA instance...")
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=retriever, return_source_documents= not False)
print("RetrievalQA instance created.")

@app.get("/")
async def root():
    return {"message": "predict.py"}

@app.post("/predict")
async def predict(item: Item):
    start = time.time()
    res = qa(item.text)
    answer, docs = res['result'], [] if False else res['source_documents']
    end = time.time()
        
    source = [
        {
            "source": document.metadata["source"], 
            "content": document.page_content
        } 
        for document in docs
        ]

    for document in docs:
        print("\n> " + document.metadata["source"] + ":")
        print(document.page_content)
    return {        
        "response": answer,
        "response time": f"Answer (took {round(end - start, 2)} s.):",
        "source": source
        }
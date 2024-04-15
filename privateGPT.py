#!/usr/bin/env python3
# pip install llama-cpp-python
# pip install cohere
# langchain 0.0.325 working for LlamaCpp but not for cohere
# pip install --upgrade langchain==0.0.335
# pip install --upgrade langchain
from dotenv import load_dotenv
# import dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp, Cohere
import chromadb

import os
import argparse
import time
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# from langchain_cohere import ChatCohere
# from langchain_core.messages import HumanMessage
# from langchain_core.callbacks import BaseCallbackManager

from langchain.embeddings.cohere import CohereEmbeddings

# from getpass import getpass

if not load_dotenv():
    print("Could not load .env file or it is empty. Please check if it exists and is readable.")
    exit(1)

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
model_n_batch = int(os.environ.get('MODEL_N_BATCH',8))
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

cohere_api_key = os.environ.get('COHERE_API_KEY') 

from constants import CHROMA_SETTINGS

def main():
    # Parse the command line arguments
    args = parse_arguments()
    # embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    cohere_api_key = os.environ.get('COHERE_API_KEY') 
    embeddings = CohereEmbeddings(model = "multilingual-22-12")
    # chroma_client = chromadb.PersistentClient(settings=CHROMA_SETTINGS , path=persist_directory)
    chroma_client = chromadb.PersistentClient(path=persist_directory)
#########################################################################################################
    
    loader=TextLoader("C:/Users/sesa652756/Downloads/0.txt")
    document=loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=0)
    texts = text_splitter.split_documents(document)

##########################################################################################################





    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS, client=chroma_client)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # retriever = Chroma.from_documents(texts,embeddings)
    # activate/deactivate the streaming StdOut callback for LLMs
    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    # callbacks_manager = BaseCallbackManager(handlers=callbacks)
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            #llm = LlamaCpp(model_path=model_path, max_tokens=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks, verbose=False)
            llm = LlamaCpp(model_path=model_path,n_ctx=2048, n_batch=model_n_batch,callbacks=callbacks, verbose=False)
        case "GPT4All":
            #llm = GPT4All(model=model_path, max_tokens=model_n_ctx, backend='gptj', n_batch=model_n_batch, callbacks=callbacks, verbose=False)
            llm = GPT4All(model=model_path,n_threads=1)
        case "Cohere":                        
            llm = Cohere(model="command-nightly", temperature=0.9)
        case _default:
            # raise exception if model_type is not supported
            raise Exception(f"Model type {model_type} is not supported. Please choose one of the following: LlamaCpp, GPT4All")
         
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff",retriever=retriever, return_source_documents= not args.hide_source)
    
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break
        if query.strip() == "":
            continue

        # Get the answer from the chain
        start = time.time()
        res = qa(query)
        
        # i try here respons of LLM without from_chain_type
        # answer=llm("what is schneider electric in max 100 words")


        answer, docs = res['result'], [] if args.hide_source else res['source_documents']
        end = time.time()
        

        # Print the result
        print("\n\n> Question:")
        print(query)
        print(f"\n> Answer (took {round(end - start, 2)} s.):")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)

def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()


if __name__ == "__main__":
    main()

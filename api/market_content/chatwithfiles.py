from langchain_community.document_loaders import PyPDFLoader, TextLoader, S3DirectoryLoader, S3FileLoader, Docx2txtLoader
from fastapi import FastAPI, Body, HTTPException, APIRouter, Request
from pydantic import BaseModel, validator
from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.prompts import PromptTemplate, ChatPromptTemplate, HumanMessagePromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
import tempfile
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI,ChatOpenAI
from config import  chroma_server_client
from langchain_community.callbacks import get_openai_callback
from config import GPT4o_mini
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN
from fastapi import FastAPI, HTTPException,Depends, Header,Query
from config import embeddings,llm_screener
# import PyPDF2
import tiktoken

load_dotenv(override=True)


# client=chroma_server_client

router = APIRouter()

# llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def count_tokens(text, model_name="gpt-4o-mini"):
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)



AI_KEY=os.getenv('AI_KEY')

async def authenticate_ai_key(x_api_key: str = Header(...)):
    if x_api_key != AI_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )

#document chat
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import TextLoader,PyPDFLoader
import os
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.memory import ConversationBufferMemory,PostgresChatMessageHistory
import boto3
import tempfile
import os
import chromadb
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from dotenv import load_dotenv
load_dotenv(override=True)

psql_url=os.getenv('DATABASE_URL')
c_bucket=os.getenv("content_bucket")



s3 = boto3.client('s3',
                  aws_access_key_id=os.getenv("access_key"),
                  aws_secret_access_key=os.getenv('sect_access_key'),
                  )

# embeddings=OpenAIEmbeddings()
#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="chromadb_for_docs")

# Define the variable for the postgres IP address

# def save_temporary_file_from_s3(bucket_name, object_key, temp_folder='temp_docs'):
#     """
#     Save a temporary file from an S3 bucket to the local machine within a specific folder.
    
#     Parameters:
#     - bucket_name: The name of the S3 bucket.
#     - object_key: The key of the file in the S3 bucket.
#     - temp_folder: The name of the folder within the project directory to save the temporary file.
    
#     Returns:
#     - temp_file_path: The local file path of the saved temporary file.
#     """
#     # Create the folder if it doesn't exist
#     os.makedirs(temp_folder, exist_ok=True)
    
#     # Define the temporary file path
#     temp_file_path = os.path.join(temp_folder, object_key.split('/')[-1])  # Use the last part of the object key as the filename
    
#     try:
#         # Download the file from S3 to the temporary file
#         s3.download_file(bucket_name, object_key, temp_file_path)
        
#         print(f"Successfully saved temporary file '{object_key}' from S3 to '{temp_file_path}'")
#         return temp_file_path
#     except Exception as e:
#         print(f"Error saving temporary file '{object_key}' from S3: {e}")
#         return None

import tempfile

def save_temporary_file_from_s3(bucket_name, object_key):
    """
    Save a temporary file from an S3 bucket to the local machine.

    Parameters:
    - bucket_name: The name of the S3 bucket.
    - object_key: The key of the file in the S3 bucket.

    Returns:
    - temp_file_path: The local file path of the saved temporary file.
    """
    try:
        # Initialize S3 client
       

        # Create a named temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file_path = temp_file.name
        temp_file.close()  # Close the file to release the lock

        # Download the file from S3 to the temporary file
        s3.download_file(bucket_name, object_key, temp_file_path)
        print(f"Successfully saved temporary file '{object_key}' from S3 to '{temp_file_path}'")

        return temp_file_path

    except Exception as e:
        print(f"Error saving temporary file '{object_key}' from S3: {e}")
        return None


def collection_exists(file_name, collections):
    return any(collection.name == file_name for collection in collections)

# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def create_doc_embeddings(object_key, file_name):
    #print(file_name)
    #print(object_key)
    # client = chromadb.PersistentClient(path="chromadb_for_docs")
    
    #if not collection_exists(file_name, client.list_collections()):
    try:
        client.get_collection(file_name)
        return {"message": "Collection already exists", "collection_name": file_name}
    except:
        bucket_name = c_bucket
        file_path = save_temporary_file_from_s3(bucket_name, object_key)
        #print(file_path)
        file_extension = object_key.split('.')[-1]
        
        # Check the file extension and route the logic accordingly
        if file_extension.lower() == 'pdf':
            print("This is a PDF file.")
            # with open(file_path, 'rb') as file:
            #     reader = PyPDF2.PdfReader(file)
            #     if reader > 20:
            #          raise HTTPException(status_code=400, detail="The PDF you're trying to upload exceeds the 20-page limit. Please upload a shorter document.")
            loader = PyPDFLoader(file_path)
            
            # Add logic for handling PDF files here
        elif file_extension.lower() == 'txt':
            print("This is a TXT file.")
            loader=TextLoader(file_path)
           
        elif file_extension.lower() == 'docx':
            print("This is a DOCX file.")
            loader=Docx2txtLoader(file_path)
        document = loader.load()
        text=document[0].page_content
        if count_tokens(text)>10000:
            return None
            #raise HTTPException(status_code=400, detail="The Document you're trying to upload exceeds the word limit. Please upload a shorter document.")
        else:
            splitter=RecursiveCharacterTextSplitter(chunk_size=2000,chunk_overlap=200)
            docs=splitter.split_documents(document)
            # docs = []

            # for doc in splitter.split_documents(document):
            #     docs.append(doc.page_content)

            #print(len(docs))
            #print(len(docs))

            collection = client.create_collection(name=file_name)
            #print("collection_created")
            count = collection.count()
            ids = []

            for count in range(1, len(docs)+1):
                ids.append(str(file_name) + str(count))

            # embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
            vs= Chroma(
                client=client,
                collection_name=file_name,
                embedding_function=embeddings)
            #print("adding started")
            vs.add_documents(documents=docs, ids=ids)
            # collection.add(
            #         documents=docs,
            #         ids=ids
            #     )
            # for index, doc in enumerate(docs):
            #     count += 1
            #     collection.add(
            #         documents=str(doc.page_content),
            #         ids=[str(file_name) + str(count)]
            #     )
            #     print('added')
            
            return {"message": "Collection created", "collection_name": file_name}
    # else:
    #     return {"message": "Collection already exists", "collection_name": file_name}


class Attachment(BaseModel):
    object_key: str
    file_name: str

@router.post("/add_attachments")
async def add_attachment(attachment: Attachment,ai_key_auth: str = Depends(authenticate_ai_key)):
    result = create_doc_embeddings(attachment.object_key, attachment.file_name)
    print(result)
    if result is None:
        raise HTTPException(status_code=400, detail="The Document you're trying to upload exceeds the word limit. Please upload a shorter document.")
    else:
        return result
######




def documentchat(file_name,question,session_id):
    with get_openai_callback() as cb:
        #llm = ChatOpenAI(temperature=0.3, model='gpt-4o-mini')
        db4 = Chroma(
            client=client,
            collection_name=file_name,
            embedding_function=embeddings,
        )
        template = """Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just use the provided context and answer.
        Don't try to make up an answer.Dont start with provided context.
        {context}

        Question: {question}
        Answer: 
        """
        prompt = PromptTemplate.from_template(template)


        history = PostgresChatMessageHistory(
            connection_string = psql_url,
            session_id=session_id,
        )
        memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, k=1,chat_memory=history)
        # qa = RetrievalQA.from_chain_type(
        #     llm=llm, chain_type="stuff", retriever=db4.as_retriever(),
        #     memory=memory,
        #     #return_source_documents=True

        # )
        # response = qa(question)

        qa = RetrievalQA.from_chain_type(
            llm=llm_screener, chain_type="stuff", retriever=db4.as_retriever(search_kwargs={"k": 10}),chain_type_kwargs={"prompt": prompt},memory=memory
            #return_source_documents=True

        )
        response = qa({"query": question})
        return {"response": response['result'],
            "Total_Tokens": cb.total_tokens,
            "Prompt_Tokens": cb.prompt_tokens,
            "Completion_Tokens": cb.completion_tokens,
            #"Total Cost (USD)": cb.total_cost}
            }


class QuestionRequest(BaseModel):
    file_name: str
    question: str


@router.post("/documentchat")
async def chat_with_document(q: QuestionRequest, session_id:str,ai_key_auth: str = Depends(authenticate_ai_key)):
    try:
        response = documentchat(q.file_name, q.question,session_id)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"
headers = {"Authorization": f"Bearer {os.getenv('HUGGINGFACEHUB_API_TOKEN')}"}

import requests,re

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        raise HTTPException(status_code=response.status_code, detail=response.json())
    return response.json()

def extract_question(text):
    matches = re.findall(r'question:\s*(?!<question>)(.*)', text)
    valid_questions = [match for match in matches if len(match.split()) > 3]
    if valid_questions:
        return valid_questions
    return []



text1="""Nvidia Corporation[a][b] (/ɛnˈvɪdiə/, en-VID-ee-ə) is an American multinational corporation and technology company headquartered in Santa Clara, California, and incorporated in Delaware.[5] It is a software and fabless company which designs and supplies graphics processing units (GPUs), application programming interfaces (APIs) for data science and high-performance computing, as well as system on a chip units (SoCs) for the mobile computing and automotive market. Nvidia is also a dominant supplier of artificial intelligence (AI) hardware and software.[6][7][8]

Nvidia's professional line of GPUs are used for edge-to-cloud computing and in supercomputers and workstations for applications in such fields as architecture, engineering and construction, media and entertainment, automotive, scientific research, and manufacturing design.[9] Its GeForce line of GPUs are aimed at the consumer market and are used in applications such as video editing, 3D rendering and PC gaming. In the second quarter of 2023, Nvidia had a market share of 80.2% in the discrete desktop GPU market.[10] The company expanded its presence in the gaming industry with the introduction of the Shield Portable (a handheld game console), Shield Tablet (a gaming tablet) and Shield TV (a digital media player), as well as its cloud gaming service GeForce Now.[11]

In addition to GPU design and manufacturing, Nvidia provides the CUDA software platform and API that allows the creation of massively parallel programs which utilize GPUs.[12][13] They are deployed in supercomputing sites around the world.[14][15] In the late 2000s, Nvidia had moved into the mobile computing market, where it produces Tegra mobile processors for smartphones and tablets as well as vehicle navigation and entertainment systems.[16][17][18] Its competitors include AMD, Intel,[19] Qualcomm[20] and AI accelerator companies such as Cerebras and Graphcore. It also makes AI-powered software for audio and video processing, e.g. Nvidia Maxine.[21]

"""
from pypdf import PdfReader

def top20questions(questions):
    
    input_prompt = f"""
    pick out the 20 most important and insightful questions from the following list of questions: {questions}
    Give output ONLY in this format and nothing else:
        'question: <question>'
    """

    data = query({
        "inputs": input_prompt,
        "parameters": {
            "max_new_tokens": 15000,
            "top_p": 0.1,
            "temperature": 0.8
        }
    })

    extracted_text = extract_question(data[0]["generated_text"])
    


    return extracted_text       


def generate_question(docs):
    all_questions=[]
    for doc  in docs:
        input_prompt = f"""
        data: {doc}
        generate ONLY 10 QUESTIONS from the previously given data. 
        The questions should be complex but keep it shorter.
        The questions should not be repetitive, ask different kinds of questions.
        Give output ONLY in this format and nothing else:
        'question: <question>'
        """

        data = query({
            "inputs": input_prompt,
            "parameters": {
                "max_new_tokens": 15000,
                "top_p": 0.1,
                "temperature": 0.8
            }
        })

        extracted_text = extract_question(data[0]["generated_text"])
        all_questions.extend(extracted_text)

        # final_questions=llm.invoke(f"pick out the 10 most important questions from the following list of questions: {all_questions}")
        final_questions=top20questions(all_questions)
    return final_questions       


class DocumentRequest(BaseModel):
    object_key: str

# @router.post("/document_suggestions")
# def document_suggestions(attachment: DocumentRequest):
#     bucket_name = "fruit-gpt"
#     object_key = attachment.object_key
#     file_path = save_temporary_file_from_s3(bucket_name, object_key)
#     file_extension = object_key.split('.')[-1].lower()
    
#     try:
#         # Check the file extension and route the logic accordingly
#         if file_extension.lower() == 'pdf':
#             print("This is a PDF file.")
#             loader = PyPDFLoader(file_path)
#             reader = PdfReader(file_path)
#             number_of_pages = len(reader.pages)
#             text=""
#             for page in reader.pages[2:10]:
#                 text += page.extract_text()
            
#             # Add logic for handling PDF files here
#         elif file_extension.lower() == 'txt':
#             print("This is a TXT file.")
#             loader=TextLoader()
            
#         elif file_extension.lower() == 'docx':
#             print("This is a DOCX file.")
#             loader=Docx2txtLoader()
#     # document = loader.load()
#     # print("this doc content")
#     # print(document.page_content)
#         else:
#             raise HTTPException(status_code=400, detail="Unsupported file extension")

#         response = generate_question(text)
#         return {"quetsions":response}

#     except FileNotFoundError:
#         raise HTTPException(status_code=404, detail="File not found")
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


@router.post("/document_suggestions")
def document_suggestions(attachment: DocumentRequest):
    bucket_name = "fruit-gpt"
    object_key = attachment.object_key
    file_path = save_temporary_file_from_s3(bucket_name, object_key)
    file_extension = object_key.split('.')[-1].lower()
    
    try:
        # Check the file extension and route the logic accordingly
        if file_extension.lower() == 'pdf':
            print("This is a PDF file.")
            loader = PyPDFLoader(file_path)
            
            # Add logic for handling PDF files here
        elif file_extension.lower() == 'txt':
            print("This is a TXT file.")
            loader=TextLoader()
           
        elif file_extension.lower() == 'docx':
            print("This is a DOCX file.")
            loader=Docx2txtLoader()
        
    # document = loader.load()
    # print("this doc content")
    # print(document.page_content)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file extension")
        document = loader.load()
        splitter=RecursiveCharacterTextSplitter(chunk_size=80000,chunk_overlap=200)
        docs=splitter.split_documents(document)

        response = generate_question(docs)
        return {"suggested_questions":response}

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


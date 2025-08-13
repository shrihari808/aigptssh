from fastapi import APIRouter
from pydantic import BaseModel, Field

from langchain_community.callbacks.manager import get_openai_callback
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from fastapi import FastAPI, HTTPException,Depends, Header,Query
from api.caching import query_similar_questions,add_questions, should_store
from api.fundamentals_rag.fundamental_chat2 import agent_with_session
from api.fundamentals_rag.corp import corp_agent_with_session
# from api.fundamentals_rag.company_bio import company_agent_with_session
from api.fundamentals_rag.screener import screen_stocks
from api.reddit_chat import reddit_rag
from api.news_rag.news_rag import web_rag,cmots_only
from api.youtube_rag.youtube_chat import yt_chat
import os
from starlette.status import HTTP_403_FORBIDDEN
from langchain_community.chat_message_histories.postgres import PostgresChatMessageHistory
from config import *
from streaming.streaming import combined_preprocessing



# Define the SQLAlchemy model
Base = declarative_base()
router=APIRouter()
psql_url=os.getenv('DATABASE_URL')

# api_keys=["gvcvcvxvkey"]
# def validate_api_key(api_key: str):
#     if api_key in api_keys:
#         return "valid"
#     else:
#         return None  # Invalid API key

# # Middleware to check for a valid API key and identify the user
# async def authenticate_api_key(api_key: str = Header(None)):
#     user_info = validate_api_key(api_key)
#     if user_info is None:
#         raise HTTPException(status_code=403, detail="Invalid API Key")
#     return user_info


OPENAI_API_KEY=os.getenv('OPENAI_API_KEY')
async def authenticate_ai_key(x_api_key: str = Header(...)):
    pass
    # if x_api_key != OPENAI_API_KEY:
    #     raise HTTPException(
    #         status_code=HTTP_403_FORBIDDEN,
    #         detail="Invalid or missing API Key",
    #     )

class InputText(BaseModel):
        input_text: str

class ChatbotResponse(Base):
    __tablename__ = 'response_history'
    #__tablename__ = 'response_history_test'

    id = Column(Integer, primary_key=True, autoincrement=True)
    question = Column(String)
    response = Column(String)
    count = Column(Integer, default=1)
    Total_Tokens = Column(Integer)
    Prompt_Tokens = Column(Integer)
    Completion_Tokens = Column(Integer)

# Configure the database connection
DATABASE_URL = psql_url
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create the table if it does not exist
Base.metadata.create_all(bind=engine)

# Function to get a database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def add_to_memory(question,response,session_id):
    # Initialize the PostgresChatMessageHistory
    history = PostgresChatMessageHistory(
        connection_string = psql_url,
        session_id=session_id,
    )
    history.add_user_message(question)
    history.add_ai_message(response)



@router.post("/chatbot/")
async def chatbot_endpoint(input_data: InputText, session_id: str, market: str = Query("IND", title="Market", description="Choose between 'US' or 'IND'"), flag: str = Query(None, title="Flag", description="Optional flag to specify the type of response"), ai_key_auth: str = Depends(authenticate_ai_key), db: Session = Depends(get_db)):
    #default_response = "I'm sorry, but as an AI Stock market Assistant, my main focus is on providing information and insights related to stocks, financial statements, and market news. If you have any questions related to those topics, feel free to ask!"
    default_response="The search query you're trying to use does not appear to be related to the Indian financial markets. Please ensure your query focuses on topics like finance, investing, stocks, or other related subjects to maintain platform quality. If your query is relevant but isn’t yielding the desired results, consider refining it to improve accuracy and alignment with financial market topics"
    valid,_,_,_= await combined_preprocessing(input_data.input_text,session_id)
    # if predict_value(input_data.input_text) == 0:
    if valid ==2:
        return {"Response": default_response}
    #REMOVED INITIAL

    # Checks if the question is already in the vector db using similariry seach
    # similar_question = query_similar_questions(input_data.input_text,'fundamental_cache')
    # print(f"Similar Question: {similar_question}")
    # cached_response = db.query(ChatbotResponse).filter(ChatbotResponse.question == similar_question).first()
    # print(f"Cached Response: {cached_response}")

    # if similar_question and cached_response:
    #     # looks up for the corresponding response for
    #     cached_response.count += 1
    #     db.commit()
    #     add_to_memory(input_data.input_text, cached_response.response, session_id=session_id)
    #     return {
    #         "Response": cached_response.response,
    #         #"Popularity": cached_response.count,
    #         "Total_Tokens": cached_response.Total_Tokens,
    #         "Prompt_Tokens": cached_response.Prompt_Tokens,
    #         "Completion_Tokens": cached_response.Completion_Tokens,
    #     }
    if flag == "news":
        response = cmots_only(input_data.input_text,session_id)
        return response
    if flag == "news_bing":
        response = await web_rag(input_data.input_text, session_id)
        return response
    if flag == "reddit":
        response = await reddit_rag(input_data.input_text,session_id)
        return response
    if flag == "youtube":
        response = await yt_chat(input_data.input_text,session_id)
        return response
    if flag=="screener":
        response = screen_stocks(input_data.input_text,session_id,db)
        return response
    
    if flag == 'corp':
        with get_openai_callback() as cb:
                prompt = input_data.input_text
                # if flag == "news":
                #     response = llm(prompt)
                # else:
                response = corp_agent_with_session(session_id, prompt, market)
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")
        return {"Response": response,
                    "Total_Tokens": cb.total_tokens,
                    "Prompt_Tokens": cb.prompt_tokens,
                    "Completion_Tokens": cb.completion_tokens,
                    # "Total Cost (USD)": cb.total_cost}
                    }
    
    # if flag == 'bio':
    #     with get_openai_callback() as cb:
    #             prompt = input_data.input_text
    #             # if flag == "news":
    #             #     response = llm(prompt)
    #             # else:
    #             response = company_agent_with_session(session_id, prompt, market)
    #             print(f"Total Tokens: {cb.total_tokens}")
    #             print(f"Prompt Tokens: {cb.prompt_tokens}")
    #             print(f"Completion Tokens: {cb.completion_tokens}")
    #             print(f"Total Cost (USD): ${cb.total_cost}")
    #     return {"Response": response,
    #                 "Total_Tokens": cb.total_tokens,
    #                 "Prompt_Tokens": cb.prompt_tokens,
    #                 "Completion_Tokens": cb.completion_tokens,
    #                 # "Total Cost (USD)": cb.total_cost}
    #                 }


    else:
        similar_question = query_similar_questions(input_data.input_text,'fundamental_cache')
        print(f"Similar Question: {similar_question}")
        cached_response = db.query(ChatbotResponse).filter(ChatbotResponse.question == similar_question).first()
        print(f"Cached Response: {cached_response}")

        if similar_question and cached_response:
            # looks up for the corresponding response for
            cached_response.count += 1
            db.commit()
            add_to_memory(input_data.input_text, cached_response.response, session_id=session_id)
            return {
                "Response": cached_response.response,
                #"Popularity": cached_response.count,
                "Total_Tokens": cached_response.Total_Tokens,
                "Prompt_Tokens": cached_response.Prompt_Tokens,
                "Completion_Tokens": cached_response.Completion_Tokens,
            }
        else:
            with get_openai_callback() as cb:
                prompt = input_data.input_text
                # if flag == "news":
                #     response = llm(prompt)
                # else:
                response = agent_with_session(session_id, prompt, market)
                print(f"Total Tokens: {cb.total_tokens}")
                print(f"Prompt Tokens: {cb.prompt_tokens}")
                print(f"Completion Tokens: {cb.completion_tokens}")
                print(f"Total Cost (USD): ${cb.total_cost}")
                # Store the new response in the cache
                print("should_store:", should_store(input_data.input_text))
                if should_store(input_data.input_text):
                    new_response = ChatbotResponse(
                        question=input_data.input_text,
                        response=response,
                        count=1,
                        Total_Tokens=cb.total_tokens,  # Populate the new columns
                        Prompt_Tokens=cb.prompt_tokens,
                        Completion_Tokens=cb.completion_tokens,
                    )
                    db.add(new_response)
                    db.commit()
                    add_questions(input_data.input_text,'fundamental_cache')

            return {"Response": response,
                    "Total_Tokens": cb.total_tokens,
                    "Prompt_Tokens": cb.prompt_tokens,
                    "Completion_Tokens": cb.completion_tokens,
                    # "Total Cost (USD)": cb.total_cost}
                    }
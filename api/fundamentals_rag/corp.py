import re
import json
from langchain.prompts import MessagesPlaceholder
from dotenv import load_dotenv
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from fuzzywuzzy import process
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from fastapi import Response, APIRouter
from langchain_core.output_parsers import StrOutputParser
import datetime
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.schema import HumanMessage, SystemMessage
from config import GPT3_16k,GPT4o_mini
import requests
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories.postgres import PostgresChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
import os
import concurrent.futures
from api.fundamentals_rag.fundamental_chat2 import find_stock_code
from fastapi import FastAPI, APIRouter
import httpx
from langchain.callbacks import get_openai_callback
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException,Depends, Header,Query
from streaming.streaming import insert_credit_usage
from config import llm_stream
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from api.caching import add_questions,verify_similarity,query_similar_questions,should_store
import aiohttp
import asyncio
from fastapi.responses import StreamingResponse

ip_address=os.getenv("PG_IP_ADDRESS")

token=os.getenv('CMOTS_BEARER_TOKEN')
psql_url=os.getenv('DATABASE_URL')

# client=chroma_server_client

DATABASE_URL = psql_url
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
# Create the table if it does not exist
Base.metadata.create_all(bind=engine)


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

def find_stock_code_(stock_name):
    df = pd.read_csv("csvdata/6000stocks.csv")

    # Extract the company names and codes from the DataFrame
    company_names = df["Company Name"].tolist()
    company_codes = df["Company Code"].tolist()
    company_symbols =df['co_symbol'].tolist()
    
    threshold=80
    # Use fuzzy matching to find the closest match to the input stock name
    match = process.extractOne(stock_name, company_names)
    match1= process.extractOne(stock_name, company_symbols)

   
    if match and match[1] >= threshold:
        idx = company_names.index(match[0])
        print(company_codes[idx])
        return company_codes[idx]
    
    # Check if match in company symbols meets the threshold
    elif match1 and match1[1] >= threshold:
        idx = company_symbols.index(match1[0])
        print(company_codes[idx])
        return company_codes[idx]

    else:
        return None


def get_nse_annocements(stock_name):
    url = f"http://airrchipapis.cmots.com/api/{stock_name}Announcement"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        data = response.json()

        formatted_data_list = []
        if 'data' in data and isinstance(data['data'], list):
            up_data_list = data['data'][:15]

            for nse_data in reversed(up_data_list):
                if isinstance(nse_data, dict):
                    formatted_data = {
                        "company_name": nse_data.get('lname', ''),
                        "caption": nse_data['caption'],
                        # "Purpose": nse_data['memo'],
                        "Date": nse_data['date'].split('T')[0],
                        # "Agenda": nse_data['agenda']
                        # Other fields...
                    }
                    formatted_data_list.append(formatted_data)

        return formatted_data_list

async def get_nse_announcements_async(stock_name):
    # Construct the API URL
    url = f"http://airrchipapis.cmots.com/api/{stock_name}Announcement"
    headers = {"Authorization": f"Bearer {token}"}

    # Make an async GET request
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return {"message": f"Failed to fetch data, status code: {response.status_code}"}

    data = response.json()

    # Initialize the formatted data list
    formatted_data_list = []
    
    if 'data' in data and isinstance(data['data'], list):
        # Get the latest 15 announcements
        up_data_list = data['data'][:15]

        # Reverse the order and format each announcement
        for nse_data in reversed(up_data_list):
            if isinstance(nse_data, dict):
                formatted_data = {
                    "company_name": nse_data.get('lname', ''),
                    "caption": nse_data.get('caption', ''),
                    "Date": nse_data.get('date', '').split('T')[0],
                }
                formatted_data_list.append(formatted_data)

    return formatted_data_list



class TickerCheckInput(BaseModel):
    stockname: str = Field(..., description="name of the stock present in the query")
    #prompt: str = Field(..., description="elaborate the provided prompt for similarity search")


class nse_bse_annocements(BaseTool):
    name = "corporate_actions2"
    description = "Useful for when you need to get only NSE ,BSE announcements"

    def _run(self, stockname: str):
        #print("i'm running")
        headlines_response = get_nse_annocements(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res=await get_nse_announcements_async(stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = TickerCheckInput

###############################################


def get_today_results():
    url = f"http://airrchipapis.cmots.com/api/Today-Results"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        data = response.json()
        if data['data']:
            formatted_data_list = []
            if 'data' in data and isinstance(data['data'], list):
                up_data_list = data['data']

                for nse_data in reversed(up_data_list):
                    if isinstance(nse_data, dict):
                        formatted_data = {
                            "company_name": nse_data.get('co_name', ''),
                            # "symbol": nse_data['caption'],
                            # "Purpose": nse_data['memo'],
                            "Result_Date": nse_data['ResultDate'].split('T')[0],
                            # "Agenda": nse_data['agenda']
                            # Other fields...
                        }
                        formatted_data_list.append(formatted_data)

            return formatted_data_list

    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        print(response.text)

async def get_today_results_as():
    # Construct the API URL
    url = "http://airrchipapis.cmots.com/api/Today-Results"
    headers = {"Authorization": f"Bearer {token}"}

    # Make an async GET request
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return {"message": f"Failed to fetch data, status code: {response.status_code}"}

    data = response.json()

    # Process the response data
    if 'data' in data and isinstance(data['data'], list):
        up_data_list = data['data']
        formatted_data_list = []

        # Reverse the order and format the data
        for nse_data in reversed(up_data_list):
            if isinstance(nse_data, dict):
                formatted_data = {
                    "company_name": nse_data.get('co_name', ''),
                    "Result_Date": nse_data.get('ResultDate', '').split('T')[0],
                }
                formatted_data_list.append(formatted_data)

        return formatted_data_list

    return {"message": "No data found"}


class today_results(BaseTool):
    name = "today_results"
    description = "Useful for when you need to information of today results or which companies going to announce results today"

    def _run(self, stockname: str):
        print("i'm running")
        response = get_today_results()

        return response

    async def _arun(self, stockname: str):
        res=await get_today_results_as()
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = TickerCheckInput

#####################################

def get_dilist():
    url = f"http://airrchipapis.cmots.com/api/DeListed"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        data = response.json()

        formatted_data_list = []
        if 'data' in data and isinstance(data['data'], list):
            up_data_list = data['data']

            for dil_data in reversed(up_data_list):
                if isinstance(dil_data, dict):
                    formatted_data = {
                        "company_name": dil_data.get('CO_NAME', ''),
                        # "symbol": dil_data['caption'],
                        # "Purpose": dil_data['memo'],
                        "From date": dil_data['FromDate'].split('T')[0],
                        # "note": dil_data['note']
                        # Other fields...
                    }
                    formatted_data_list.append(formatted_data)

        return formatted_data_list

async def get_dilist_as():
    # API endpoint
    url = "http://airrchipapis.cmots.com/api/DeListed"
    headers = {"Authorization": f"Bearer {token}"}

    # Make an async GET request
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        return {"message": f"Failed to fetch data, status code: {response.status_code}"}

    # Parse the response data
    data = response.json()
    formatted_data_list = []

    if 'data' in data and isinstance(data['data'], list):
        up_data_list = data['data']

        # Process the data
        for dil_data in reversed(up_data_list):
            if isinstance(dil_data, dict):
                formatted_data = {
                    "company_name": dil_data.get('CO_NAME', ''),
                    "From date": dil_data.get('FromDate', '').split('T')[0],
                }
                formatted_data_list.append(formatted_data)

    return formatted_data_list

class delisted_companies(BaseTool):
    name = "delisted_companies"
    description = "Useful for when you need to information about delisted companies in NSE AND BSE"

    def _run(self, stockname: str):
        print("i'm running")
        response = get_dilist()

        return response

    async def _arun(self, stockname: str):
        res= await get_dilist_as()
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = TickerCheckInput


#############################################

def get_agm_egm(type,stock_name):
    code=int(find_stock_code(stock_name))
    print(code)
    if code is None:
        return "No data for this stock. Only for Indian market stocks."
    if type ==1:
        url = f"http://airrchipapis.cmots.com/api/AGM/{code}"
    else:
        url = f"http://airrchipapis.cmots.com/api/EGM/{code}"


    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        data = response.json()
        if (data.get("data") == [] or data.get("data") is None):
            return "No data available "
        else:
            agm_data = data['data'][-1]
            formatted_data = {
                "company_name": agm_data['co_name'],
                #"symbol": agm_data['symbol'],
                "Purpose": agm_data['Purpose'],
                "GM date": agm_data['GMdate'].split('T')[0],
                "Description": agm_data['Description']

            }

        return formatted_data

async def get_agm_egm_as(type, stock_name):
    code = int(find_stock_code(stock_name))
    print(code)

    if code is None:
        return "No data for this stock. Only for Indian market stocks."

    # Determine the URL based on the type
    if type == 1:
        url = f"http://airrchipapis.cmots.com/api/AGM/{code}"
    else:
        url = f"http://airrchipapis.cmots.com/api/EGM/{code}"

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make an async GET request
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code != 200:
        return f"Error: Failed to fetch data, status code: {response.status_code}"

    # Parse the response JSON
    data = response.json()
    if not data.get("data"):
        return "No data available"

    # Extract the latest AGM/EGM data
    agm_data = data["data"][-1]
    formatted_data = {
        "company_name": agm_data.get("co_name", ""),
        "Purpose": agm_data.get("Purpose", ""),
        "GM date": agm_data.get("GMdate", "").split('T')[0],
        "Description": agm_data.get("Description", ""),
    }

    return formatted_data


class meetCheckInput(BaseModel):
    type: int = Field(..., description="if user is asking about AGM set type as 1 or EGM set type as 2")
    stockname: str = Field(..., description="name of the stock present in the query")
    #prompt: str = Field(..., description="elaborate the provided prompt for similarity search")


class agm_egm(BaseTool):
    name = "agm_egm"
    description = "Useful for when you need to get information about AGM and EGM meetings about a company"

    def _run(self, type:int,stockname: str):
        print("i'm running")
        headlines_response = get_agm_egm(type,stockname)

        return headlines_response

    async def _arun(self, type:int,stockname: str):
        res=await get_agm_egm_as(type,stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = meetCheckInput

#upcoming meetings

def get_next_meetings(type):
    if type==1:
        set='bookcloser'
    elif type==2:
        set='bonus'
    elif type==3:
        set='dividend'
    elif type==4:
        set='split'
    elif type==5:
        set='buyback'
    elif type==6:
        set='result'
    
    #stock_code=int(find_stock_code(stock_name))
    url = f"http://airrchipapis.cmots.com/api/corp-action-WKMonth-details/mon/{set}/50"
    
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse the response JSON
        data = response.json()

        # Fields to remove
        fields_to_remove = ["co_code","sc_code","symbol", "Pricediff", "PerChange", "note","CurrentPrice","isin","TradeDate","divper"]

        # Filter and process the data
        formatted_data_list = []
        if 'data' in data and isinstance(data['data'], list):
            for item in data['data']:
                if isinstance(item, dict):
                    # Remove unwanted fields
                    filtered_item = {
                        key: value for key, value in item.items() if key not in fields_to_remove
                    }
                    # # Format the data for specific fields
                    formatted_data = {}
                    #     "company_name": filtered_item.get('co_name', ''),
                    #     "Date": filtered_item.get('BookCloserDate', '').split('T')[0] if 'BookCloserDate' in filtered_item else '',
                    #     "Agenda": filtered_item.get('agenda', '')
                    # # }
                    formatted_data_list.append(filtered_item)

        # Print the formatted data
        return formatted_data_list
    else:
        return f"Failed to fetch data: {response.status_code} - {response.text}"
    
async def get_next_meetings_as(meeting_type):
    # Map meeting type to endpoint parameter
    meeting_map = {
        1: "bookcloser",
        2: "bonus",
        3: "dividend",
        4: "split",
        5: "buyback",
        6: "result"
    }
        # Validate and determine the type
    if meeting_type == 7:
        # If type is 7, hit all 6 types concurrently
        return await get_all_meetings(meeting_map)
    
    # Validate and determine the type
    set_value = meeting_map.get(meeting_type)
    if not set_value:
        return "Invalid meeting type. Valid options are 1 (bookcloser), 2 (bonus), 3 (dividend), 4 (split), 5 (buyback), or 6 (result)."
    
    url = f"http://airrchipapis.cmots.com/api/corp-action-WKMonth-details/mon/{set_value}/50"
    
    # Set up headers with authorization
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
    
    # Check if the request was successful
    if response.status_code != 200:
        return f"Failed to fetch data: {response.status_code} - {response.text}"
    
    # Parse the response JSON
    data = response.json()
    
    # Fields to remove from the response data
    fields_to_remove = [
        "co_code", "sc_code", "symbol", "Pricediff", "PerChange",
        "note", "CurrentPrice", "isin", "TradeDate", "divper"
    ]
    
    # Filter and process the data
    formatted_data_list = []
    if 'data' in data and isinstance(data['data'], list):
        for item in data['data']:
            if isinstance(item, dict):
                # Remove unwanted fields
                filtered_item = {
                    key: value for key, value in item.items() if key not in fields_to_remove
                }
                formatted_data_list.append(filtered_item)
    
    return formatted_data_list

async def get_all_meetings(meeting_map):
    # Create the URLs for all 6 types
    urls = [
        f"http://airrchipapis.cmots.com/api/corp-action-WKMonth-details/mon/{meeting_map[1]}/50",
        f"http://airrchipapis.cmots.com/api/corp-action-WKMonth-details/mon/{meeting_map[2]}/50",
        f"http://airrchipapis.cmots.com/api/corp-action-WKMonth-details/mon/{meeting_map[3]}/50",
        f"http://airrchipapis.cmots.com/api/corp-action-WKMonth-details/mon/{meeting_map[4]}/50",
        f"http://airrchipapis.cmots.com/api/corp-action-WKMonth-details/mon/{meeting_map[5]}/50",
        f"http://airrchipapis.cmots.com/api/corp-action-WKMonth-details/mon/{meeting_map[6]}/50"
    ]
    
    # Set up headers with authorization
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        # Send all 6 requests concurrently
        responses = await asyncio.gather(*[client.get(url, headers=headers) for url in urls])

    # Process each response and filter data
    all_formatted_data = {}
    fields_to_remove = [
        "co_code", "sc_code", "symbol", "Pricediff", "PerChange",
        "note", "CurrentPrice", "isin", "TradeDate", "divper"
    ]
    
    for i, response in enumerate(responses):
        if response.status_code == 200:
            data = response.json()
            formatted_data_list = []
            if 'data' in data and isinstance(data['data'], list):
                for item in data['data']:
                    if isinstance(item, dict):
                        # Remove unwanted fields
                        filtered_item = {
                            key: value for key, value in item.items() if key not in fields_to_remove
                        }
                        formatted_data_list.append(filtered_item)
            # Map the results to the meeting type (1, 2, 3, etc.)
            all_formatted_data[meeting_map[i+1]] = formatted_data_list
        else:
            all_formatted_data[meeting_map[i+1]] = f"Failed to fetch data: {response.status_code} - {response.text}"
    
    return all_formatted_data


class upCheckInput(BaseModel):
    type: int = Field(..., description="if user is asking about book closure set type as 1,for Bonus set type as 2,for Dividends set type as 3,for Splits set type as 4,for BuyBack set type as 5,for results set type as 6,for upcoming corporate events set type as 7 ")
    #stockname: str = Field(..., description="name of the stock present in the query")
    #prompt: str = Field(..., description="elaborate the provided prompt for similarity search")


class upcoming_corporate_Actions(BaseTool):
    name = "upcoming_corporate_Actions"
    description = "Useful for when you need to get information about upcoming book closures,bonus,rights issues,stock splits,results and buy backs . "

    def _run(self, type:int):
        print("i'm running")
        response = get_next_meetings(type)

        return response

    async def _arun(self, type: int):
        res =await get_next_meetings_as(type)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = upCheckInput


#corp tools

def get_rights(url):
    # Set up the headers with the authorization token
    
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        try:
            # Extract relevant data from the response
            company_data = response.json()

            if company_data['data']:
                company_data = company_data['data'][0]

                # Format the data
                formatted_data = {
                    "company_name": company_data['co_name'],
                    "announcement_date": company_data['AnnouncementDate'].split('T')[0],
                    "record_date": company_data['recorddate'].split('T')[0],
                    "right_date": company_data['RightDate'].split('T')[0],
                    "rights_ratio": company_data['RightsRatio'],
                    "premium": company_data['premium'],
                }
                return formatted_data

        except (KeyError, IndexError):
            print("Error while processing the response data.")
        # Return the result as a dictionary

    else:
        # If the request was not successful, return the result with key and None
        return None

async def get_rights_a(url):
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers)

            if response.status_code != 200:
                return {"error": f"Failed to fetch data: {response.status_code} - {response.text}"}

            company_data = response.json()

            if 'data' not in company_data or not isinstance(company_data['data'], list) or not company_data['data']:
                return {"error": "No data available in response"}

            company_data = company_data['data'][0]
            
            formatted_data = {
                "company_name": company_data.get('co_name', ''),
                "announcement_date": company_data.get('AnnouncementDate', '').split('T')[0] if 'AnnouncementDate' in company_data else None,
                "record_date": company_data.get('recorddate', '').split('T')[0] if 'recorddate' in company_data else None,
                "right_date": company_data.get('RightDate', '').split('T')[0] if 'RightDate' in company_data else None,
                "rights_ratio": company_data.get('RightsRatio', ''),
                "premium": company_data.get('premium', ''),
            }
            return formatted_data

        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}
        except (KeyError, IndexError, ValueError) as e:
            return {"error": f"Data processing failed: {str(e)}"}

def get_merge(url):
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract relevant data from the response
        original_dict = response.json()

        if original_dict['data']:
            original_dict = original_dict['data'][0]

            # Format the data
            updated_dict = {
                'Company Name': original_dict['co_name'],
                'Company Code': original_dict['co_code'],
                'Announcement Date': original_dict['AnnouncementDate'],
                'Merger/Demerger Date': original_dict['Merger_Demerger_Date'],
                'Record Date': original_dict['recorddate'],
                'Merger Ratio': original_dict['mgrRatio'],
                'Merged Into Code': original_dict['MergedInto_Code'],
                'Merged Into ISIN': original_dict['MergedInto_ISIN'],
                'Merged Into Name': original_dict['MergedIntoName'],
                'Type': original_dict['Type'],
                'Institution Name': original_dict['INSTNAME']
            }

            # Return the result as a dictionary
            return updated_dict
    else:
        # If the request was not successful, return the result with key and None
        return None


async def get_merge_a(url):
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}
    
    async with httpx.AsyncClient() as client:
        try:
            # Make a GET request with headers asynchronously
            response = await client.get(url, headers=headers)

            # Check if the request was successful
            if response.status_code != 200:
                return {"error": f"Failed to fetch data: {response.status_code} - {response.text}"}

            # Parse the response JSON
            original_dict = response.json()

            # Validate the response structure
            if 'data' not in original_dict or not isinstance(original_dict['data'], list) or not original_dict['data']:
                return {"error": "No data available in response"}

            # Extract the first data item
            original_dict = original_dict['data'][0]

            # Format the response
            updated_dict = {
                'Company Name': original_dict.get('co_name', ''),
                'Company Code': original_dict.get('co_code', ''),
                'Announcement Date': original_dict.get('AnnouncementDate', '').split('T')[0] if 'AnnouncementDate' in original_dict else None,
                'Merger/Demerger Date': original_dict.get('Merger_Demerger_Date', '').split('T')[0] if 'Merger_Demerger_Date' in original_dict else None,
                'Record Date': original_dict.get('recorddate', '').split('T')[0] if 'recorddate' in original_dict else None,
                'Merger Ratio': original_dict.get('mgrRatio', ''),
                'Merged Into Code': original_dict.get('MergedInto_Code', ''),
                'Merged Into ISIN': original_dict.get('MergedInto_ISIN', ''),
                'Merged Into Name': original_dict.get('MergedIntoName', ''),
                'Type': original_dict.get('Type', ''),
                'Institution Name': original_dict.get('INSTNAME', '')
            }

            # Return the formatted result
            return updated_dict

        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}

        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}


def get_bonus(url):
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract relevant data from the response
        company_data = response.json()
        if company_data['data']:
            company_data = company_data['data'][0]
            formatted_data = {
                "company_name": company_data['co_name'],
                "announcement_date": company_data['AnnouncementDate'].split('T')[0],
                "record_date": company_data['RecordDate'].split('T')[0],
                "bonus_date": company_data['BonusDate'].split('T')[0],
                "bonus_ratio": company_data['BonusRatio'],
            }

            # Return the result as a dictionary
            return formatted_data
    else:
        # If the request was not successful, return the result with key and None
        return None

async def get_bonus_a(url):
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        try:
            # Make a GET request with headers asynchronously
            response = await client.get(url, headers=headers)

            # Check if the request was successful
            if response.status_code != 200:
                return {"error": f"Failed to fetch data: {response.status_code} - {response.text}"}

            # Parse the response JSON
            company_data = response.json()

            # Validate the response structure
            if 'data' not in company_data or not isinstance(company_data['data'], list) or not company_data['data']:
                return {"error": "No data available in response"}

            # Extract the first data item
            company_data = company_data['data'][0]

            # Format the response
            formatted_data = {
                "company_name": company_data.get('co_name', ''),
                "announcement_date": company_data.get('AnnouncementDate', '').split('T')[0] if 'AnnouncementDate' in company_data else None,
                "record_date": company_data.get('RecordDate', '').split('T')[0] if 'RecordDate' in company_data else None,
                "bonus_date": company_data.get('BonusDate', '').split('T')[0] if 'BonusDate' in company_data else None,
                "bonus_ratio": company_data.get('BonusRatio', '')
            }

            # Return the formatted result
            return formatted_data

        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}

        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}
        

def get_splits(url):
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract relevant data from the response
        company_data = response.json()
        if company_data['data']:
            company_data = company_data['data'][0]
            # Format the data
            formatted_data = {
                "company_name": company_data['co_name'],
                "announcement_date": company_data['AnnouncementDate'].split('T')[0],
                "record_date": company_data['recorddate'].split('T')[0],
                "split_date": company_data['SplitDate'].split('T')[0],
                "split_ratio": company_data['SplitRatio'],
                "face value before split": company_data['FVBefore'],
                "face value after split": company_data['FVAfter'],

            }

            # Return the result as a dictionary
            return formatted_data
    else:
        # If the request was not successful, return the result with key and None
        return None


async def get_splits_a(url):
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        try:
            # Make a GET request with headers asynchronously
            response = await client.get(url, headers=headers)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Extract relevant data from the response
                company_data = response.json()

                if company_data['data']:
                    company_data = company_data['data'][0]

                    # Format the data
                    formatted_data = {
                        "company_name": company_data.get('co_name', ''),
                        "announcement_date": company_data.get('AnnouncementDate', '').split('T')[0] if 'AnnouncementDate' in company_data else None,
                        "record_date": company_data.get('recorddate', '').split('T')[0] if 'recorddate' in company_data else None,
                        "split_date": company_data.get('SplitDate', '').split('T')[0] if 'SplitDate' in company_data else None,
                        "split_ratio": company_data.get('SplitRatio', ''),
                        "face value before split": company_data.get('FVBefore', ''),
                        "face value after split": company_data.get('FVAfter', ''),
                    }

                    # Return the formatted result
                    return formatted_data
                else:
                    return {"error": "No data available in the response"}
            else:
                return {"error": f"Failed to fetch data: {response.status_code} - {response.text}"}

        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}

        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}


def get_bb(url):
    # Set up the headers with the authorization token
    
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract relevant data from the response
        company_data = response.json()
        if company_data['data']:
            company_data = company_data['data'][0]
            # Format the data
            formatted_data = {
                "company_name": company_data['co_name'],
                "announcement_date": company_data['AnnouncementDate'].split('T')[0],
                "Buy back from date": company_data['BBFromdate'].split('T')[0],
                "Buy back to date": company_data['BBToDate'].split('T')[0],
                "Buy back date": company_data['BuyBackDate'].split('T')[0],
                "Max price": company_data['maxbuybackprice'],
                # "Total aggrement": company_data['totalaggregateamoun'],

            }

            # Return the result as a dictionary
            return formatted_data
    else:
        # If the request was not successful, return the result with key and None
        return None


async def get_bb_a(url):
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        try:
            # Make a GET request with headers asynchronously
            response = await client.get(url, headers=headers)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Extract relevant data from the response
                company_data = response.json()

                if company_data.get('data'):
                    company_data = company_data['data'][0]

                    # Format the data
                    formatted_data = {
                        "company_name": company_data.get('co_name', ''),
                        "announcement_date": company_data.get('AnnouncementDate', '').split('T')[0] if 'AnnouncementDate' in company_data else None,
                        "Buy back from date": company_data.get('BBFromdate', '').split('T')[0] if 'BBFromdate' in company_data else None,
                        "Buy back to date": company_data.get('BBToDate', '').split('T')[0] if 'BBToDate' in company_data else None,
                        "Buy back date": company_data.get('BuyBackDate', '').split('T')[0] if 'BuyBackDate' in company_data else None,
                        "Max price": company_data.get('maxbuybackprice', ''),
                    }

                    # Return the formatted result
                    return formatted_data
                else:
                    return {"error": "No data available in the response"}
            else:
                return {"error": f"Failed to fetch data: {response.status_code} - {response.text}"}

        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}

        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}

def get_book_closure(url):
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract relevant data from the response
        company_data = response.json()
        if company_data['data']:
            company_data = company_data['data'][0]
            # Format the data
            formatted_data = {
                "company_name": company_data['Co_name'],
                "announcement_date": company_data['AnnouncementDate'].split('T')[0],
                "Book closure from date": company_data['BCFromdate'].split('T')[0],
                "Book closure to date": company_data['BCToDate'].split('T')[0],
                "Agenda": company_data['Agenda'],
                # "Total aggrement": company_data['totalaggregateamoun'],

            }

            # Return the result as a dictionary
            return formatted_data
    else:
        # If the request was not successful, return the result with key and None
        return None

async def get_book_closure_a(url):
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        try:
            # Make a GET request with headers asynchronously
            response = await client.get(url, headers=headers)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Extract relevant data from the response
                company_data = response.json()

                if company_data.get('data'):
                    company_data = company_data['data'][0]

                    # Format the data
                    formatted_data = {
                        "company_name": company_data.get('Co_name', ''),
                        "announcement_date": company_data.get('AnnouncementDate', '').split('T')[0] if 'AnnouncementDate' in company_data else None,
                        "Book closure from date": company_data.get('BCFromdate', '').split('T')[0] if 'BCFromdate' in company_data else None,
                        "Book closure to date": company_data.get('BCToDate', '').split('T')[0] if 'BCToDate' in company_data else None,
                        "Agenda": company_data.get('Agenda', ''),
                    }

                    # Return the formatted result
                    return formatted_data
                else:
                    return {"error": "No data available in the response"}
            else:
                return {"error": f"Failed to fetch data: {response.status_code} - {response.text}"}

        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}

        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}


def get_board_meetings(url):
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Extract relevant data from the response
        company_data = response.json()
        if company_data['data']:
            company_data = company_data['data'][0]
            # Format the data
            formatted_data = {
                "company_name": company_data['Co_name'],
                "Board meeting date": company_data['BMdate'].split('T')[0],
                #"Book closure from date": company_data['BCFromdate'].split('T')[0],
                #"Book closure to date": company_data['BCToDate'].split('T')[0],
                "Agenda": company_data['Description'],
                # "Total aggrement": company_data['totalaggregateamoun'],

            }

            # Return the result as a dictionary
            return formatted_data
    else:
        # If the request was not successful, return the result with key and None
        return None

async def get_board_meetings_a(url):
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        try:
            # Make a GET request with headers asynchronously
            response = await client.get(url, headers=headers)

            # Check if the request was successful (status code 200)
            if response.status_code == 200:
                # Extract relevant data from the response
                company_data = response.json()

                if company_data.get('data'):
                    company_data = company_data['data'][0]

                    # Format the data
                    formatted_data = {
                        "company_name": company_data.get('Co_name', ''),
                        "Board meeting date": company_data.get('BMdate', '').split('T')[0] if 'BMdate' in company_data else None,
                        "Agenda": company_data.get('Description', ''),
                    }

                    # Return the formatted result
                    return formatted_data
                else:
                    return {"error": "No data available in the response"}
            else:
                return {"error": f"Failed to fetch data: {response.status_code} - {response.text}"}

        except httpx.RequestError as e:
            return {"error": f"Request failed: {str(e)}"}

        except Exception as e:
            return {"error": f"An error occurred: {str(e)}"}



def get_corporate_actions(type,stock_name):
    stock_code=int(find_stock_code(stock_name))
    print(stock_code)
    bonus_url = f"http://airrchipapis.cmots.com/api/BonusIssues/{stock_code}"
    right_url = f'http://airrchipapis.cmots.com/api/RightIssues/{stock_code}'
    split_url = f'http://airrchipapis.cmots.com/api/Split-of-FaceValue/{stock_code}'
    mer_url = f'http://airrchipapis.cmots.com/api/MergerDemerger/{stock_code}'
    buyback_url = f'http://airrchipapis.cmots.com/api/BuyBack/{stock_code}'
    book_url = f"http://airrchipapis.cmots.com/api/BookCloser/{stock_code}",
    board_meeting_url=f"http://airrchipapis.cmots.com/api/BookCloser/{stock_code}"

    if type==1:
        book_closure = get_book_closure(book_url)
        return book_closure
    elif type ==2:
        board_meeting=get_board_meetings(board_meeting_url)
        return board_meeting
    elif type ==3:
        bonus = get_bonus(bonus_url)
        return bonus
    elif type ==4:
        rights = get_rights(right_url)
        return rights
    elif type ==5:
        splits = get_splits(split_url)
        return splits
    elif type ==6:
        merger = get_merge(mer_url)
        return merger
    elif type ==7:
        buyback = get_bb(buyback_url)
        return buyback

async def get_corporate_actions_as(type, stock_name):
    stock_code = int(find_stock_code(stock_name))
    print(stock_code)
    
    bonus_url = f"http://airrchipapis.cmots.com/api/BonusIssues/{stock_code}"
    right_url = f'http://airrchipapis.cmots.com/api/RightIssues/{stock_code}'
    split_url = f'http://airrchipapis.cmots.com/api/Split-of-FaceValue/{stock_code}'
    mer_url = f'http://airrchipapis.cmots.com/api/MergerDemerger/{stock_code}'
    buyback_url = f'http://airrchipapis.cmots.com/api/BuyBack/{stock_code}'
    book_url = f"http://airrchipapis.cmots.com/api/BookCloser/{stock_code}"
    board_meeting_url = f"http://airrchipapis.cmots.com/api/BookCloser/{stock_code}"

    async with httpx.AsyncClient() as client:
        if type == 1:
            # Call the book closure API
            return await get_book_closure_a(book_url)
        elif type == 2:
            # Call the board meeting API
            return await get_board_meetings_a(board_meeting_url)
        elif type == 3:
            # Call the bonus API
            return await get_bonus_a(bonus_url)
        elif type == 4:
            # Call the rights API
            return await get_rights_a(right_url)
        elif type == 5:
            # Call the splits API
            return await get_splits_a(split_url)
        elif type == 6:
            # Call the merger API
            return await get_merge_a(mer_url)
        elif type == 7:
            # Call the buyback API
            return await get_bb_a(buyback_url)
        elif type == 8:
            # Call all APIs concurrently
            responses = await asyncio.gather(
                get_book_closure_a(book_url),
                get_board_meetings_a(board_meeting_url),
                get_bonus_a(bonus_url),
                get_rights_a(right_url),
                get_splits_a(split_url),
                get_merge_a(mer_url),
                get_bb_a(buyback_url)
            )
            
            # Return the results as a dictionary mapping API names to their responses
            return {
                "book_closure": responses[0],
                "board_meeting": responses[1],
                "bonus": responses[2],
                "rights": responses[3],
                "splits": responses[4],
                "merger": responses[5],
                "buyback": responses[6]
            }

class corpCheckInput(BaseModel):
    type: int = Field(..., description="if user is asking about book closure set type as 1, Board Meetings  set type as 2,for Bonus set type as 3,for Rights set type as 4,for Splits set type as 5,for Merger/Demerger set type as 6,for BuyBack set type as 7,for all corporate acrions of a company set type as 8")
    stockname: str = Field(..., description="name of the stock present in the query")
    #prompt: str = Field(..., description="elaborate the provided prompt for similarity search")


class corporate_Actions(BaseTool):
    name = "corporate_Actions"
    description = "Useful for when you need to get information about book closures, board meetings, bonus,rights issues,stock splits,merger and demerger and buy backs of a particular. "

    def _run(self, type:int,stockname: str):
        print("i'm running")
        headlines_response = get_corporate_actions(type,stockname)

        return headlines_response

    async def _arun(self,type:int, stockname: str):
        res= await get_corporate_actions_as(type,stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = corpCheckInput


tools=[nse_bse_annocements(),today_results(),delisted_companies(),agm_egm(),corporate_Actions(),upcoming_corporate_Actions()]#screener(),

# tool=[nse_bse_annocements()]

system_message_ind = HumanMessage(
    content="""Assume the role of an Expert Stock Market Assistant/Analyst for the indian stock market.
You should answer user queries in an analytical yet beginner friendly way.so that user can gain actionable insights from your analysis.
YOU ANSWERS SHOULD BE ACCURATE USING THE PROVIDED CONTEXT FROM the tools.
If the query cannot be satisfactorily answered using the available tools.
DONT TRY TO DO CALCULATIONS.
NEVER PROVIDE FORMULAS FOR ANYTHING.
When a user asks a question, follow these steps:
1. Identify the relevant financial data needed to answer the query.
2. Analyze the retrieved data and any generated charts to extract key insights and trends.
3. Formulate a concise response that directly addresses the user's question, focusing on the most important findings from your analysis.

- Avoid just simply regurgitating the raw data from the tools. Instead, provide a thoughtful interpretation and summary as well.

KEEP YOUR ANSWERS AS CONCISE AS POSSIBLE.USER PREFERS CONCISE ANSWERS.
Note that The Indian financial year, also known as the fiscal year (fy), 
runs from April 1 to March 31, divided into four quarters: Q1 (April-June), Q2 (July-September), Q3 (October-December), and Q4 (January-March).
*USE ONLY CRORES instead of millions and billions*
*NOTE- DONT MAKE ANY STOCK BUY ADVICE/RECOMMENDATION IF USER ASKING ABOUT BUY ADVICE/RECOMMENDATION FORMULATE NICE ANSWER*
"""
)


agent_kwargs_ind = {
    "system_message": system_message_ind,
    "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")]
}


#memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, k=2)


# agent2 = initialize_agent(
#     vectordb_tools,
#     GPT4o_mini,
#     # agent=AgentType.OPENAI_MULTI_FUNCTIONS,
#     agent_kwargs=agent_kwargs_ind,
#     memory=memory,
#     agent=AgentType.OPENAI_FUNCTIONS,
#     verbose=True,

# )

def corp_agent_with_session(session_id: str, prompt: str,market:str):

    # if market=="US":
    #     tools=tools_US
    # elif market=="IND":
    #     tools=tool_new
    # langchain.debug = True
    # Initialize the PostgresChatMessageHistory
    history = PostgresChatMessageHistory(
        connection_string = psql_url,
        session_id=session_id,
    )
    
    # Initialize ConversationBufferWindowMemory
    memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, k=6, chat_memory=history)

    agent2 = initialize_agent(
    tools,
    GPT4o_mini,
    # agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    agent_kwargs=agent_kwargs_ind,
    memory=memory,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,

    )

    response=agent2.run(prompt)
    # history.add_user_message(prompt)
    # history.add_ai_message(result)
    return response


#streaming logic
AI_KEY=os.getenv('AI_KEY')
async def authenticate_ai_key(x_api_key: str = Header(...)):
    if x_api_key != AI_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )

async def agent_stream(plan_id,user_id,session_id,prompt,db):
    # similar_question = query_similar_questions(prompt,'fundamental_cache')
    # print(f"Similar Question: {similar_question}")
    # cached_response = db.query(ChatbotResponse).filter(ChatbotResponse.question == similar_question).first()
    # print(f"Cached Response: {cached_response}")

    # if similar_question and cached_response:
    #     # looks up for the corresponding response for
    #     cached_response.count += 1
    #     db.commit()
    #     add_to_memory(prompt, cached_response.response, session_id=session_id)
    #     res=cached_response.response
    #     # chunks = res.split('. ') 
    #     for chunk in res:
    #         yield chunk.encode("utf-8")  # Yield each chunk as bytes
    #         await asyncio.sleep(0.01)  # Adjust the delay as needed for a smoother stream
                
    #     return
    
    history = PostgresChatMessageHistory(
        connection_string = psql_url,
        session_id=session_id,
    )

    # Initialize ConversationBufferWindowMemory
    memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, k=6, chat_memory=history)

    agent2 = initialize_agent(
    tools,
    llm_stream,
    # agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    agent_kwargs=agent_kwargs_ind,
    memory=memory,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,

    )
    with get_openai_callback() as cb:
        async for event in agent2.astream_events(
            {"input": prompt},
            version="v1",
            ):
            kind = event["event"]
            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                #print(content)
                if content:
                    await asyncio.sleep(0.01) 
                        #print(answer)
                    yield content.encode("utf-8")
                else:
                    pass
            if kind == "on_chat_model_end":
                # print(aggregate)
                #print(f'{event["data"]["output"]["generations"][0]}\n')
                final =event["data"]["output"]["generations"]


        total_tokens=cb.total_tokens/1000
        answer=final[0][0]['text']
        #print(total_tokens)

        #store_into_userplan(user_id,total_tokens)
        insert_credit_usage(user_id,plan_id,total_tokens)
        # if should_store(prompt):
        #     new_response = ChatbotResponse(
        #         question=prompt,
        #         response=answer,
        #         count=1,
        #         Total_Tokens=cb.total_tokens,  # Populate the new columns
        #         Prompt_Tokens=cb.prompt_tokens,
        #         Completion_Tokens=cb.completion_tokens,
        #     )
        #     db.add(new_response)
        #     db.commit()
        #     add_questions(prompt,'fundamental_cache')


class InRequest(BaseModel):
    query: str

Base = declarative_base()
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


app = FastAPI()
corp_rag = APIRouter()
@corp_rag.post("/corporate")
async def final_stream(request: InRequest, 
        session_id: str = Query(...), 
        user_id: int = Query(...), 
        plan_id: int = Query(...),
        ai_key_auth: str = Depends(authenticate_ai_key),
        db: Session = Depends(get_db)
    ):
    prompt=request.query
    return StreamingResponse(agent_stream(plan_id,user_id,session_id,prompt,db), media_type="text/event-stream")
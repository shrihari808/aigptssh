from groq import Groq
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
from config import GPT3_16k,chroma_server_client, default_ef,GPT4o_mini
import requests
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories.postgres import PostgresChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
import os
import concurrent.futures
import aiohttp
import asyncio
from fastapi.responses import StreamingResponse
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from fastapi import FastAPI, APIRouter
import httpx
from langchain.callbacks import get_openai_callback
from starlette.status import HTTP_403_FORBIDDEN
from fastapi.responses import JSONResponse
from fastapi import FastAPI, HTTPException,Depends, Header,Query
from streaming.streaming import insert_credit_usage
from config import llm_stream,vs_promoter
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from api.caching import add_questions,verify_similarity,query_similar_questions,should_store




ip_address=os.getenv("PG_IP_ADDRESS")

token=os.getenv('CMOTS_BEARER_TOKEN')
psql_url=os.getenv('DATABASE_URL')

client=chroma_server_client

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
    
    threshold=90
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

def find_stock_code(stock_name):
    df = pd.read_csv("csvdata/6000stocks.csv")

    # Extract the company names, codes, and symbols from the DataFrame
    company_names = df["Company Name"].tolist()
    company_codes = df["Company Code"].tolist()
    company_symbols = df["co_symbol"].tolist()
    
    threshold = 90
    symbol_threshold = 95
    
    # Use fuzzy matching to find the closest matches to the input stock name
    match = process.extractOne(stock_name, company_names)
    match1 = process.extractOne(stock_name, company_symbols)

    # # Print matches for debugging
    # print("Match by name:", match)
    # print("Match by symbol:", match1)
    
    # Determine which match is stronger and meets its respective threshold
    if match and match1:
        # Compare the scores to find the best match
        if match[1] >= threshold and (match1[1] < symbol_threshold or match[1] > match1[1]):
            idx = company_names.index(match[0])
            print("Best match by name:", company_codes[idx])
            return company_codes[idx]
        elif match1[1] >= symbol_threshold:
            idx = company_symbols.index(match1[0])
            print("Best match by symbol:", company_codes[idx])
            return company_codes[idx]
    elif match and match[1] >= threshold:
        idx = company_names.index(match[0])
        print("Best match by name:", company_codes[idx])
        return company_codes[idx]
    elif match1 and match1[1] >= symbol_threshold:
        idx = company_symbols.index(match1[0])
        print("Best match by symbol:", company_codes[idx])
        return company_codes[idx]

    return None




def get_collection_name(stock_name):
    # Load the CSV file into a DataFrame
    #df = pd.read_csv("company_codes2.csv")
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
        return f"stock_{company_codes[idx]}"
    
    # Check if match in company symbols meets the threshold
    elif match1 and match1[1] >= threshold:
        idx = company_symbols.index(match1[0])
        return f"stock_{company_codes[idx]}"

    else:
        return None


def rank_documents2(original_list, query):
    client = Groq()
    titles = [item.split(":")[0] for item in original_list]
    completion = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": """FOLLOW INSTRCUTIONS.I will provide you with a list of titles of docs and user query.
                Rank the titles based on their relevance to query and return the just TOP 4 IN THE SAME PYTHON LIST IN THE ORDER ITS BEING ASKED IN THE QUERY WITH THE MOST RELEVANT FIRST AND SO ON.
                \nFOLLOW INSTRCUTIO\n  OUTPUT SHOULD BE JUST THE PYTHON LIST WITH THE RE RANKEND TITLES,PLEASE dont return NOTHING ELSE"""
            },
            {
                "role": "user",
                "content": " \n    query:what was profit in q3 2021 and q3 2023?\n    docs:['Quarterly Results Dec 2020 (Q3 of FY 2020-2021) ',\n 'Quarterly Results Dec 2021 (Q3 of FY 2021-2022) ',\n 'Quarterly Results Dec 2023 (Q3 of FY 2023-2024) ',\n 'Quarterly Results Sep 2021 (Q2 of FY 2021-2022) ',\n 'Quarterly Results Mar 2023 (Q4 of FY 2022-2023) ',\n 'Quarterly Results Jun 2023 (Q1 of FY 2023-2024) ',\n 'Quarterly Results Sep 2023 (Q2 of FY 2023-2024) ',\n 'Quarterly Results Jun 2021 (Q1 of FY 2021-2022) ',\n 'Quarterly Results Sep 2022 (Q2 of FY 2022-2023) ',\n 'Quarterly Results Dec 2022 (Q3 of FY 2022-2023) ']"
            },
            {
                "role": "assistant",
                "content": "['Quarterly Results Dec 2021 (Q3 of FY 2021-2022)', 'Quarterly Results Dec 2023 (Q3 of FY 2023-2024)', 'Quarterly Results Dec 2022 (Q3 of FY 2022-2023)', 'Quarterly Results Sep 2021 (Q2 of FY 2021-2022)']"
            },
            {
                "role": "user",
                "content": f"\n query:{query} \n titles:{titles}"
            }
        ],
        temperature=0.1,
        max_tokens=1024,
        top_p=1,
        stream=False,
        stop=None,
    )
    titles_list=completion.choices[0].message.content
    print("titles",titles_list)
    ordered_list = [item for title in titles_list[:4] for item in original_list if title in item][:4]


    return ordered_list


def reranker(original_list,q):
    titles = [item.split(":")[0] for item in original_list]

    template = """You are an intelligent assistant that can rank passages based on their relevancy to the query.
    I will provide you with a list of titles of docs.Rank the titles based on their relevance to query and return the just TOP 4 IN THE SAME PYTHON LIST.
    OUTPUT SHOULD BE JUST THE PYTHON  LIST WITH THE RERANKEND TITLES,NOTHING ELSE.
    query:{query}
    docs:{docs}"""
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    chain = LLMChain(
    prompt=prompt,
    llm=GPT3_16k,
    output_parser=output_parser
    )
    titles_list = eval(chain.run(query=q, docs=titles))
    print("reranked titles")
    print(titles_list)

    
    ordered_list = [item for title in titles_list for item in original_list if title in item][:4]

    return ordered_list

def fun2(stock_name,q):
    #convert stock name into company code
    code=get_collection_name(stock_name)
    print(code)

    current_date = datetime.date.today()
    #print(current_date.strftime("%d/%m/%Y"))
    template = """You are a machine designed to rewrite and expand queries to make them suitable for getting relevant results using similarity searches on a vector database of financial data stored in JSON format. 
    Today's date {date}, use this context and the current date to make queries more descriptive and specific.
    Note that The Indian financial year, also known as the fiscal year (fy), 
    runs from April 1 to March 31, divided into four quarters: Q1 (April-June), Q2 (July-September), Q3 (October-December), and Q4 (January-March).
    Examples:
    Input: Net profit over the last few years.
    Output: Net profit in 2023, 2022, and 2021.
    Input: compare the cashflow \pnl \ results
    Output: compare the cashflow \pnl \results in (whatever the latest financial year is) in this case 2024.

    query:{query}
    """
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()

    chain = LLMChain(
    prompt=prompt,
    llm=GPT3_16k,
    output_parser=output_parser
    )
    new_query = chain.run(query=q,date=current_date)
    print("new_query",new_query)
    collection=client.get_collection(name=code,embedding_function=default_ef)
    #   collection=persistent_client.get_collection(name=code)
    print("collection called")
    docs=collection.query(
        query_texts=[new_query],
        n_results=15,
    )
    print("docs recevied")
    #output=reranker(docs['documents'][0],new_query)       
    #print("reranked list:",output)

    return docs



def run_queries_in_parallel(stock_names, query):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(fun2, stock, query): stock for stock in stock_names}

        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                result = future.result()
                results[stock] = result
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class TickerCheckInput(BaseModel):
    stockname: str = Field(..., description="name of the stock present in the query")
    #prompt: str = Field(..., description="elaborate the provided prompt for similarity search")

class tool1(BaseTool):
    name = "stockdatatool"
    description = "use this tool to get financial info about any stock which are not covered in daily_ratios tools"

    def _run(self, stockname: list,prompt: str):
        print("i'm running")
        headlines_response = run_queries_in_parallel(stockname,prompt)

        return headlines_response

    def _arun(self, stockname: str,prompt: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = TickerCheckInput



# llm1 = ChatOpenAI(temperature=0.2, model="gpt-4o-2024-05-13") 
# llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-1106")

def remove_description(data):
    """
    Removes the 'Description' field from each dictionary in the input list.

    Parameters:
    data (list of dict): List of dictionaries from which 'Description' field is to be removed.

    Returns:
    list of dict: List of dictionaries with 'Description' field removed.
    """
    for item in data:
        if 'Description' in item or 'remark' in item or 'isin' in item or 'co_code' in item:
            item.pop('Description', None)
            item.pop('remark', None)
            item.pop('isin',None)
            item.pop('co_code',None)
    return data
  

def get_current_date():
    return datetime.datetime.now().date()


date=get_current_date()
#print(date)
corp_prompt = """Given the JSON data {data}, today's date {date}, and the user's question {query}, analyze the JSON data carefully.
If the user's query requires using today's date to fetch records, do so accordingly.
Provide a comprehensive and accurate response based on the user's query and the information available in the JSON data
"""

c_q_prompt=PromptTemplate(template=corp_prompt,input_variables=['data','query','date'])
#llm=ChatOpenAI(temperature = 0.2 ,model ='gpt-4o-mini')
corp_chain=LLMChain(prompt=c_q_prompt,llm=GPT4o_mini)


msg="No data available for this stock. This service is only for Indian market stocks. If this is an Indian stock, please ensure the correct stock name or ticker is used."

def bonus(type,year,query):
    if type==1:
        url="http://airrchipapis.cmots.com/api/BonusIssues/-"
        #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzE2NDQ1NDM0LCJleHAiOjE3MTkzODMwMzQsImlhdCI6MTcxNjQ0NTQzNCwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.FS_3GZ4PzbnXepFT0wYJa0NdvY4mZoCua2Yvyj_lY50"  # Replace with your actual authorization token
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Print the response content
            data=response.json()
            D=data['data']
            filtered_data = [entry for entry in D if entry['BonusDate'].startswith(year)]
            data_clean=remove_description(filtered_data)
            #return data_clean
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            print(response.text)
    elif type ==2:
        url="http://airrchipapis.cmots.com/api/Split-of-FaceValue/-"
        #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzE2NDQ1NDM0LCJleHAiOjE3MTkzODMwMzQsImlhdCI6MTcxNjQ0NTQzNCwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.FS_3GZ4PzbnXepFT0wYJa0NdvY4mZoCua2Yvyj_lY50"  # Replace with your actual authorization token
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Print the response content
            data=response.json()
            D=data['data']
            filtered_data = [entry for entry in D if entry['SplitDate'].startswith(year)]
            data_clean=remove_description(filtered_data)
            #return data_clean
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            print(response.text)
    elif type ==3:
        url="http://airrchipapis.cmots.com/api/BuyBack/-"
        #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzE2NDQ1NDM0LCJleHAiOjE3MTkzODMwMzQsImlhdCI6MTcxNjQ0NTQzNCwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.FS_3GZ4PzbnXepFT0wYJa0NdvY4mZoCua2Yvyj_lY50"  # Replace with your actual authorization token
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Print the response content
            data=response.json()
            D=data['data']
            filtered_data = [entry for entry in D if entry['AnnouncementDate'].startswith(year)]
            data_clean=remove_description(filtered_data)
            #return data_clean
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            print(response.text)

    elif type ==4:
        url="http://airrchipapis.cmots.com/api/RightIssues/-"
        #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzE2NDQ1NDM0LCJleHAiOjE3MTkzODMwMzQsImlhdCI6MTcxNjQ0NTQzNCwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.FS_3GZ4PzbnXepFT0wYJa0NdvY4mZoCua2Yvyj_lY50"  # Replace with your actual authorization token
        headers = {"Authorization": f"Bearer {token}"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            # Print the response content
            data=response.json()
            D=data['data']
            filtered_data = [entry for entry in D if entry['AnnouncementDate'].startswith(year)]
            data_clean=remove_description(filtered_data)
            #return data_clean
        else:
            # Print an error message if the request was not successful
            print(f"Error: {response.status_code}")
            print(response.text)

    date=get_current_date()
    #print(date)
    #answer from llm chain
    res=corp_chain.predict(query=query,data=data_clean,date=date)
    return res



class queryCheckInput(BaseModel):
    query:str =Field(...,description="return complete query user mentioned wihtout removing anything")
    type: int = Field(..., description="is user asking about bonus set type as 1 ,for splits,face values set type as 2,for buybacks set type as 3,for rights issues set type as 4")
    year: str = Field(..., description="year mentioned in the prompt if no year mentiones set year as 2024")
    # year: str = Field(..., description="year mentioned in the query , if no year mentioned keep 2023")
    # prompt: str = Field(..., description="give me complete prompt user asked")
    

    
class screener_bonus(BaseTool):
    name = "stock_screener"
    description = "use this tool when you need info about upcoming bonus issues, stock splits , buybacks ,right issues"

    def _run(self, type: int,year:str,query:str):
        print("i'm running")
        # print(type)
        # print(query)
        headlines_response = bonus(type,year,query)

        return headlines_response

    def _arun(self, stockname: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = queryCheckInput


class stocknameInput(BaseModel):
    #stockname: list = Field(..., description="list of all names of stocks")
    stockname: str = Field(..., description="name of stock present in user query")

def get_daily_ratios(stock_name):
    code=find_stock_code(stock_name)
    #print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    headers = {
        'Authorization': f'Bearer {token}'
    }
    url = f'http://airrchipapis.cmots.com/api/DailyRatios/{code}/C'
    url1 = f'http://airrchipapis.cmots.com/api/YearlyRatio/{code}/C'

    response_daily = requests.get(url, headers=headers)
    result_daily=response_daily.json()
    if (result_daily.get("data") == [] or result_daily.get("data") is None):
        url = f'http://airrchipapis.cmots.com/api/DailyRatios/{code}/S'
        response_daily = requests.get(url, headers=headers)
        result_daily=response_daily.json()
    response_year = requests.get(url1, headers=headers)
    result_yearly=response_year.json()
    if (result_yearly.get("data") == [] or result_yearly.get("data") is None):
        url1 = f'http://airrchipapis.cmots.com/api/YearlyRatio/{code}/S'
        response_year = requests.get(url1, headers=headers)
        result_yearly=response_year.json()

    return {"latest_data":result_daily,"last 4 years data":result_yearly}
    # daily_ratios = {}
    # for ratio in result['data']:
    #     for key, value in ratio.items():
    #         daily_ratios[key] = value
    # return json.dumps({stock_name: daily_ratios})


async def get_daily_ratios_async(stock_name):
    # Simulating find_stock_code function
    code=find_stock_code(stock_name)# Example stock code; replace this with the actual logic
    if code:
        code = int(code)
    if code is None:
        return msg
    
    headers = {
        'Authorization': f'Bearer {token}'
    }
    
    async with aiohttp.ClientSession(headers=headers) as session:
        # Fetch daily ratios
        url_daily_c = f'http://airrchipapis.cmots.com/api/DailyRatios/{code}/C'
        url_daily_s = f'http://airrchipapis.cmots.com/api/DailyRatios/{code}/S'
        url_yearly_c = f'http://airrchipapis.cmots.com/api/YearlyRatio/{code}/C'
        url_yearly_s = f'http://airrchipapis.cmots.com/api/YearlyRatio/{code}/S'

        async def fetch_data(url):
            async with session.get(url) as response:
                return await response.json()
        
        # Fetch daily data
        result_daily = await fetch_data(url_daily_c)
        if result_daily.get("data") == [] or result_daily.get("data") is None:
            result_daily = await fetch_data(url_daily_s)
        
        # Fetch yearly data
        result_yearly = await fetch_data(url_yearly_c)
        if result_yearly.get("data") == [] or result_yearly.get("data") is None:
            result_yearly = await fetch_data(url_yearly_s)
        
        return {
            "latest_data": result_daily,
            "last_4_years_data": result_yearly
        }



def get_daily_ratios_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_daily_ratios, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                #print(data)
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results



class daily_yearly_ratios(BaseTool):
    name = "daily_yearly_ratios"
    description = "Use this tool only when you need information about daily and yearly ratios of a company which also provide info about the following financial metrics* - Market Capitalization (MCAP),Enterprise Value (EV),Price-to-Earnings Ratio (PE Ratio),Price-to-Book Value Ratio (PBV Ratio),Dividend Yield,Earnings Per Share(EPS),Book Value Per Share(BookValue),Return on Assets(ROA),Return on Equity(ROE),Return on Capital Employed(ROCE),Earnings Before Interest and Taxes(EBIT),Earnings Before Interest,Taxes,Depreciation,and Amortization(EBITDA),EV/Sales,EV/EBITDA,Net Income Margin,Gross Income Margin,Asset Turnover Ratio,Current Ratio,Debt-to-Equity Ratio,Free Cash Flow Margin (FCF Margin),Sales to Total Assets Ratio,NetDebt/FCF,Net Debt to EBITDA Ratio,EBITDA Margin,Total Shareholders Equity,Short-term Debt,Long-term Debt,Diluted Earnings Per Share(EPS Diluted),Net Sales,Net Profit,Annual Dividend,Cost of Goods Sold(COGS).IF USER IS NOT ASKING ABOUT THEM, DON'T USE THIS TOOL"

    def _run(self, stockname:str):
        #pass
        #print("i'm running")
        # print(type)
        # print(query)
        headlines_response = get_daily_ratios(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        response =await get_daily_ratios_async(stockname)
        return response
        #raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput


#############################################

def get_profit_loss(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    headers = {
        'Authorization': f'Bearer {token}'
    }
    url = f"http://airrchipapis.cmots.com/api/ProftandLoss/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/ProftandLoss/{code}/S"

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Return the response content as JSON
        # data = response.json()
        # if (data.get("data") == [] or data.get("data") is None):
        columns_to_remove = ['Income from Investment and Financial Services',
                             'Income from Insurance Operations',
                             'Other Operating Revenue',
                             'Less: Excise Duty / GSTInternally Manufactured Intermediates Consumed',
                             'Purchases of Stock-in-Trade',
                             '   Administrative and Selling Expenses',
                             'Profit Before Exceptional Items and Tax',
                             'Profit Before Extraordinary Items and Tax',
                             'Other Adjustments Before Tax',
                             'MAT Credit Entitlement',
                             'Other Tax',
                             'Adjust for Previous Year',
                             'Extraordinary Items After Tax',
                             'Discontinued Operations After Tax',
                             'Profit / (Loss) from Discontinuing Operations',
                             'Tax Expenses of Discontinuing Operations',
                             'Profit Attributable to Shareholders',
                             'Adjustments to Net Income',
                             'Profit Attributable to Equity Shareholders',
                             'Weighted Average Shares - Basic']

        # Remove entries with specified 'COLUMNNAME', considering leading white spaces
        filtered_data = [entry for entry in data['data'] if entry['COLUMNNAME'].strip() not in columns_to_remove]

        # Update the 'data' field with the filtered data
        data['data'] = filtered_data

        # Print the modified data
        # return data
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        print(response.text)
        return

        # Dictionary to store year-wise data
    yearwise_data = {}
    financial_data = data['data']
    # Process each data dictionary
    for data in financial_data:
        asset_type = data['COLUMNNAME'].rstrip(':')  # Remove trailing ':' if present

        for key, value in data.items():
            if key.startswith('Y20'):
                # Extracting the year from the key
                year = key[1:5]  # Extracting from the 2nd character onwards
                # Creating year entry if it doesn't exist
                if year not in yearwise_data:
                    yearwise_data[year] = {}

                # Adding asset type and value for the current year
                yearwise_data[year][asset_type] = f'{value}'

    return yearwise_data

async def get_profit_loss_as(stock_name):
    #code = await asyncio.to_thread(find_stock_code, stock_name)
    code=find_stock_code(stock_name)  # Run `find_stock_code` in a separate thread if it's synchronous
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg

    # API URLs
    url = f"http://airrchipapis.cmots.com/api/ProftandLoss/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/ProftandLoss/{code}/S"

    # Set up headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        # Make the first GET request
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # If data is empty or None, try the secondary URL
            if not data.get("data"):
                response = await client.get(urls, headers=headers)
                data = response.json()
        else:
            # Handle unsuccessful response
            print(f"Error: {response.status_code}")
            print(response.text)
            return

    # Columns to be removed
    columns_to_remove = [
        'Income from Investment and Financial Services',
        'Income from Insurance Operations',
        'Other Operating Revenue',
        'Less: Excise Duty / GSTInternally Manufactured Intermediates Consumed',
        'Purchases of Stock-in-Trade',
        '   Administrative and Selling Expenses',
        'Profit Before Exceptional Items and Tax',
        'Profit Before Extraordinary Items and Tax',
        'Other Adjustments Before Tax',
        'MAT Credit Entitlement',
        'Other Tax',
        'Adjust for Previous Year',
        'Extraordinary Items After Tax',
        'Discontinued Operations After Tax',
        'Profit / (Loss) from Discontinuing Operations',
        'Tax Expenses of Discontinuing Operations',
        'Profit Attributable to Shareholders',
        'Adjustments to Net Income',
        'Profit Attributable to Equity Shareholders',
        'Weighted Average Shares - Basic'
    ]

    # Filter out entries with specified 'COLUMNNAME'
    filtered_data = [entry for entry in data.get('data', []) if entry['COLUMNNAME'].strip() not in columns_to_remove]
    data['data'] = filtered_data

    # Dictionary to store year-wise data
    yearwise_data = {}
    financial_data = data['data']

    # Process each data dictionary
    for entry in financial_data:
        asset_type = entry['COLUMNNAME'].rstrip(':')  # Remove trailing ':' if present

        for key, value in entry.items():
            if key.startswith('Y20'):  # Assuming year keys start with 'Y20XX'
                year = key[1:5]  # Extract the year
                if year not in yearwise_data:
                    yearwise_data[year] = {}
                yearwise_data[year][asset_type] = f'{value}'

    return yearwise_data

def get_pl_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_profit_loss, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results

class profit_loss(BaseTool):
    name = "profit_loss"
    description = "This tool is an indispensable asset for retrieving the profit and loss statements of specific " \
                  "companies. It enables users to access detailed information concerning various financial aspects " \
                  "such as 'Revenue From Operations', 'Other Income', 'Total Revenue', 'Changes in Inventories', " \
                  "'Employee Benefits', 'Total Other Expenses', 'Finance Costs', 'Depreciation and Amortization', " \
                  "'Total Expenses', 'Profit Before Tax', 'Taxation', 'Profit After Tax', 'Earnings Per Share - " \
                  "Basic', 'Earnings Per Share - Diluted', 'Operating Profit before Depreciation', 'Operating Profit " \
                  "after Depreciation', 'Dividend Per Share', 'Dividend Percentage', 'Equity Dividend', and 'Weighted " \
                  "Average Shares - Diluted'. This comprehensive tool facilitates in-depth analysis of financial " \
                  "performance, aiding users in making informed decisions and strategic assessments."

    def _run(self, stockname: str):
        #pass
        #print("i'm running")
        headlines_response = get_profit_loss(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res=await get_profit_loss_as(stockname)
        return res
        #raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

################################################

def get_balance_sheet(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    # Construct the URL with the company code
    url = f"http://airrchipapis.cmots.com/api/BalanceSheet/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/BalanceSheet/{code}/S"
   
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers
        # Make a GET request with headers
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Return the response content as JSON
        #data = response.json()

        columns_to_remove = ['Non-Current Assets:', 'Intangible Assets under Development', 'Capital Work in Progress',
                             'Investment Properties', 'Investments of Life Insurance Business',
                             'Biological Assets other than Bearer Plants (Non Current)', 'Loans - Long-term',
                             'Insurance Related Assets (Non Current)',
                             'Deferred Tax Assets', 'Current Assets:',
                             'Biological Assets other than Bearer Plants (Current)', 'Current Tax Assets - Short-term',
                             'Insurance Related Assets (Current)', 'Assets Classified as Held for Sale',
                             'Current Liabilities:', 'Insurance Related Liabilities (Current)',
                             'Liabilities Directly Associated with Assets Classified as Held for Sale',
                             'Current Tax Liabilities - Short-term', 'Other Short term Provisions',
                             'Non-Current Liabilities:', 'Debt Securities', 'Lease Liabilities (Non Current)',
                             'Other Long term Liabilities', 'Others Financial Liabilities - Long-term',
                             'Insurance Related Liabilities (Non Current)', 'Other Long term Provisions',
                             'Deferred Tax Liabilities', 'Shareholders Funds:', 'Preference Capital',
                             'Unclassified Capital', 'Reserves and Surplus', 'Other Equity Components',
                             'Total Shareholder\'s Fund',
                             'Contingent Liabilities and Commitments (to the Extent Not Provided for)',
                             'Ordinary Shares :', 'Authorised:', 'Number of Equity Shares - Authorised',
                             'Amount of Equity Shares - Authorised', 'Par Value of Authorised Shares',
                             'Susbcribed & fully Paid up :', 'Par Value', 'Susbcribed & fully Paid up Shares',
                             'Susbcribed & fully Paid up CapItal', 'Right-of-Use Assets']

        # Remove entries with specified 'COLUMNNAME', considering leading white spaces
        filtered_data = [entry for entry in data['data'] if entry['COLUMNNAME'].strip() not in columns_to_remove]

        # Update the 'data' field with the filtered data
        data['data'] = filtered_data

        # Print the modified data
        # return data
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        print(response.text)
        return

        # Dictionary to store year-wise data
    yearwise_data = {}
    financial_data = data['data']
    # Process each data dictionary
    for data in financial_data:
        asset_type = data['COLUMNNAME'].rstrip(':')  # Remove trailing ':' if present

        for key, value in data.items():
            if key.startswith('Y20'):
                # Extracting the year from the key
                year = key[1:5]  # Extracting from the 2nd character onwards
                # Creating year entry if it doesn't exist
                if year not in yearwise_data:
                    yearwise_data[year] = {}

                # Adding asset type and value for the current year
                yearwise_data[year][asset_type] = f'{value} '

    return yearwise_data


async def get_balance_sheet_as(stock_name):
    #code = await asyncio.to_thread(find_stock_code, stock_name)
    code =find_stock_code(stock_name)  # Run `find_stock_code` in a thread if it's synchronous
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg

    # Construct the URLs with the company code
    url = f"http://airrchipapis.cmots.com/api/BalanceSheet/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/BalanceSheet/{code}/S"

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        # Make the first GET request
        response = await client.get(url, headers=headers)
        if response.status_code == 200:
            data = response.json()
            # If data is empty or None, try the secondary URL
            if not data.get("data"):
                response = await client.get(urls, headers=headers)
                data = response.json()
        else:
            # Handle unsuccessful response
            print(f"Error: {response.status_code}")
            print(response.text)
            return

    # Columns to be removed
    columns_to_remove = [
        'Non-Current Assets:', 'Intangible Assets under Development', 'Capital Work in Progress',
        'Investment Properties', 'Investments of Life Insurance Business',
        'Biological Assets other than Bearer Plants (Non Current)', 'Loans - Long-term',
        'Insurance Related Assets (Non Current)', 'Deferred Tax Assets', 'Current Assets:',
        'Biological Assets other than Bearer Plants (Current)', 'Current Tax Assets - Short-term',
        'Insurance Related Assets (Current)', 'Assets Classified as Held for Sale',
        'Current Liabilities:', 'Insurance Related Liabilities (Current)',
        'Liabilities Directly Associated with Assets Classified as Held for Sale',
        'Current Tax Liabilities - Short-term', 'Other Short term Provisions',
        'Non-Current Liabilities:', 'Debt Securities', 'Lease Liabilities (Non Current)',
        'Other Long term Liabilities', 'Others Financial Liabilities - Long-term',
        'Insurance Related Liabilities (Non Current)', 'Other Long term Provisions',
        'Deferred Tax Liabilities', 'Shareholders Funds:', 'Preference Capital',
        'Unclassified Capital', 'Reserves and Surplus', 'Other Equity Components',
        'Total Shareholder\'s Fund',
        'Contingent Liabilities and Commitments (to the Extent Not Provided for)',
        'Ordinary Shares :', 'Authorised:', 'Number of Equity Shares - Authorised',
        'Amount of Equity Shares - Authorised', 'Par Value of Authorised Shares',
        'Susbcribed & fully Paid up :', 'Par Value', 'Susbcribed & fully Paid up Shares',
        'Susbcribed & fully Paid up CapItal', 'Right-of-Use Assets'
    ]

    # Remove entries with specified 'COLUMNNAME'
    filtered_data = [entry for entry in data.get('data', []) if entry['COLUMNNAME'].strip() not in columns_to_remove]
    data['data'] = filtered_data

    # Dictionary to store year-wise data
    yearwise_data = {}
    financial_data = data['data']

    # Process each data dictionary
    for entry in financial_data:
        asset_type = entry['COLUMNNAME'].rstrip(':')  # Remove trailing ':' if present

        for key, value in entry.items():
            if key.startswith('Y20'):  # Assuming year keys start with 'Y20XX'
                year = key[1:5]  # Extract the year
                if year not in yearwise_data:
                    yearwise_data[year] = {}
                yearwise_data[year][asset_type] = f'{value}'

    return yearwise_data


def get_bs_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_balance_sheet, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results

class balance_sheet(BaseTool):
    name = "balance_sheet"
    description = "This tool proves invaluable when seeking the balance sheet of a particular company. It offers " \
                  "specific insights into key financial elements such as 'Fixed Assets', 'Non-current Investments', " \
                  "'Long-term Loans and Advances', 'Other Non-Current Assets', 'Total Non-Current Assets', " \
                  "'Inventories', 'Current Investments', 'Cash and Cash Equivalents', 'Trade Receivables', " \
                  "'Short-term Loans and Advances', 'Other Current Assets', 'Total Current Assets', 'Total Assets', " \
                  "'Borrowings', 'Trade Payables', 'Other Current Liabilities', 'Total Current Liabilities', " \
                  "'Net Current Assets', 'Long-term Borrowings', 'Total Non-Current Liabilities', 'Share Capital', " \
                  "'Other Equity', 'Total Equity', and 'Total Equity and Liabilities'"

    def _run(self, stockname: str):
        #pass
        # print("i'm running")
        headlines_response = get_balance_sheet(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res= await get_balance_sheet_as(stockname)
        return res
        #raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

###########################################

def get_cashflow(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/CashFlow/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/CashFlow/{code}/S"
     # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        #data = response.json()

        # Specify the column names to remove
        columns_to_remove = [
            'None',  # Assuming the appearance of 'None' is a placeholder and can be considered useless
            'Adjustments : ',
            'Dividend Received 1',
            'Dividend Received 2',
            'Others 1',
            'Others 2',
            'Others 3',
            'Others 4',
            'Others 5',
            'Others 6',
            'Excess Depreciation W/b',
            'Premium on Lease of land',
            'Payment Towards VRS',
            "Prior Year's Taxation",
            'Gain on Forex Exch. Transactions',
            'Others 4',
            'Capital Subsidy Received',
            'Investment in Subsidiaires',
            'Investment in Group Cos.',
            'Issue of Shares on Acquisition of Cos.',
            'Cancellation of Investment in Cos. Acquired',
            'Inter Corporate Deposits',
            'Share Application Money',
            'Shelter Assistance Reserve',
            'Others 5',
            'Cash Flow From Operating Activities'
            # 'Cash and Cash Equivalents at Beginning of the year',
            # 'Net Cash from Operating Activities',
            # 'Net Cash used in Investing Activities',
            # 'Net Cash used in Financing Activities',
            # 'Net Inc./(Dec.) in Cash and Cash Equivalent',
        ]

        # Ensure 'data' contains the list of dictionaries
        if 'data' in data and isinstance(data['data'], list):
            # Remove entries with 'COLUMNNAME' being None
            filtered_data = [
                entry for entry in data['data'] if entry.get('COLUMNNAME') is not None
            ]

            # Filter out entries with specified 'COLUMNNAME', considering leading white spaces
            filtered_data = [
                entry for entry in filtered_data if entry.get('COLUMNNAME', '').strip() not in columns_to_remove
            ]

        # Update the 'data' field with the filtered data
        data['data'] = filtered_data

        # Remove entries with specified 'COLUMNNAME', considering leading white spaces
        # filtered_data = [entry for entry in data['data'] if entry['COLUMNNAME'].strip().lower() not in columns_to_remove]

        # Update the 'data' field with the filtered data
        data['data'] = filtered_data

    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        print(response.text)

    # Dictionary to store year-wise data
    yearwise_data = {}
    financial_data = data['data']
    # Process each data dictionary
    for data in financial_data:
        asset_type = data['COLUMNNAME'].rstrip(':')  # Remove trailing ':' if present

        for key, value in data.items():
            if key.startswith('Y20'):
                # Extracting the year from the key
                year = key[1:5]  # Extracting from the 2nd character onwards

                # Only consider years starting with 'Y20'
                if year in ('2024', '2021', '2022', '2023'):  # Adjust years as needed
                    # Creating year entry if it doesn't exist
                    if year not in yearwise_data:
                        yearwise_data[year] = {}

                    # Adding asset type and value for the current year with "CR" if value is non-zero
                    if value != 0.0:
                        yearwise_data[year][asset_type] = f'{value} CR'

    return yearwise_data

async def get_cashflow_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg

    url = f"http://airrchipapis.cmots.com/api/CashFlow/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/CashFlow/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        # Initial request
        response = await client.get(url, headers=headers)
        data = response.json()

        # Fallback request if initial data is empty or None
        if data.get("data") == [] or data.get("data") is None:
            response = await client.get(urls, headers=headers)
            data = response.json()

    # Check if the request was successful
    if response.status_code == 200:
        columns_to_remove = [
            'None',
            'Adjustments : ',
            'Dividend Received 1',
            'Dividend Received 2',
            'Others 1',
            'Others 2',
            'Others 3',
            'Others 4',
            'Others 5',
            'Others 6',
            'Excess Depreciation W/b',
            'Premium on Lease of land',
            'Payment Towards VRS',
            "Prior Year's Taxation",
            'Gain on Forex Exch. Transactions',
            'Capital Subsidy Received',
            'Investment in Subsidiaires',
            'Investment in Group Cos.',
            'Issue of Shares on Acquisition of Cos.',
            'Cancellation of Investment in Cos. Acquired',
            'Inter Corporate Deposits',
            'Share Application Money',
            'Shelter Assistance Reserve',
            'Cash Flow From Operating Activities',
        ]

        if 'data' in data and isinstance(data['data'], list):
            filtered_data = [
                entry for entry in data['data']
                if entry.get('COLUMNNAME') and entry.get('COLUMNNAME').strip() not in columns_to_remove
            ]
            data['data'] = filtered_data
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return {"message": f"Error {response.status_code}: Unable to fetch cash flow data"}

    yearwise_data = {}
    financial_data = data['data']

    for data in financial_data:
        asset_type = data['COLUMNNAME'].rstrip(':')
        for key, value in data.items():
            if key.startswith('Y20'):
                year = key[1:5]
                if year in ('2024', '2021', '2022', '2023'):  # Adjust years as needed
                    if year not in yearwise_data:
                        yearwise_data[year] = {}
                    if value != 0.0:
                        yearwise_data[year][asset_type] = f'{value} CR'

    return yearwise_data


def get_cf_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_cashflow, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results

class cash_flow(BaseTool):
    name = "cash_flow"
    description = "This tool is invaluable for accessing the cash flow of a specific company. It provides detailed " \
                  "information on essential components such as 'Cash and Cash Equivalents at Beginning of the Year', " \
                  "'Net Profit before Tax & Extraordinary Items', 'Depreciation', 'Interest (Net)', 'Profit/Loss on " \
                  "Sales of Assets', 'Profit/Loss on Sales of Investments', 'Profit/Loss in Forex', " \
                  "'Total Adjustments (PBT & Extraordinary Items)', 'Operating Profit before Working Capital " \
                  "Changes', and 'Trade & Other Receivables'"

    def _run(self, stockname: str):
        #pass
        # print("i'm running")
        headlines_response = get_cashflow(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res= await get_cashflow_as(stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

##################################################

class qrcheck(BaseModel):
    stockname: str = Field(..., description="name of stock mentioned in user query")
    quartername: str = Field(..., description="qurater metioned in user query -use only Q1,Q2,Q3,Q4")
    year: int = Field(..., description="year mentioned in user query")

def get_quarterly_results(stock_name,qr,yr):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    asked_data=f"{qr}-{yr}"
    # Define the API endpoint URL with the stock code
    api_url = f'http://airrchipapis.cmots.com/api/QuarterlyResults/{code}/C'
    urls = f'http://airrchipapis.cmots.com/api/QuarterlyResults/{code}/S'
    
    # Set up the headers with the global token
    headers = {'Authorization': f'Bearer {token}'}
    
    # Make the GET request to the API with headers
    response = requests.get(api_url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Get the original data from the API response
        original_data = response.json()['data']
        
        restructured_data = {}
        for entry in original_data:
            column_name = entry["COLUMNNAME"]
            rid = entry["RID"]
            
            for key, value in entry.items():
                if key.startswith("Y"):
                    year = int(key[1:5])
                    month = int(key[5:7])
                    
                    # Map month number to month name and quarter
                    month_info = {
                        3: ("Mar", "Q4"),
                        6: ("Jun", "Q1"),
                        9: ("Sep", "Q2"),
                        12: ("Dec", "Q3")
                    }
                    if month not in month_info:
                        continue  # Skip if not a quarter-end month
                    
                    month_name, quarter = month_info[month]
                    
                    # Calculate the financial year
                    # if month <= 3:
                    #     financial_year = f"{year-1}-{year}"
                    # else:
                    #     financial_year = f"{year}-{year+1}"
                    
                    # # Format the date and add financial year info
                    # date_key = f"{month_name} {year}"
                    # fy_info = f"{quarter} of FY {financial_year}"
                    # full_key = f"Quarterly Results {date_key} ({fy_info})"
                    
                    if month <= 3:
                        financial_year = f"{year-1}"
                    else:
                        financial_year = f"{year}"
                        
                    # Format the date and add financial year info
                    date_key = f"{month_name} {year}"
                    fy_info = f"{quarter}-{financial_year}"
                    #full_key = f"{date_key} ({fy_info})"
                    full_key = fy_info
                    if full_key not in restructured_data:
                        restructured_data[full_key] = {}
                    
                    restructured_data[full_key][column_name] = value
        
        # Sort the data by date in reverse order
        def sort_key(x):
            # Use regular expression to extract year and month
            match = re.search(r'(\w+)\s(\d{4})', x)
            if match:
                month, year = match.groups()
                year = int(year)
                month_order = {'Mar': 3, 'Dec': 2, 'Sep': 1, 'Jun': 0}
                return (-year, -month_order[month])
            return (0, 0)  # Default return if the pattern doesn't match
        
        sorted_dates = sorted(restructured_data.keys(), key=sort_key)
        #print(sorted_dates)
        sorted_data = {}
        for date in sorted_dates:
            if date == asked_data:
                sorted_data[date] = restructured_data[date]
        
        # Convert the restructured and sorted data to JSON format
        #json_data = json.dumps(sorted_data, indent=4)
        return sorted_data
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code} - {response.text}")
        return None
    
async def get_quarterly_results_as(stock_name, qr, yr):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg
    
    asked_data = f"{qr}-{yr}"
    api_url = f'http://airrchipapis.cmots.com/api/QuarterlyResults/{code}/C'
    urls = f'http://airrchipapis.cmots.com/api/QuarterlyResults/{code}/S'
    headers = {'Authorization': f'Bearer {token}'}

    async with httpx.AsyncClient() as client:
        response = await client.get(api_url, headers=headers)
        data = response.json()

        if data.get("data") == [] or data.get("data") is None:
            response = await client.get(urls, headers=headers)
            data = response.json()

    if response.status_code == 200:
        original_data = data['data']
        restructured_data = {}

        for entry in original_data:
            column_name = entry["COLUMNNAME"]
            rid = entry["RID"]

            for key, value in entry.items():
                if key.startswith("Y"):
                    year = int(key[1:5])
                    month = int(key[5:7])
                    
                    # Map month number to month name and quarter
                    month_info = {
                        3: ("Mar", "Q4"),
                        6: ("Jun", "Q1"),
                        9: ("Sep", "Q2"),
                        12: ("Dec", "Q3")
                    }
                    if month not in month_info:
                        continue
                    
                    month_name, quarter = month_info[month]

                    if month <= 3:
                        financial_year = f"{year-1}"
                    else:
                        financial_year = f"{year}"

                    date_key = f"{month_name} {year}"
                    fy_info = f"{quarter}-{financial_year}"
                    full_key = fy_info

                    if full_key not in restructured_data:
                        restructured_data[full_key] = {}

                    restructured_data[full_key][column_name] = value

        # Sort the data by date in reverse order
        def sort_key(x):
            match = re.search(r'(\w+)\s(\d{4})', x)
            if match:
                month, year = match.groups()
                year = int(year)
                month_order = {'Mar': 3, 'Dec': 2, 'Sep': 1, 'Jun': 0}
                return (-year, -month_order[month])
            return (0, 0)

        sorted_dates = sorted(restructured_data.keys(), key=sort_key)
        sorted_data = {}

        for date in sorted_dates:
            if date == asked_data:
                sorted_data[date] = restructured_data[date]

        return sorted_data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    

def get_qr_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_quarterly_results, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class quarterly_results(BaseTool):
    name = "quaterly_results"
    description = "This tool provides comprehensive access to quarterly results of a company **USE THIS TOOL ONLY WHEN NEED OF QUARTERLY DATA **, covering key components such as 'Gross Sales/Income from operations', 'Net Sales/Income from operations', 'Other Operating Income', 'Total Income from operations (net)', 'Total Expenses', 'Finance Costs', 'Profit from ordinary activities after finance costs but before exceptional items', 'Total Tax', 'Net Profit after tax for the Period', 'Total Comprehensive Income', 'Equity', 'Reserve & Surplus', 'Face Value', 'EPS', 'Book Value (Unit Curr.)', 'Dividend Per Share(Rs.)', 'Dividend (%)', 'Debt Equity Ratio', 'Coverage Ratio', 'Earnings before Interest and Tax (EBIT)', 'Earnings before Interest, Taxes, Depreciation and Amortization (EBITDA)', 'Administrative and Selling Expenses', and 'Cost of Sales'."

    def _run(self, stockname: str,quartername:str ,year:int):
        #pass
        # print("i'm running")
        headlines_response = get_quarterly_results(stockname,quartername,year)

        return headlines_response

    async def _arun(self, stockname: str,quartername:str ,year:int):
        res=await get_quarterly_results_as(stockname,quartername,year)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = qrcheck

####################################################  

def get_yearly_results(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    # Define the API endpoint URL with the stock code
    api_url = f'http://airrchipapis.cmots.com/api/Yearly-Results/{code}/C'
    urls = f'http://airrchipapis.cmots.com/api/Yearly-Results/{code}/S'

    # Set up the headers with the global token
    headers = {'Authorization': f'Bearer {token}'}

    # Make the GET request to the API with headers
    response = requests.get(api_url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Restructure the data with properly formatted dates
        original_data = response.json()
        original_data = original_data['data']
        restructured_data = {}
        for entry in original_data:
            column_name = entry["COLUMNNAME"]
            rid = entry["RID"]
            for year_key in entry.keys():
                if year_key.startswith("Y"):
                    year = year_key[1:5]  # Extract year from the year_key
                    month = year_key[5:]  # Extract month from the year_key
                    formatted_date = f"Yearly_Results_{year}-{month} in crs"  # Format date as "Yearly_Results_YYYY-MM"
                    if formatted_date not in restructured_data:
                        restructured_data[formatted_date] = {}
                    restructured_data[formatted_date][column_name] = entry[year_key]

        # Convert the restructured data to JSON format
        #json_data = json.dumps(restructured_data, indent=4)
        return restructured_data
    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code} - {response.text}")
        return None


async def get_yearly_results_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg

    api_url = f'http://airrchipapis.cmots.com/api/Yearly-Results/{code}/C'
    urls = f'http://airrchipapis.cmots.com/api/Yearly-Results/{code}/S'
    headers = {'Authorization': f'Bearer {token}'}

    async with httpx.AsyncClient() as client:
        response = await client.get(api_url, headers=headers)
        data = response.json()

        if data.get("data") == [] or data.get("data") is None:
            response = await client.get(urls, headers=headers)
            data = response.json()

    if response.status_code == 200:
        original_data = data.get('data', [])
        restructured_data = {}

        for entry in original_data:
            column_name = entry["COLUMNNAME"]
            rid = entry["RID"]

            for year_key in entry.keys():
                if year_key.startswith("Y"):
                    year = year_key[1:5]  # Extract year
                    month = year_key[5:]  # Extract month
                    formatted_date = f"Yearly_Results_{year}-{month} in crs"  # Format the date
                    
                    if formatted_date not in restructured_data:
                        restructured_data[formatted_date] = {}
                    
                    restructured_data[formatted_date][column_name] = entry[year_key]

        return restructured_data
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    

def get_yr_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_yearly_results, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class yearly_results(BaseTool):
    name = "yearly_results"
    description = "This tool provides comprehensive access to yearly results not yearly ratios of a company, covering key components such as 'Gross Sales/Income from operations', 'Net Sales/Income from operations', 'Other Operating Income', 'Total Income from operations (net)', 'Total Expenses', 'Finance Costs', 'Profit from ordinary activities after finance costs but before exceptional items', 'Total Tax', 'Net Profit after tax for the Period', 'Total Comprehensive Income', 'Equity', 'Reserve & Surplus', 'Face Value', 'EPS', 'Book Value (Unit Curr.)', 'Dividend Per Share(Rs.)', 'Dividend (%)', 'Debt Equity Ratio', 'Coverage Ratio', 'Earnings before Interest and Tax (EBIT)', 'Earnings before Interest, Taxes, Depreciation and Amortization (EBITDA)', 'Administrative and Selling Expenses', and 'Cost of Sales'."

    def _run(self, stockname: str):
        #pass
        # print("i'm running")
        headlines_response = get_yearly_results(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res=await get_yearly_results_as(stockname)
        return res
        #raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput


# ratios

def get_margin(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/MarginRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/MarginRatios/{code}/S"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        #data = response.json()
        if data['data']:
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{int(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v for k, v in entry.items() if
                 k != 'co_code'} for entry in financial_data]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {str(int(entry['YRC']) // 100): {k: v for k, v in entry.items() if k != 'YRC'} for entry in
                           modified_financial_data}

            # Display the result
            return result_dict


async def get_margin_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/MarginRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/MarginRatios/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        data = response.json()

        if data.get("data") == [] or data.get("data") is None:
            response = await client.get(urls, headers=headers)
            data = response.json()

    if response.status_code == 200:
        if data.get('data'):
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{int(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v 
                 for k, v in entry.items() if k != 'co_code'} 
                for entry in financial_data
            ]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {
                str(int(entry['YRC']) // 100): {
                    k: v for k, v in entry.items() if k != 'YRC'
                } 
                for entry in modified_financial_data
            }

            return result_dict
        else:
            return {"message": "No financial data available"}
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    

def get_mr_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_margin, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class margin_ratios(BaseTool):
    name = "margin_ratios"
    description = "USE THIS TOOL TO GET INFO ABOUT MARGIN RATIOS WHICH ALSO INCLUDE PBIDTIM EBITM PreTaxMargin PATM CPM"

    def _run(self, stockname: str):
        
        # print("i'm running")
        headlines_response = get_margin(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res=await get_margin_as(stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

###########################


def get_performance(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/PerformanceRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/PerformanceRatios/{code}/S"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        #data = response.json()
        if data['data']:
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v for k, v in entry.items() if
                 k != 'co_code'} for entry in financial_data]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {str(int(entry['YRC']) // 100): {k: v for k, v in entry.items() if k != 'YRC'} for entry in
                           modified_financial_data}

            # Display the result
            return result_dict


async def get_performance_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/PerformanceRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/PerformanceRatios/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        data = response.json()

        if not data.get("data"):
            response = await client.get(urls, headers=headers)
            data = response.json()

    if response.status_code == 200:
        if data.get('data'):
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v 
                 for k, v in entry.items() if k != 'co_code'} 
                for entry in financial_data
            ]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {
                str(int(entry['YRC']) // 100): {
                    k: v for k, v in entry.items() if k != 'YRC'
                } 
                for entry in modified_financial_data
            }

            return result_dict
        else:
            return {"message": "No performance data available"}
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    

def get_pr_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_performance, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class performance_ratios(BaseTool):
    name = "performance_ratios"
    description = "USE THIS TOOL TO GET INFO ABOUT PERFORMANCE RATIOS WHICH ALSO INCLUDE ROA AND ROE INFORMATION"

    def _run(self, stockname: str):
        
        # print("i'm running")
        headlines_response = get_performance(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res = await get_performance_as(stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

####################################

def get_efficiency(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/EfficiencyRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/EfficiencyRatios/{code}/S"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        #data = response.json()
        if data['data']:
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v for k, v in entry.items() if
                 k != 'co_code'} for entry in financial_data]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {str(int(entry['YRC']) // 100): {k: v for k, v in entry.items() if k != 'YRC'} for entry in
                           modified_financial_data}

            # Display the result
            return result_dict

async def get_efficiency_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/EfficiencyRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/EfficiencyRatios/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        data = response.json()

        if not data.get("data"):
            response = await client.get(urls, headers=headers)
            data = response.json()

    if response.status_code == 200:
        if data.get('data'):
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v 
                 for k, v in entry.items() if k != 'co_code'} 
                for entry in financial_data
            ]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {
                str(int(entry['YRC']) // 100): {
                    k: v for k, v in entry.items() if k != 'YRC'
                } 
                for entry in modified_financial_data
            }

            return result_dict
        else:
            return {"message": "No efficiency data available"}
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    

def get_er_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_efficiency, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class efficiency_ratios(BaseTool):
    name = "efficiency_ratios"
    description = "USE THIS TOOL TO GET INFO ABOUT EFFICIENCY RATIOS WHICH ALSO INCLUDE FIXEDCAPITALS_SALES, RECEIVABLEDAYS, INVENTORYDAYS, PAYABLEDAYS"

    def _run(self, stockname: str):
        #pass
        # print("i'm running")
        headlines_response = get_efficiency(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res= await get_efficiency_as(stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

##################################

def get_future(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/FinancialStabilityRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/FinancialStabilityRatios/{code}/S"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        #data = response.json()
        if data['data']:
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v for k, v in entry.items() if
                 k != 'co_code'} for entry in financial_data]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {str(int(entry['YRC']) // 100): {k: v for k, v in entry.items() if k != 'YRC'} for entry in
                           modified_financial_data}

            # Display the result
            return result_dict


async def get_future_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/FinancialStabilityRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/FinancialStabilityRatios/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)
        data = response.json()

        if not data.get("data"):
            response = await client.get(urls, headers=headers)
            data = response.json()

    if response.status_code == 200:
        if data.get('data'):
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v 
                 for k, v in entry.items() if k != 'co_code'} 
                for entry in financial_data
            ]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {
                str(int(entry['YRC']) // 100): {
                    k: v for k, v in entry.items() if k != 'YRC'
                } 
                for entry in modified_financial_data
            }

            return result_dict
        else:
            return {"message": "No financial stability data available"}
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    

def get_fr_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_future, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class Financial_Stability_ratios(BaseTool):
    name = "Financial_Stability_ratios"
    description = "USE THIS TOOL TO GET INFO ABOUT Financial Stability RATIOS WHICH ALSO INCLUDE TOTALDEBT_EQUITY, CURRENTRATIO, QUICKRATIO, INTERESTCOVER, TOTALDEBT_MCAP"

    def _run(self, stockname: str):
        # pass
        # # print("i'm running")
        headlines_response = get_future(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res= await get_future_as(stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

#################################

def get_val(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/ValuationRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/ValuationRatios/{code}/S"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        #data = response.json()
        if data['data']:
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v for k, v in entry.items() if
                 k != 'co_code'} for entry in financial_data]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {str(int(entry['YRC']) // 100): {k: v for k, v in entry.items() if k != 'YRC'} for entry in
                           modified_financial_data}

            # Display the result
            return result_dict


async def get_val_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/ValuationRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/ValuationRatios/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        # First request
        response = await client.get(url, headers=headers)
        data = response.json()

        # If no data is found, try the second URL
        if not data.get("data"):
            response = await client.get(urls, headers=headers)
            data = response.json()

    # If the response was successful (status code 200)
    if response.status_code == 200:
        if data.get('data'):
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v 
                 for k, v in entry.items() if k != 'co_code'} 
                for entry in financial_data
            ]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {
                str(int(entry['YRC']) // 100): {
                    k: v for k, v in entry.items() if k != 'YRC'
                } 
                for entry in modified_financial_data
            }

            return result_dict
        else:
            return {"message": "No valuation data available"}
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None
    

def get_vr_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_val, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class Valuation_ratios(BaseTool):
    name = "Valuation_ratios"
    description = "USE THIS TOOL TO GET INFO ABOUT Valuation RATIOS WHICH ALSO INCLUDE PE, PRICE_BOOKVALUE, DIVIDENDYIELD, EV_EBITDA, MCAP_SALES"

    def _run(self, stockname: str):
        
        # print("i'm running")
        headlines_response = get_val(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res =await get_val_as(stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

##################################


def get_cash(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/CashFlowRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/CashFlowRatios/{code}/S"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        #data = response.json()
        if data['data']:
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v for k, v in entry.items() if
                 k != 'co_code'} for entry in financial_data]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {str(int(entry['YRC']) // 100): {k: v for k, v in entry.items() if k != 'YRC'} for entry in
                           modified_financial_data}

            # Display the result
            return result_dict

async def get_cash_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/CashFlowRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/CashFlowRatios/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        # First request
        response = await client.get(url, headers=headers)
        data = response.json()

        # If no data is found, try the second URL
        if not data.get("data"):
            response = await client.get(urls, headers=headers)
            data = response.json()

    # If the response was successful (status code 200)
    if response.status_code == 200:
        if data.get('data'):
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v 
                 for k, v in entry.items() if k != 'co_code'} 
                for entry in financial_data
            ]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {
                str(int(entry['YRC']) // 100): {
                    k: v for k, v in entry.items() if k != 'YRC'
                } 
                for entry in modified_financial_data
            }

            return result_dict
        else:
            return {"message": "No cash flow data available"}
    else:
        print(f"Error: {response.status_code} - {response.text}")
        return None

def get_cr_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_cash, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class cashflow_ratios(BaseTool):
    name = "cashflow_ratios"
    description = "USE THIS TOOL TO GET INFO ABOUT CASH FLOW RATIOS WHICH ALSO INCLUDE CASHFLOWPERSHARE, PRICETOCASHFLOWRATIO, FREECASHFLOWPERSHARE, PRICETOFREECASHFLOW, FREECASHFLOWYIELD, SALESTOCASHFLOWRATIO"

    def _run(self, stockname: str):
        
        # print("i'm running")
        headlines_response = get_cash(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res=await get_cash_as(stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

#######################################

def get_liq(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/LiquidityRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/LiquidityRatios/{code}/S"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        #data = response.json()
        if data['data']:
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, float)) and k != 'YRC' else v for k, v in entry.items() if
                 k not in ['co_code', 'InterestExpended_to_Interestearned', 'Interestincome_to_Totalfunds',
                           'InterestExpended_to_Totalfunds', 'CASA']} for entry in financial_data]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {str(int(entry['YRC']) // 100): {k: v for k, v in entry.items() if k != 'YRC'} for entry in
                           modified_financial_data}

            return result_dict
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

async def get_liq_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg

    url = f"http://airrchipapis.cmots.com/api/LiquidityRatios/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/LiquidityRatios/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        # First request
        response = await client.get(url, headers=headers)
        data = response.json()

        # If no data is found, try the second URL
        if not data.get("data"):
            response = await client.get(urls, headers=headers)
            data = response.json()

    # If the response was successful (status code 200)
    if response.status_code == 200:
        if data.get('data'):
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v 
                 for k, v in entry.items() if k not in ['co_code', 'InterestExpended_to_Interestearned', 'Interestincome_to_Totalfunds', 'InterestExpended_to_Totalfunds', 'CASA']}
                for entry in financial_data
            ]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {
                str(int(entry['YRC']) // 100): {
                    k: v for k, v in entry.items() if k != 'YRC'
                } 
                for entry in modified_financial_data
            }

            return result_dict
        else:
            return {"message": "No liquidity ratio data available"}
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

def get_lr_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_liq, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class Liquidity_ratios(BaseTool):
    name = "Liquidity_ratios"
    description = "USE THIS TOOL TO GET INFO ABOUT Liquidity RATIOS WHICH ALSO INCLUDE LOANS_TO_DEPOSITS, CASH_TO_DEPOSITS, INVESTMENT_TODEPOSITS, INCLOAN_TO_DEPOSIT, CREDIT_TO_DEPOSITS, INTERESTEXPENDED_TO_INTERESTEARNED, INTERESTINCOME_TO_TOTALFUNDS, INTERESTEXPENDED_TO_TOTALFUNDS, CASA"

    def _run(self, stockname: str):
        
        # print("i'm running")
        headlines_response = get_lr_multiple(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res=await get_liq_as(stockname)
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

######################################


def get_roe(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/RatiosReturn/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/RatiosReturn/{code}/S"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        #data = response.json()
        if data['data']:
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, float)) and k != 'YRC' else v for k, v in entry.items()
                 if
                 k not in ['CO_CODE']} for entry in financial_data]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {str(int(entry['YRC']) // 100): {k: v for k, v in entry.items() if k != 'YRC'} for entry
                           in
                           modified_financial_data}

            return result_dict
    else:
        print(f"Error: {response.status_code}")
        print(response.text)

async def get_roe_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg

    url = f"http://airrchipapis.cmots.com/api/RatiosReturn/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/RatiosReturn/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        # First request
        response = await client.get(url, headers=headers)
        data = response.json()

        # If no data is found, try the second URL
        if not data.get("data"):
            response = await client.get(urls, headers=headers)
            data = response.json()

    # If the response was successful (status code 200)
    if response.status_code == 200:
        if data.get('data'):
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v 
                 for k, v in entry.items() if k != 'CO_CODE'}
                for entry in financial_data
            ]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {
                str(int(entry['YRC']) // 100): {
                    k: v for k, v in entry.items() if k != 'YRC'
                } 
                for entry in modified_financial_data
            }

            return result_dict
        else:
            return {"message": "No return on equity data available"}
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    
def get_rr_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_roe, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class returns_ratios(BaseTool):
    name = "returns_ratios"
    description = "USE THIS TOOL TO GET INFO ABOUT RETURNS RATIOS WHICH ALSO INCLUDE RETURN_ROE, RETURN_ROE_NETPROFIT, RETURN_ROE_NETWORTH, RETURN_ROCE, RETURN_ROCE_EBIT, RETURN_ROCE_CAPITALEMPLOYED, RETURN_RETURNONASSETS, RETURN_ROA_NETPROFIT, RETURN_ROA_TOTALFIXEDASSETS"

    def _run(self, stockname: str):
        
        # print("i'm running")
        headlines_response = get_rr_multiple(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res = await get_roe_as(stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

##################################

def get_solvency(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/RatiosSolvency/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/RatiosSolvency/{code}/S"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        #data = response.json()
        if data['data']:
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, float)) and k != 'YRC' else v for k, v in entry.items() if
                 k not in ['CO_CODE']} for entry in financial_data]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {str(int(entry['YRC'])): {k: v for k, v in entry.items() if k != 'YRC'} for entry in
                           modified_financial_data}

            # quarter_mapping = {'03': 'Q1', '06': 'Q2', '09': 'Q3', '12': 'Q4'}

            # Modify the 'QtrEnd' values in the result_dict keys
            updated_result_dict = {
                f"{entry[:4]}": values
                for entry, values in result_dict.items()
            }

            # Display the result with updated keys
            return updated_result_dict

    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        print(response.text)


async def get_solvency_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg

    url = f"http://airrchipapis.cmots.com/api/RatiosSolvency/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/RatiosSolvency/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        # First request
        response = await client.get(url, headers=headers)
        data = response.json()

        # If no data is found, try the second URL
        if not data.get("data"):
            response = await client.get(urls, headers=headers)
            data = response.json()

    # If the response was successful (status code 200)
    if response.status_code == 200:
        if data.get('data'):
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v 
                 for k, v in entry.items() if k != 'CO_CODE'}
                for entry in financial_data
            ]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {
                str(int(entry['YRC'])): {
                    k: v for k, v in entry.items() if k != 'YRC'
                } 
                for entry in modified_financial_data
            }

            # Modify the 'YRC' values in the result_dict keys for the year
            updated_result_dict = {
                f"{entry[:4]}": values
                for entry, values in result_dict.items()
            }

            return updated_result_dict
        else:
            return {"message": "No solvency data available"}
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    

def get_sr_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_solvency, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class Solvency_ratios(BaseTool):
    name = "Solvency_ratios"
    description = "USE THIS TOOL TO GET INFO ABOUT Solvency RATIOS WHICH ALSO INCLUDE SOLVENCY_TOTALDEBTTOEQUITYRATIO, SOLVENCY_TOTALDEBTTOEQUITYRATIO_TOTALDEBT, SOLVENCY_TOTALDEBTTOEQUITYRATIO_NETWORTH, SOLVENCY_INTERESTCOVERAGERATIO, SOLVENCY_INTERESTCOVERAGERATIO_EBIT, SOLVENCY_INTERESTCOVERAGERATIO_INTERESTPAYMENTS, SOLVENCY_CURRENTRATIO, SOLVENCY_CURRENTRATIO_CURRENTASSET, SOLVENCY_CURRENTRATIO_CURRENTLIABILITIES"

    def _run(self, stockname: str):
        
        # print("i'm running")
        headlines_response = get_sr_multiple(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res= await get_solvency_as(stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

################################

def get_growth(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    
    url = f"http://airrchipapis.cmots.com/api/GrowthRatio/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/GrowthRatio/{code}/S"
    #token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1bmlxdWVfbmFtZSI6ImFpcnJjaGlwIiwicm9sZSI6IkFkbWluIiwibmJmIjoxNzA4OTM5ODM2LCJleHAiOjE3MTY3MTU4MzYsImlhdCI6MTcwODkzOTgzNiwiaXNzIjoiaHR0cDovL2xvY2FsaG9zdDo1MDE5MSIsImF1ZCI6Imh0dHA6Ly9sb2NhbGhvc3Q6NTAxOTEifQ.kpUxCvvDw_6bVYHinm0hXa-2bZ21BSWlthSeZMl4mp0"  # Replace with your actual authorization token
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)
    data = response.json()
    if (data.get("data") == [] or data.get("data") is None):
        response = requests.get(urls, headers=headers)
        data = response.json()

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        #data = response.json()
        if data['data']:
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, float)) and k != 'YRC' else v for k, v in entry.items() if
                 k not in ['co_code']} for entry in financial_data]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {str(int(entry['YRC'])): {k: v for k, v in entry.items() if k != 'YRC'} for entry in
                           modified_financial_data}

            # quarter_mapping = {'03': 'Q1', '06': 'Q2', '09': 'Q3', '12': 'Q4'}

            # Modify the 'QtrEnd' values in the result_dict keys
            updated_result_dict = {
                f"{entry[:4]}": values
                for entry, values in result_dict.items()
            }

            # Display the result with updated keys
            return updated_result_dict

    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        print(response.text)

async def get_growth_as(stock_name):
    code = find_stock_code(stock_name)
    print(code)
    if code:
        code = int(code)
    if code is None:
        return msg

    url = f"http://airrchipapis.cmots.com/api/GrowthRatio/{code}/C"
    urls = f"http://airrchipapis.cmots.com/api/GrowthRatio/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    async with httpx.AsyncClient() as client:
        # Try the first URL
        response = await client.get(url, headers=headers)
        data = response.json()

        # If no data, try the second URL
        if not data.get("data"):
            response = await client.get(urls, headers=headers)
            data = response.json()

    # Check for success response and valid data
    if response.status_code == 200:
        if data.get('data'):
            financial_data = data['data'][:4]

            # Process financial data
            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, int)) and k != 'YRC' else v 
                 for k, v in entry.items() if k not in ['co_code']}
                for entry in financial_data
            ]

            # Create dictionary with 'YRC' values as keys
            result_dict = {
                str(int(entry['YRC'])): {
                    k: v for k, v in entry.items() if k != 'YRC'
                }
                for entry in modified_financial_data
            }

            # Modify keys to include only the year
            updated_result_dict = {
                f"{entry[:4]}": values
                for entry, values in result_dict.items()
            }

            return updated_result_dict
        else:
            return {"message": "No growth ratio data available"}
    else:
        # Handle errors
        print(f"Error: {response.status_code}")
        print(response.text)
        return None
    

def get_gr_multiple(stock_names):
    results = {}

    # Using ThreadPoolExecutor for parallel processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(get_growth, stock): stock for stock in stock_names}
        
        for future in concurrent.futures.as_completed(futures):
            stock = futures[future]
            try:
                data = future.result()
                results[stock] = data
            except Exception as exc:
                results[stock] = f"An error occurred: {exc}"

    return results


class growth_ratios(BaseTool):
    name = "growth_ratios"
    description = "USE THIS TOOL TO GET INFO ABOUT GROWTH RATIOS WHICH ALSO INCLUDE NETSALESGROWTH, EBITDAGROWTH, EBITGROWTH, PATGROWTH, EPS"

    def _run(self, stockname: str):
        
        # print("i'm running")
        headlines_response = get_gr_multiple(stockname)

        return headlines_response

    async def _arun(self, stockname: str):
        res= await get_growth_as(stockname)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = stocknameInput

##############################

def get_ratios(stock_name):
    
    margin = get_margin(stock_name)
    perf = get_performance(stock_name)
    eff = get_efficiency(stock_name)
    fut = get_future(stock_name)
    val = get_val(stock_name)
    cash = get_cash(stock_name)
    liq = get_liq(stock_name)
    roe = get_roe(stock_name)
    sov = get_solvency(stock_name)
    growth = get_growth(stock_name)
    return {'Margin ratios': margin, "performance ratios": perf, "Efficiency": eff, "Financial Stability Ratios": fut,
            'Valuation Ratios': val, "Cash Flow Ratios ": cash, "Liquidity Ratio": liq, "Return Ratios": roe,
            "Solvency Ratios": sov ,"Growth Ratio":growth
            }

class ratios(BaseTool):
    name = "ratios"
    description = "USE THIS TOOL TO GET INFO ABOUT DIFFERENT RATIOS- margin ratios,performance ratios,efficiency ratios,Financial Stability Ratios"\
                    "Valuation Ratios,Cash Flow Ratios,Liquidity Ratio,Return Ratios,Solvency Ratios,Growth Ratios of any stock"

    def _run(self, stockname: str):
        print("i'm running")
        headlines_response = get_ratios(stockname)

        return headlines_response

    def _arun(self, stockname: str):
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = TickerCheckInput

############################

def get_investor_holding(stock_name):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return "No data for this stock. Only for Indian market stocks."
    url = f"http://airrchipapis.cmots.com/api/ShareholdingMorethanonePerDetails/{code}"
    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Make a GET request with headers and payload
    response = requests.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Parse JSON response
        data = response.json()
        if data['data']:
            # Sort data by `yrc` in descending order
            sorted_data = sorted(data['data'], key=lambda x: x['yrc'], reverse=True)
            
            # Get the latest 4 unique `yrc` values
            latest_yrcs = []
            filtered_data = []
            for item in sorted_data:
                if item['yrc'] not in latest_yrcs:
                    latest_yrcs.append(item['yrc'])
                if item['yrc'] in latest_yrcs[:1]:
                    filtered_data.append({"yrc": item['yrc'], "name": item['name'], "PercentageStakeHolding": item['PercentageStakeHolding']})

            # Group the results by `yrc`
            grouped_data = {}
            for item in filtered_data:
                yrc = item["yrc"]
                if yrc not in grouped_data:
                    grouped_data[yrc] = []
                grouped_data[yrc].append({"name": item["name"], "PercentageStakeHolding": item["PercentageStakeHolding"]})

            # Print the grouped data based on yrc
            #print(grouped_data)
            return grouped_data


def get_shareholding_data(stock_name,year):
    code=find_stock_code(stock_name)
    print(code)
    if code:
        code=int(code)
    if code is None:
        return msg
    url = f"http://airrchipapis.cmots.com/api/ShareholdingMorethanonePerDetails/{code}"
    headers = {"Authorization": f"Bearer {token}"}
    
    # Make a GET request with headers
    response = requests.get(url, headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        
        if data['data']:
            # If year is specified (not 0), filter entries containing that year in `yrc`
            if year != 0:
                filtered_data = [
                    {"yrc": item['yrc'], "name": item['name'], "PercentageStakeHolding": item['PercentageStakeHolding']}
                    for item in data['data'] if str(year) in str(item['yrc'])
                ]
            
            # If year is 0, get the latest 4 unique `yrc` values
            else:
                # Sort data by `yrc` in descending order
                sorted_data = sorted(data['data'], key=lambda x: x['yrc'], reverse=True)
                
                # Get the latest 4 unique `yrc` values
                latest_yrcs = []
                filtered_data = []
                for item in sorted_data:
                    if item['yrc'] not in latest_yrcs:
                        latest_yrcs.append(item['yrc'])
                    if item['yrc'] in latest_yrcs[:4]:
                        filtered_data.append({"yrc": item['yrc'], "name": item['name'], "PercentageStakeHolding": item['PercentageStakeHolding']})
            
            # Group the results by `yrc`
            grouped_data = {}
            for item in filtered_data:
                yrc = item["yrc"]
                if yrc not in grouped_data:
                    grouped_data[yrc] = []
                grouped_data[yrc].append({"name": item["name"], "PercentageStakeHolding": item["PercentageStakeHolding"]})
            
            return grouped_data

    else:
        print("Error:", response.status_code)
        return None

async def get_shareholding_data_as(stock_name, year):
    # Find the stock code
    code = find_stock_code(stock_name)
    if code is None:
        return msg
    
    try:
        code = int(code)
    except ValueError:
        return msg

    # API URL and headers
    url = f"http://airrchipapis.cmots.com/api/ShareholdingMorethanonePerDetails/{code}"
    headers = {"Authorization": f"Bearer {token}"}

    # Make an async GET request
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    if response.status_code != 200:
        print("Error:", response.status_code)
        return {"message": f"Failed to fetch data, status code: {response.status_code}"}

    data = response.json()

    if not data.get('data'):
        return {"message": "No shareholding data available"}

    # Filter data based on the year
    if year != 0:
        filtered_data = [
            {"yrc": item['yrc'], "name": item['name'], "PercentageStakeHolding": item['PercentageStakeHolding']}
            for item in data['data'] if str(year) in str(item['yrc'])
        ]
    else:
        # Sort and find the latest 4 unique `yrc` values
        sorted_data = sorted(data['data'], key=lambda x: x['yrc'], reverse=True)
        latest_yrcs = []
        filtered_data = []
        for item in sorted_data:
            if item['yrc'] not in latest_yrcs:
                latest_yrcs.append(item['yrc'])
            if item['yrc'] in latest_yrcs[:4]:
                filtered_data.append({
                    "yrc": item['yrc'],
                    "name": item['name'],
                    "PercentageStakeHolding": item['PercentageStakeHolding']
                })

    # Group the results by `yrc`
    grouped_data = {}
    for item in filtered_data:
        yrc = item["yrc"]
        if yrc not in grouped_data:
            grouped_data[yrc] = []
        grouped_data[yrc].append({
            "name": item["name"],
            "PercentageStakeHolding": item["PercentageStakeHolding"]
        })

    return grouped_data


class yearCheckInput(BaseModel):
    stockname: str = Field(..., description="name of the stock present in the query")
    year: int = Field(..., description="year mentioned in the query ,if no year is mentioned in query set year as 0")



class company_holding(BaseTool):
    name = "company_holding"
    #description = "USE THIS TOOL TO GET INFO INVESTOR AND PROMOTER HOLDINGS IN A PARTICULAR COMPANY"
    description="USE THIS TOOL TO GET INFORMATION ABOUT INVESTORS AND PROMOTERS WHO HOLD SHARES IN A PARTICULAR COMPANY"


    def _run(self, stockname: str ,year:int):
       # pass
        # print("i'm running")
        response = get_shareholding_data(stockname,year)

        return response

    async def _arun(self, stockname: str,year:int):
        res = await get_shareholding_data_as(stockname,year)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = yearCheckInput

#############################################
import asyncpg
import os
import aiofiles
import csv
import json

db_url = os.getenv('DATABASE_URL')

async def get_co_codes_by_name(names, yrc=202409):
    """
    Asynchronously retrieves company codes and their total stake percentage from the database
    for a given list of names and a specific year/quarter code (yrc).
    This function now uses the global DB_POOL for efficient connection management.
    """
    if not DB_POOL:
        print("ERROR: Database connection pool is not initialized.")
        return []

    # Create placeholders for the list of names in the SQL query
    placeholders = ', '.join(f"${i+1}" for i in range(len(names)))
    
    # Construct the SQL query to get the total stake for each company
    query = f"""
    SELECT co_code, SUM("PercentageStakeHolding") AS total_stake
    FROM public."shareHoldingPatternMoreThan1"
    WHERE "name" IN ({placeholders})
      AND "yrc" = ${len(names) + 1}
    GROUP BY co_code
    ORDER BY co_code ASC;
    """

    # Combine the names and yrc into a single list of parameters for the query
    params = names + [yrc]
    
    try:
        # Acquire a connection from the pool
        async with DB_POOL.acquire() as connection:
            # Execute the query with the prepared parameters
            results = await connection.fetch(query, *params)

            # Process and return the results if any are found
            if results:
                return [(record['co_code'], record['total_stake']) for record in results]
            else:
                return []  # Return an empty list if no matching records are found
    except asyncpg.PostgresError as e:
        # Log any database errors that occur
        print(f"Database error in get_co_codes_by_name: {e}")
        return []


async def get_company_info_from_csv(co_codes):
    """
    Retrieve company names from a CSV file based on co_code matching the list of co_codes.

    Parameters:
        co_codes (list): List of co_codes to search in the CSV file.

    Returns:
        list: A list of company names that match the co_codes.
    """
    company_info = []

    # Path to your CSV file
    csv_file_path = r'csvdata/6000stocks.csv'

    try:
        with open(csv_file_path, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file)
            
            for row in reader:
                try:
                    # Convert 'Company Code' to float and then to int, handling float-like values like '6.0'
                    co_code = int(float(row['Company Code']))
                    
                    # Only process rows where co_code matches one of the provided co_codes
                    if co_code in co_codes:
                        company_info.append(row['Company Name'])
                except ValueError as e:
                    # Handle rows with invalid co_code or missing data
                    print(f"Skipping row with invalid data: {row['Company Code']} (Error: {e})")
    except FileNotFoundError:
        print(f"Error: The CSV file at {csv_file_path} was not found.")
    except Exception as e:
        print(f"Error reading the CSV file: {e}")
    
    return company_info

from langchain.output_parsers import CommaSeparatedListOutputParser
async def  get_similar_promoter_names(name,listt):
    p_prompt = """
    You are a highly skilled Indian stock market investor. Your task is to process the given list of names and classify them based on the following criteria:

    Institutional Names: Identify and select all names from the list that closely match or are similar to the institutional name provided.
    Person Names: If the name represents an individual, select only those that are an exact match or contain initials/abbreviations.

    Institutional or person name:{in}
    list of names:{lt}
    {format_instructions}



    """
    output_parser = CommaSeparatedListOutputParser()
    format_instructions = output_parser.get_format_instructions()
    GPT4o_mini=ChatOpenAI(temperature=0.2, model="gpt-4o-mini")
    P_prompt = PromptTemplate(template=p_prompt, input_variables=["q"],partial_variables={"format_instructions": format_instructions})
    name_chain = P_prompt | GPT4o_mini | CommaSeparatedListOutputParser()
    input_data = {
        "in":name,
        "lt":listt
    }
    #res=llm_chain_res.predict(query=q)
    res=name_chain.ainvoke(input_data)
    return await res

async def get_promoter_holdings(name_to_search):
    """
    Main function to get company name and percentage stake in JSON format based on the company name.

    Parameters:
        name_to_search (str): The name of the company to search for.

    Returns:
        str: JSON string with company name and percentage stake.
    """
    # Retrieve co_codes and Percentage Stake Holding for the given company name
    results = vs_promoter.similarity_search_with_score(
    name_to_search,k=100
         )
    # name=results[0].page_content
    #print(name)
    simi_names = []  # Initialize an empty list to store page_content

    # if results:  # Check if results are not empty
    #     first_record = results[0]
        
    #     if first_record[1] > 0.2:
    #         # If the first record satisfies the condition, include only it
    #         simi_names.append(first_record[0].page_content)
    #     else:
    #         # If the first record does not satisfy the condition, include all others that do
    #         simi_names = [r[0].page_content for r in results if r[1] <= 0.2]

    if results:  # Check if results are not empty
    # Find the index of the first record with a distance less than 0.2
        index = next((i for i, r in enumerate(results) if r[1] < 0.2), None)
        
        if index is not None:
            # If a record with distance < 0.2 exists, include it and the next 10 names
            simi_names = [r[0].page_content for r in results[index:index + 10]]
        else:
            # If no record with distance < 0.2 exists, include all names
            simi_names = [r[0].page_content for r in results]
    similar_names =await get_similar_promoter_names(name_to_search,simi_names)
    # print(len(similar_names))
    # print(similar_names)
    co_codes_and_stake = await get_co_codes_by_name(similar_names)

    if co_codes_and_stake:
        # Extract co_codes from the result to match with the CSV
        co_codes = [record[0] for record in co_codes_and_stake]

        # Fetch company info from CSV using the retrieved co_codes
        company_names = await get_company_info_from_csv(co_codes)

        if company_names:
            result = []
            for idx, company_name in enumerate(company_names):
                # For each company name, fetch the corresponding Percentage Stake Holding from the database result
                percentage_stake = co_codes_and_stake[idx][1]
                result.append({
                    "company_name": company_name,
                    "percentage_stake": percentage_stake
                })

            # Return the result in JSON format
            return json.dumps(result)
        else:
            return json.dumps({"error": "No company info found for the retrieved co_codes."})
    else:
        return json.dumps({"error": f"No co_codes found for '{name_to_search}'."})

class promoter(BaseModel):
    name: str = Field(..., description="name of the promoter/company present in the query,dont change the name consider as it is capitilaise each word")


class promoter_Institutional_holding(BaseTool):
    name = "promoter_Institutional_holding"
    #description = "USE THIS TOOL TO GET INFO ABOUT PARTICULAR PROMOTER OR COMPANY , IN WHICH STOCKS THEY INVESTED OR HAVE HOLDINGS"
    description="USE THIS TOOL TO GET INFORMATION ABOUT THE STOCKS ,THAT A PARTICULAR COMPANY/PERSON HAS INVESTED IN."

    def _run(self, stockname: str ,year:int):
       # pass
        # print("i'm running")
        #response = (stockname,year)

        pass

    async def _arun(self, name: str):
        res = await get_promoter_holdings(name)
        return res
        raise NotImplementedError("This tool does not support async")

    args_schema: Optional[Type[BaseModel]] = promoter

vectordb_tools=[daily_yearly_ratios(),profit_loss(),quarterly_results(),balance_sheet(),cash_flow(),yearly_results(),margin_ratios(),performance_ratios(),efficiency_ratios(),Financial_Stability_ratios(),Valuation_ratios(),cashflow_ratios(),Liquidity_ratios(),
                growth_ratios(),returns_ratios(),Solvency_ratios(),company_holding(),promoter_Institutional_holding()]#screener(),

system_message_ind = HumanMessage(
    content="""Assume the role of an Expert Stock Market Assistant/Analyst for the stock market.
**WHILE ANSWERING HOLDINGS TYPE OF QUESTIONS RETURN ALL STOCKS YOU ARE GETTING FROM TOOL,DONT IGNORE ANY STOCKS INCLUDE ALL STOCKS**
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
*IF USER QUESTION IS ABOUT BUY/SELL THEN FORMULATE NICE ANSWER CONSISTS OF THIS -'Frruit is an AI-powered capital markets search engine designed to help you search , discover and analyze market data to make better-informed decisions. However, Frruit does not provide personalized investment advice or recommendations to buy or sell specific securities. For tailored investment guidance, please consult with a licensed financial advisor(USE THIS TEXT ONLY WHEN USER ASKING ABOUT BUY/SELL QUESTIONS).'AND PROVIDE DATA YOU HAVE SO THAT USER CAN DECIDE*

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

def agent_with_session(session_id: str, prompt: str,market:str):

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
    vectordb_tools,
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


# tools=[daily_yearly_ratios()]
#llm_stream = ChatOpenAI(temperature=0.5, model="gpt-4o-mini",callbacks=[StreamingStdOutCallbackHandler()],streaming=True,stream_usage=True)

AI_KEY=os.getenv('AI_KEY')
async def authenticate_ai_key(x_api_key: str = Header(...)):
    if x_api_key != AI_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )



async def agent_stream_not(plan_id,user_id,prompt_history_id,session_id,prompt):
    history = PostgresChatMessageHistory(
        connection_string = psql_url,
        session_id=session_id,
    )

    # Initialize ConversationBufferWindowMemory
    memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, k=6, chat_memory=history)

    agent2 = initialize_agent(
    vectordb_tools,
    llm_stream,
    # agent=AgentType.OPENAI_MULTI_FUNCTIONS,
    agent_kwargs=agent_kwargs_ind,
    memory=memory,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=False,

    )
    with get_openai_callback() as cb:
        async for event in agent2.astream_events(
            {"input": 'what is pe of acc'},
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
        # async for chunk in agent2.astream({"input":prompt}):
        #     if isinstance(chunk, dict):
        #         # Ignore if chunk is a dictionary
        #         continue

        #     if chunk is not None:
        #         # Process non-dict chunks
        #         answer = chunk
        #         #answer = chunk.content
        #         aggregate = chunk if aggregate is None else aggregate + chunk
        #         if answer is not None:
        #             await asyncio.sleep(0.01) 
        #             #print(answer)
        #             yield answer.encode("utf-8")
        #         else:
        #             pass
        #     else:
        #         print("Received None chunk")

        total_tokens=cb.total_tokens/1000
        print(total_tokens)

        #store_into_userplan(user_id,total_tokens)
        insert_credit_usage(user_id,plan_id,total_tokens)


async def agent_stream(plan_id,user_id,session_id,prompt,db):
    similar_question = query_similar_questions(prompt,'fundamental_cache')
    print(f"Similar Question: {similar_question}")
    cached_response = db.query(ChatbotResponse).filter(ChatbotResponse.question == similar_question).first()
    print(f"Cached Response: {cached_response}")

    if similar_question and cached_response:
        # looks up for the corresponding response for
        cached_response.count += 1
        db.commit()
        add_to_memory(prompt, cached_response.response, session_id=session_id)
        res=cached_response.response
        # chunks = res.split('. ') 
        for chunk in res:
            yield chunk.encode("utf-8")  # Yield each chunk as bytes
            await asyncio.sleep(0.01)  # Adjust the delay as needed for a smoother stream
                
        return
    
    history = PostgresChatMessageHistory(
        connection_string = psql_url,
        session_id=session_id,
    )

    # Initialize ConversationBufferWindowMemory
    memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, k=6, chat_memory=history)

    agent2 = initialize_agent(
    vectordb_tools,
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
        if should_store(prompt):
            new_response = ChatbotResponse(
                question=prompt,
                response=answer,
                count=1,
                Total_Tokens=cb.total_tokens,  # Populate the new columns
                Prompt_Tokens=cb.prompt_tokens,
                Completion_Tokens=cb.completion_tokens,
            )
            db.add(new_response)
            db.commit()
            add_questions(prompt,'fundamental_cache')


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
fund_rag = APIRouter()
@fund_rag.post("/fundamental")
async def final_stream(request: InRequest, 
        session_id: str = Query(...), 
        user_id: int = Query(...), 
        plan_id: int = Query(...),
        ai_key_auth: str = Depends(authenticate_ai_key),
        db: Session = Depends(get_db)
    ):
    prompt=request.query
    return StreamingResponse(agent_stream(plan_id,user_id,session_id,prompt,db), media_type="text/event-stream")
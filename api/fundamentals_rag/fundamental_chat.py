from groq import Groq
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
from config import GPT3_16k,GPT4o_mini, chroma_server_client, default_ef
import requests
from langchain_openai import ChatOpenAI
from langchain_community.chat_message_histories.postgres import PostgresChatMessageHistory
from langchain.memory import ConversationBufferWindowMemory
import os
import asyncio
import json
import aiohttp

from dotenv import load_dotenv


load_dotenv(override=True)

ip_address = os.getenv("PG_IP_ADDRESS")
token=os.getenv("CMOTS_BEARER_TOKEN")

client = chroma_server_client

# async def find_stock_code(stock_name):
#     df = pd.read_csv("csvdata/company_codes.csv")
#     company_names = df["Company Name"].tolist()
#     company_codes = df["Company Code"].tolist()
#     threshold = 80
#     match = process.extractOne(stock_name, company_names)
#     if match and match[1] >= threshold:
#         idx = company_names.index(match[0])
#         return company_codes[idx]
#     return 0

# async def get_collection_name(stock_name):
#     df = pd.read_csv("csvdata/company_codes.csv")
#     company_names = df["Company Name"].tolist()
#     company_codes = df["Company Code"].tolist()
#     threshold = 80
#     match = process.extractOne(stock_name, company_names)
#     if match and match[1] >= threshold:
#         idx = company_names.index(match[0])
#         return f"stock_{company_codes[idx]}"
#     return "Not Found"
async def get_code(stock_name):
    df = pd.read_csv("csvdata\6000stocks.csv")

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

async def get_collection_name(stock_name):
    # Load the CSV file into a DataFrame
    #df = pd.read_csv("company_codes2.csv")
    df = pd.read_csv("csvdata\6000stocks.csv")

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
    

async def rank_documents2(original_list, query):
    client = Groq()
    titles = [item.split(":")[0] for item in original_list]
    completion = await client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[
            {
                "role": "system",
                "content": """FOLLOW INSTRUCTIONS. I will provide you with a list of titles of docs and user query.
                Rank the titles based on their relevance to query and return the just TOP 4 IN THE SAME PYTHON LIST IN THE ORDER ITS BEING ASKED IN THE QUERY WITH THE MOST RELEVANT FIRST AND SO ON.
                \nFOLLOW INSTRUCTIONS\n  OUTPUT SHOULD BE JUST THE PYTHON LIST WITH THE RE RANKEND TITLES, PLEASE DON'T RETURN NOTHING ELSE"""
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
    titles_list = completion.choices[0].message.content
    ordered_list = [item for title in titles_list[:4] for item in original_list if title in item][:4]
    return ordered_list

async def reranker(original_list, q):
    titles = [item.split(":")[0] for item in original_list]
    template = """You are an intelligent assistant that can rank passages based on their relevancy to the query.
    I will provide you with a list of titles of docs. Rank the titles based on their relevance to query and return the just TOP 4 IN THE SAME PYTHON LIST.
    OUTPUT SHOULD BE JUST THE PYTHON LIST WITH THE RERANKEND TITLES, NOTHING ELSE.
    query:{query}
    docs:{docs}"""
    prompt = ChatPromptTemplate.from_template(template)
    output_parser = StrOutputParser()
    chain = LLMChain(prompt=prompt, llm=GPT3_16k, output_parser=output_parser)
    titles_list = eval(await chain.arun(query=q, docs=titles))
    ordered_list = [item for title in titles_list for item in original_list if title in item][:4]
    return ordered_list


import asyncio

async def fun2(stock_name, q):
    code = await get_collection_name(stock_name)
    current_date = datetime.date.today()
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
    chain = LLMChain(prompt=prompt, llm=GPT3_16k, output_parser=output_parser)
    new_query = await chain.arun(query=q, date=current_date)
    collection = client.get_collection(name=code, embedding_function=default_ef)
    
    # Wrap the synchronous query method in an asynchronous function
    async def async_query():
        return await asyncio.to_thread(collection.query, query_texts=[new_query], n_results=20)
    
    # Await the result of async_query
    docs = await async_query()
    #output = await reranker(docs['documents'][0], new_query)
    return docs

class TickerCheckInput(BaseModel):
    stockname: str = Field(..., description="name of the stock")
    prompt: str = Field(..., description="elaborate the provided prompt for similarity search")

class tool1(BaseTool):
    name = "stockdatatool"
    description = "use this tool to get financial info about any stock which are not covered in daily_ratios tools"

    async def _arun(self, stockname: str, prompt: str):
        print("i'm running")
        headlines_response = await fun2(stockname, prompt)
        return headlines_response

    def _run(self, stockname: str, prompt: str):
        return asyncio.run(self._arun(stockname, prompt))

    args_schema: Optional[Type[BaseModel]] = TickerCheckInput


llm1 = ChatOpenAI(temperature=0.2, model="gpt-4o-2024-05-13")
llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-1106")

def remove_description(data):
    for item in data:
        if 'Description' in item or 'remark' in item or 'isin' in item or 'co_code' in item:
            item.pop('Description', None)
            item.pop('remark', None)
            item.pop('isin', None)
            item.pop('co_code', None)
    return data

def get_current_date():
    return datetime.datetime.now().date()

date = get_current_date()
corp_prompt = """Given a json data {data}, today's date {date} the user question {query}, use today's date if needed to fetch records.
analyse the json data properly and answer based on user query. Please provide complete answer.
"""
c_q_prompt = PromptTemplate(template=corp_prompt, input_variables=['data', 'query', 'date'])
llm=ChatOpenAI(temperature = 0.5 ,model ='gpt-4o-mini')
corp_chain = LLMChain(prompt=c_q_prompt, llm=llm)

async def bonus(type, year, query):
    if type == 1:
        url = "http://airrchipapis.cmots.com/api/BonusIssues/-"
    elif type == 2:
        url = "http://airrchipapis.cmots.com/api/Split-of-FaceValue/-"
    elif type == 3:
        url = "http://airrchipapis.cmots.com/api/BuyBack/-"
    elif type == 4:
        url = "http://airrchipapis.cmots.com/api/RightIssues/-"
    else:
        return "Invalid type"

    token = os.getenv("CMOTS_BEARER_TOKEN")
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()
        D = data['data']
        filtered_data = [entry for entry in D if entry['AnnouncementDate' if type == 3 else 'BonusDate' if type == 1 else 'SplitDate' if type == 2 else 'AnnouncementDate'].startswith(year)]
        data_clean = remove_description(filtered_data)
        date = get_current_date()
        res = await corp_chain.arun(query=query, data=data_clean, date=date)
        return res
    return f"Error: {response.status_code}"

class queryCheckInput(BaseModel):
    query: str = Field(..., description="return complete query user mentioned without removing anything")
    type: int = Field(..., description="is user asking about bonus set type as 1 ,for splits, face values set type as 2, for buybacks set type as 3, for rights issues set type as 4")
    year: str = Field(..., description="year mentioned in the prompt if no year mentioned set year as 2024")

class screener_bonus(BaseTool):
    name = "stock_screener"
    description = "use this tool when you need info about upcoming bonus issues, stock splits , buybacks ,right issues"

    def _run(self, type: int,year:str,query:str):
        print("i'm running")
        # print(type)
        # print(query)
        headlines_response = bonus(type,year,query)

        return headlines_response

    def _run(self, type: int, year: str, query: str):
        return asyncio.run(self._arun(type, year, query))

    args_schema: Optional[Type[BaseModel]] = queryCheckInput



async def get_daily_ratios(stock_name):
    code = await int(get_code(stock_name))
    headers = {
        'Authorization': f'Bearer {token}'
    }
    #url = f'http://airrchipapis.cmots.com/api/DailyRatios/{code}/S'
    url = f'http://airrchipapis.cmots.com/api/YearlyRatio/{code}/S'
    
    async with aiohttp.ClientSession() as session:
        async with session.get(url, headers=headers) as response:
            result = await response.json()
    
    daily_ratios = {}
    for ratio in result['data']:
        for key, value in ratio.items():
            daily_ratios[key] = value
    
    return json.dumps({"Financial ratios(yearly data)": daily_ratios})

class stocknameInput(BaseModel):
    stockname: str = Field(..., description="name of the stock")

class dailyratios(BaseTool):
    name = "daily_ratios"
    description = "use this tool ONLY WHEN YOU NEED info about this - Market Capitalization (mcap), Enterprise Value, Price-to-Earnings Ratio, Price-to-Book Value Ratio, Dividend Yield, Earnings Per Share, Book Value Per Share, Return on Assets, Return on Equity, Return on Capital Employed, Earnings Before Interest and Taxes, Earnings Before Interest, Taxes, Depreciation, and Amortization, Enterprise Value to Sales Ratio, Enterprise Value to EBITDA Ratio, Net Income Margin, Gross Income Margin, Asset Turnover Ratio, Current Ratio, Debt-to-Equity Ratio, Sales to Total Assets Ratio, Net Debt to EBITDA Ratio, EBITDA Margin, Total Shareholders' Equity, Short-term Debt, Long-term Debt, Shares Outstanding, Diluted Earnings Per Share, Net Sales, Net Profit, Annual Dividend, Cost of Goods Sold, PEG Ratio, Dividend Payout Ratio, Industry Price-to-Earnings Ratio,"
    "IF USER IS NOT ASKING ABOUT ABOVE DONT USE THIS TOOL"

    async def _arun(self, stockname:str):
        print("i'm running")
        # print(type)
        # print(query)
        headlines_response = get_daily_ratios(stockname)

        return headlines_response

    def _run(self, stockname: str):
        return asyncio.run(self._arun(stockname))


    args_schema: Optional[Type[BaseModel]] = stocknameInput

vectordb_tools = [tool1(), screener_bonus(),dailyratios()]

system_message_ind = HumanMessage(
    content="""You are Frruit an  EXPERT AI Stock Market Assistant/Analyst for the indian stock market, Developed by Frruit.
You should answer user queries in an analytical yet beginner friendly way. So that user can gain actionable insights from your analysis.
YOU ANSWERS SHOULD BE ACCURATE USING THE PROVIDED CONTEXT FROM the tools.
If the query cannot be satisfactorily answered using the available tools.
DONT TRY TO DO CALCULATIONS.
NEVER PROVIDE FORMULAS FOR ANYTHING.
When a user asks a question, follow these steps:
1. Identify the relevant financial data needed to answer the query.
2. Analyze the retrieved data and any generated charts to extract key insights and trends.
3. Formulate a concise response that directly addresses the user's question, focusing on the most important findings from your analysis.

- Avoid just simply regurgitating the raw data from the tools. Instead, provide a thoughtful interpretation and summary as well.

KEEP YOUR ANSWERS AS CONCISE AS POSSIBLE. USER PREFERS CONCISE ANSWERS.
Note that The Indian financial year, also known as the fiscal year (fy), 
runs from April 1 to March 31, divided into four quarters: Q1 (April-June), Q2 (July-September), Q3 (October-December), and Q4 (January-March).
use lakhs and crores instead of millions and billions"""
)

agent_kwargs_ind = {
    "system_message": system_message_ind,
}

memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, k=2)

agent2 = initialize_agent(
    vectordb_tools,
    GPT4o_mini,
    agent_kwargs=agent_kwargs_ind,
    memory=memory,
    agent=AgentType.OPENAI_FUNCTIONS,
    verbose=True,
)

async def agent_with_session(session_id: str, prompt: str, market: str):
    history = PostgresChatMessageHistory(
        connection_string=f"postgresql://postgresql:1234@{ip_address}:5432/frruitmicro",
        session_id=session_id,
    )
    memory = ConversationBufferWindowMemory(memory_key="memory", return_messages=True, k=2, chat_memory=history)
    response = await agent2.arun(input=prompt)
    return response
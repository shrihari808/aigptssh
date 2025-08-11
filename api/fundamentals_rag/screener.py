from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI
import os
from langchain_community.callbacks import get_openai_callback
from langchain_core.output_parsers import JsonOutputParser
from langchain.llms import OpenAI
import boto3
import pandas as pd
from time import sleep
import io
import datetime
from dotenv import load_dotenv
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session
from api.caching import add_questions,verify_similarity,query_similar_questions
from config import llm_screener,GPT4o_mini

load_dotenv(override=True)


AWS_ACCESS_KEY = os.getenv('AWS_ACCESS_KEY')
AWS_SECRET_KEY = os.getenv('AWS_SECRET_KEY')
AWS_REGION = os.getenv('AWS_REGION')
S3_OUTPUT_LOCATION = os.getenv('S3_OUTPUT_LOCATION')
openai_api_key=os.getenv('OPENAI_API_KEY')
s_bucket=os.getenv('screener_bucket')

#caching
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





s3_client= boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)

athena_client = boto3.client(
    'athena',
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION
)


#llm1 = ChatOpenAI(temperature=0.3, model="gpt-4o-2024-05-13") 
# llm = ChatOpenAI(temperature = 0.5 ,model ='gpt-4o-mini')
#llm = ChatOpenAI(temperature=0.3, model="gpt-3.5-turbo-1106")
#llm = ChatOpenAI(temperature=0.2, model="gpt-3.5-turbo-0125")


def execute_athena_query(query, database, output_location):
    response = athena_client.start_query_execution(
        QueryString=query,
        QueryExecutionContext={
            'Database': database
        },
        ResultConfiguration={
            'OutputLocation': output_location
        }
    )
    return response


def get_query_results(query_execution_id):
    while True:
        response = athena_client.get_query_execution(QueryExecutionId=query_execution_id)
        status = response['QueryExecution']['Status']['State']
        if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            return response
        sleep(5)


def get_completed_financial_year():
    today = datetime.date.today()
    if today.month >= 4:
        start_year = today.year
        end_year = today.year + 1
    else:
        start_year = today.year - 1
        end_year = today.year
    return start_year



def get_completed_financial_year_and_quarter():
    today = datetime.date.today()
    
    # Determine the completed financial year
    if today.month >= 4:
        start_year = today.year 
        end_year = today.year
    else:
        start_year = today.year - 1
        end_year = today.year - 1

    # Determine the completed quarter
    if today.month >= 4 and today.month <= 6:
        completed_quarter = 'Q4'
        completed_quarter_year = start_year
    elif today.month >= 7 and today.month <= 9:
        completed_quarter = 'Q1'
        completed_quarter_year = end_year
    elif today.month >= 10 and today.month <= 12:
        completed_quarter = 'Q2'
        completed_quarter_year = end_year
    else:
        completed_quarter = 'Q3'
        completed_quarter_year = end_year

    return (start_year, completed_quarter)

com_prompt = """
You are a highly skilled Indian stock market investor and financial advisor. Your task is to validate whether a given question contains any stock name or company name not sector or industry name.

Given question: {q}  
Output the result in JSON format:  

"valid": Return 1 if the question contains any stock name or company name, otherwise return 0.  


"""

N_prompt = PromptTemplate(template=com_prompt, input_variables=["q"])
name_chain = N_prompt | GPT4o_mini | JsonOutputParser()

res_prompt = """
    You are screener specialist your job is to provide info required to build sql query.YOUR JOB IS TO FIND METRIC ,SECTOR ASKED BY THE USER QUERY{query} .
    BELOW ARE THE METRICS ,TABLE NAME AND SECTOR NAMES USE ONLY THOSE AND RETURN IN JSON FORMAT.

    Table_name=yearly_ratios: [
        'MCAP', 'EV', 'PE', 'PBV', 'DIVYIELD', 'DividendPayout', 'EPS', 'BookValue',
        'ROA', 'ROE', 'ROCE', 'EBIT', 'EBITDA', 'EV_Sales', 'EV_EBITDA', 'NetIncomeMargin',
        'GrossIncomeMargin', 'AssetTurnover', 'CurrentRatio',, 'FCF_Margin',
        'Sales_TotalAsset', 'NetDebt_FCF', 'NetDebt_EBITDA', 'EBITDA_Margin', 'TotalShareHoldersEquity',
        'ShorttermDebt', 'LongtermDebt', 'SharesOutstanding', 'NetSales', 'Netprofit', 'AnnualDividend',
        'COGS', 'RetainedEarnings'
    ]
    Table_name = daily_ratios:[
    "MCAP", "EV", "PE", "PBV", "DIVYIELD", "EPS", "BookValue",
    "ROA_TTM", "ROE_TTM", "ROCE_TTM", "EBIT_TTM", "EBITDA_TTM", "EV_Sales_TTM",
    "EV_EBITDA_TTM", "NetIncomeMargin_TTM", "GrossIncomeMargin_TTM", "AssetTurnover_TTM",
    "CurrentRatio_TTM", "Sales_TotalAssets_TTM", "NetDebt_EBITDA_TTM",
    "EBITDA_Margin_TTM", "TotalShareHoldersEquity_TTM", "ShorttermDebt_TTM",
    "LongtermDebt_TTM", "SharesOutstanding", "EPSDiluted", "NetSales", "Netprofit",
    "AnnualDividend", "COGS", "PEGRatio_TTM", "DividendPayout_TTM", "Industry_PE"
]
    Table_name=balanceSheet:['Non-Current Assets:', 'Fixed Assets', '   Property, Plant and Equipments', '   Right-of-Use Assets', '    Intangible Assets', '    Intangible Assets under Development', 'Capital Work in Progress', 'Non-current Investments ', '   Investment Properties', '   Investments in Subsidiaries, Associates and Joint venture', '   Investments of Life Insurance Business', '   Investments - Long-term', 'Long-term Loans and Advances', 'Other Non-Current Assets', 'Long-term Loans and Advances and Other Non-Current Assets ', '   Biological Assets other than Bearer Plants (Non Current)', '   Loans - Long-term', '   Others Financial Assets - Long-term', '   Current Tax Assets - Long-term', '   Insurance Related Assets (Non Current)', '   Other Non-current Assets (LT)', 'Deferred Tax Assets', 'Total Non Current Assets', 'Current Assets:', 'Inventories', 'Biological Assets other than Bearer Plants (Current)', 'Current Investments', 'Cash and Cash Equivalents ', '   Cash and Cash Equivalents', '   Bank Balances Other Than Cash and Cash Equivalents', 'Trade Receivables', 'Short-term Loans and Advances', 'Other Current Assets', 'Short-term Loans and Advances and Other Current Assets ', '   Loans - Short-term', '   Others Financial Assets - Short-term', '   Current Tax Assets - Short-term', '   Insurance Related Assets (Current)', '   Other Current Assets (ST)', '   Assets Classified as Held for Sale', 'Total Current Assets', 'TOTAL ASSETS', 'Current Liabilities:', 'Short term Borrowings', 'Lease Liabilities (Current)', 'Trade Payables', 'Other Current Liabilities ', '   Others Financial Liabilities - Short-term', '   Insurance Related Liabilities (Current)', '   Other Current Liabilities', '   Liabilities Directly Associated with Assets Classified as Held for Sale', 'Provisions ', '   Current Tax Liabilities - Short-term', '   Other Short term Provisions', 'Total Current Liabilities', 'Net Current Asset', 'Non-Current Liabilities:', 'Long term Borrowings ', '   Debt Securities', '   Borrowings', '   Deposits', 'Lease Liabilities (Non Current)', 'Other Long term Liabilities ', '   Others Financial Liabilities - Long-term', '   Insurance Related Liabilities (Non Current)', '   Other Non-Current Liabilities', 'Long term Provisions ', '   Current Tax Liabilities - Long-term', '   Other Long term Provisions', 'Deferred Tax Liabilities', 'Total Non Current Liabilities', 'Shareholders’ Funds:', 'Share Capital ', '   Equity Capital', '   Preference Capital', '   Unclassified Capital', 'Other Equity ', '   Reserves and Surplus', '   Other Equity Components', "Total Shareholder's Fund", 'Total Equity', 'TOTAL EQUITY AND LIABILITIES', 'Contingent Liabilities and Commitments (to the Extent Not Provided for)', 'Ordinary Shares :', 'Authorised:', 'Number of Equity Shares - Authorised', 'Amount of Equity Shares - Authorised', 'Par Value of Authorised Shares', 'Susbcribed & fully Paid up :', 'Par Value', 'Susbcribed & fully Paid up Shares', 'Susbcribed & fully Paid up CapItal']

    Table_name=cashflow:['Cash and Cash Equivalents at Beginning of the year', None, 'Cash Flow From Operating Activities', None, 'Net Profit before Tax & Extraordinary Items', 'Adjustments : ', 'Depreciation', 'Interest (Net)', 'Dividend Received 1', 'P/L on Sales of Assets', 'P/L on Sales of Invest', 'Prov. & W/O (Net)', 'P/L in Forex', 'Fin. Lease & Rental Charges', 'Others 1', 'Total Adjustments (PBT & Extraordinary Items)', 'Operating Profit before Working Capital Changes', 'Adjustments : ', 'Trade & 0ther Receivables', 'Inventories', 'Trade Payables', 'Loans & Advances', 'Investments', 'Net Stock on Hire', 'Leased Assets Net of Sale', 'Trade Bill(s) Purchased', 'Change in Borrowing', 'Change in Deposits', 'Others 2', 'Total Adjustments (OP before Working Capital Changes)', 'Cash Generated from/(used in) Operations', 'Adjustments : ', 'Interest Paid(Net)', 'Direct Taxes Paid', 'Advance Tax Paid', 'Others 3', 'Total Adjustments(Cash Generated from/(used in) Operations', 'Cash Flow before Extraordinary Items', 'Extraordinary Items :', 'Excess Depreciation W/b', 'Premium on Lease of land', 'Payment Towards VRS', "Prior Year's Taxation", 'Gain on Forex Exch. Transactions', 'Others 4', 'Total Extraordinary Items', 'Net Cash from Operating Activities', None, 'Cash Flow from Investing Activities', None, 'Investment in Assets :', 'Purchased of Fixed Assets', 'Capital Expenditure', 'Sale of Fixed Assets', 'Capital WIP', 'Capital Subsidy Received', 'Financial / Capital Investment :', 'Purchase of Investments', 'Sale of Investments', 'Investment Income', 'Interest Received', 'Dividend Received 2', 'Invest.In Subsidiaires', 'Loans to Subsidiaires', 'Investment in Group Cos.', 'Issue of Shares on Acquisition of Cos.', 'Cancellation of Investment in Cos. Acquired', 'Acquisition of Companies', 'Inter Corporate Deposits', 'Others 5', 'Net Cash used in Investing Activities', None, 'Cash Flow From Financing Activities', None, 'Proceeds :', 'Proceeds from Issue of shares (incl. share premium)', 'Proceed from Issue of Debentures', 'Proceed from 0ther Long Term Borrowings', 'Proceed from Bank Borrowings', 'Proceed from Short Tem Borrowings', 'Proceed from Deposits', 'Share Application Money', 'Cash/Capital Investment Subsidy', 'Loans from a Corporate Body', 'Payments :', 'Share Application Money Refund', 'On Redemption of Debenture', 'Of the Long Tem Borrowings', 'Of the Short Term Borrowings', 'Of Financial Liabilities', 'Dividend Paid', 'Shelter Assistance Reserve', 'Interest Paid in Financing Activities', 'Others 6', 'Net Cash used in Financing Activities', None, 'Net Inc./(Dec.) in Cash and Cash Equivalent', 'Cash and Cash Equivalents at End of the year']
    Table_name=yearlyresults or quaterlyresults(use this table when user asks for quaterly data):['Gross Sales/Income from operations', 'Less: Excise duty', 'Net Sales/Income from operations', 'Other Operating Income', 'Total Income from operations (net)', 'Total Expenses', '    Cost of Sales', '    Employee Cost', '    Depreciation, amortization and depletion expense', '    Provisions & Write Offs', '    Administrative and Selling Expenses', '    Other Expenses', '    Pre Operation Expenses Capitalised', 'Profit from operations before other income, finance costs and exceptional items', 'Other Income', 'Profit from ordinary activities before finance costs and exceptional items', 'Finance Costs', 'Profit from ordinary activities after finance costs but before exceptional items', 'Exceptional Items', 'Other Adjustments Before Tax', 'Profit from ordinary activities before tax', 'Total Tax', 'Net profit from Ordinary Activities After Tax', 'Profit / (Loss) from Discontinued Operations', 'Net profit from Ordinary Activities/Discontinued Operations After Tax', 'Extraordinary items', 'Other Adjustments After Tax', 'Net Profit after tax for the Period', 'Other Comprehensive Income', 'Total Comprehensive Income', 'Equity', 'Reserve & Surplus', 'Face Value', 'EPS:', '    EPS before Exceptional/Extraordinary items-Basic', '    EPS before Exceptional/Extraordinary items-Diluted', '    EPS after Exceptional/Extraordinary items-Basic', '    EPS after Exceptional/Extraordinary items-Diluted', 'Book Value (Unit Curr.)', 'No. of Employees', 'Debt Service Coverage Ratio', 'Interest Service Coverage Ratio', 'Debenture Redemption Reserve (Rs cr)', 'Paid up Debt Capital (Rs cr)']
    Table_name=pnlstatement:['Revenue From Operations ', '   Sale of Products', '   Sale of Services', '   Income from Investment and Financial Services', '   Income from Insurance Operations', '   Other Operating Revenue', 'Less: Excise Duty / GST', 'Revenue From Operations - Net', 'Other Income', 'Total Revenue', 'Changes in Inventories', 'Cost of Material Consumed', 'Internally Manufactured Intermediates Consumed', 'Purchases of Stock-in-Trade', 'Employee Benefits', 'Total Other Expenses ', '   Manufacturing / Operating Expenses', '   Administrative and Selling Expenses', '   Other Expenses', 'Finance Costs', 'Depreciation and Amortization', 'Total Expenses', 'Profit Before Exceptional Items and Tax', 'Exceptional Items Before Tax', 'Profit Before Extraordinary Items and Tax', 'Extraordinary Items Before Tax', 'Other Adjustments Before Tax', 'Profit Before Tax', 'Taxation', '   Current Tax', '   MAT Credit Entitlement', '   Deferred Tax', '   Other Tax', '   Adjust for Previous Year', 'Profit After Tax', 'Extraordinary Items After Tax', 'Discontinued Operations After Tax', '  Profit / (Loss) from Discontinuing Operations', '  Tax Expenses of Discontinuing Operations', 'Profit Attributable to Shareholders', 'Adjustments to Net Income', 'Preference Dividend', 'Profit Attributable to Equity Shareholders', 'Earning Per Share - Basic', 'Earning Per Share - Diluted', 'Operation Profit before Depreciation', 'Operating Profit after Depreciation', 'Dividend Per Share', 'Dividend Percentage', 'Equity Dividend', 'Weighted Average Shares - Basic', 'Weighted Average Shares - Diluted']
    Table_name=margin_ratios:['PBIDTIM', 'EBITM', 'PreTaxMargin', 'PATM', 'CPM']
    Table_name=performance_ratios:['ROA', 'ROE']
    Table_name=efficiency_ratios:['FixedCapitals_Sales', 'ReceivableDays', 'InventoryDays', 'PayableDays']
    Table_name=financial_stability_ratios: ['TotalDebt_Equity', 'CurrentRatio', 'QuickRatio', 'InterestCover', 'TotalDebt_MCap']
    Table_name=valuation_ratios:['pe', 'Price_BookValue', 'EV_EBITDA', 'Mcap_Sales']
    Table_name=cashflow_ratios:['CashFlowPerShare', 'PricetoCashFlowRatio', 'FreeCashFlowperShare', 'PricetoFreeCashFlow','FreeCashFlowYield', 'Salestocashflowratio']
    Table_name=liquidity_ratios:['Loans_to_Deposits', 'Cash_to_Deposits','Investment_toDeposits','IncLoan_to_Deposit','Credit_to_Deposits','InterestExpended_to_Interestearned','Interestincome_to_Totalfunds', 'InterestExpended_to_Totalfunds']
    Table_name=return_ratios:['Return_ROE','Return_ROE_NetProfit','Return_ROE_Networth','Return_ROCE','Return_ROCE_EBIT','Return_ROCE_CapitalEmployed','Return_ReturnOnAssets','Return_ROA_NetProfit','Return_ROA_TotalFixedAssets']
    Table_name=growth_ratios:['NetSalesGrowth','EBITDAGrowth','EBITGrowth','PATGrowth','EPS']
    Table_name =solvency_ratios:[Solvency_TotalDebtToEquityRatio", "Solvency_TotalDebtToEquityRatio_TotalDebt", "Solvency_TotalDebtToEquityRatio_Networth", "Solvency_InterestCoverageRatio", "Solvency_InterestCoverageRatio_EBIT", "Solvency_InterestCoverageRatio_InterestPayments", "Solvency_CurrentRatio", "Solvency_CurrentRatio_CurrentAsset", "Solvency_CurrentRatio_CurrentLiabilities"]
    
    Table_name=shareholdingpattern: ['TotalPromoter_Shares','TotalPromoter_PerShares','TotalPromoter_PledgeShares','TotalPromoter_PerPledgeShares','TotalNonPromoter_Shares','TotalNonPromoter_PerShares','Total_Promoter_NonPromoter_Shares',
                                'Total_Promoter_NonPromoter_PerShares','Total_Promoter_NonPromoter_PledgeShares','Total_Promoter_NonPromoter_PerPledgeShares','Total_NoofShareholders',
                                'Promoters_Holding','FIIs_Holding','DIIs_Holding','Public_Holding']
   
sector_names= [
    "Aerospace & Defence", "Agro Chemicals", "Air Transport Service", "Alcoholic Beverages",
    "Auto Ancillaries", "Automobile", "Banks", "Bearings", "Cables",
    "Capital Goods - Electrical Equipment", "Capital Goods-Non Electrical Equipment",
    "Castings, Forgings & Fastners", "Cement", "Cement - Products", "Ceramic Products",
    "Chemicals", "Computer Education", "Construction", "Consumer Durables",
    "Credit Rating Agencies", "Crude Oil & Natural Gas", "Diamond, Gems and Jewellery",
    "Diversified", "Dry cells", "E-Commerce/App based Aggregator", "Edible Oil", "Education",
    "Electronics", "Engineering", "Entertainment", "ETF", "Ferro Alloys", "Fertilizers",
    "Finance", "Financial Services", "FMCG", "Gas Distribution", "Glass & Glass Products",
    "Healthcare", "Hotels & Restaurants", "Infrastructure Developers & Operators",
    "Infrastructure Investment Trusts", "Insurance", "IT - Hardware", "IT - Software",
    "Leather", "Logistics", "Marine Port & Services", "Media - Print/Television/Radio",
    "Mining & Mineral products", "Miscellaneous", "Non Ferrous Metals", "Oil Drill/Allied",
    "Packaging", "Paints/Varnish", "Paper", "Petrochemicals", "Pharmaceuticals",
    "Plantation & Plantation Products", "Plastic products", "Plywood Boards/Laminates",
    "Power Generation & Distribution", "Power Infrastructure", "Printing & Stationery",
    "Quick Service Restaurant", "Railways", "Readymade Garments/ Apparells",
    "Real Estate Investment Trusts", "Realty", "Refineries", "Refractories", "Retail",
    "Ship Building", "Shipping", "Steel", "Stock/ Commodity Brokers", "Sugar",
    "Telecom-Handsets/Mobile", "Telecomm Equipment & Infra Services", "Telecomm-Service",
    "Textiles", "Tobacco Products", "Trading", "Tyres"
]

    Rules for year:
    1.If user asking about quaterly results use quaterlyresults table or metrics are from share holding pattern.
        current quarter {qr}:quarter mentioned in {query},if its q4 return 03 ,q3 as 12,q2 as 09,q1 as 06.
        append above quarter to year user mentioned or append quarter to {year}.if qaurter is q4 increment year by 1 year.
        year should be YYYYQQ.
    2.Else:
        year mentioned in user query: {query} and append '03' return YYYY03 if no year mentioned append '03' to {year}.

    Rules for Table_name:
    1.If user asking about quaterly data from yearlyresults table .set quaterlyresults as table name.
    2.Else keep table name in which metric belongs to.

    The output should be in json format:

    "data":
        "sector":Name of the sectors mentioned in the {query}.If no sector mentioned in {query} return 1,take only sector from sectors_name list which is provided.
        "metrics": list of metrics present in the prompt if there are no metrics mentioned in query set metrics as empty.
            "Metric Name": Name of the metric ,return exact metric from above as it is with spaces.
            "Table_name": follow rules for table_name.
            "condition": whether it is >(greater than),<(less than),>=(greater than or equal to),<=(less than or equal to),=(equal to),
            "value": value associated to that metric .consider only number dont consider symbols and units and dont convert them into full figures.
        "year": If metric belongs to bonus issues or splits tables don't append '03' return year in YYYY format else follow rules for year.
        "type":check whether user query: {query} is asking about consolidated or standard .If standard return type as 'S' if consolidated return type as 'C' ,if query doesnot both return C.if metric is from Shareholdingpattern table set type as 1.

   
    """

R_prompt = PromptTemplate(template=res_prompt, input_variables=["query","year"])
llm_chain_res= LLMChain(prompt=R_prompt, llm=llm_screener)
chain = R_prompt | llm_screener | JsonOutputParser()


sql_prompt = """
    You need to generate a SQL query based on specific criteria provided and user query. The input will include:
    This is the input data {data} and this is the user query : {query}
    The sector of companies you want to query.If sector is 1 dont include in sql query.
    IF type is 1 dont include that in sql query.
    Metrics criteria: a list of metrics along with conditions (>, <, =) and values.
    The year for which you want to retrieve data.If user asking for daily ratios dont include year logic in sql query.
    Your task is to create a SQL query template that can be filled in with the provided inputs to generate the final SQL query.
    Try cast each metric to DOUBLE
    

    ### Example Queries and Guidelines:

    **The query should return the company name, NSE symbol, metrics, and type, and it should cast each metric to DOUBLE where applicable.**

    1. **Metrics from Yearly Ratios, Margin Ratios, Financial Stability Ratios, Solvency Ratios, Performance Ratios, Efficiency Ratios, Valuation Ratios, Cashflow Ratios, Liquidity Ratios, Return Ratios, Growth Ratios, and Solvency Ratios Tables:**

    - Example Query: list companies having pe ratio greater than 100 or ROA greater than 20 in 2023 in banking sector
    - Example SQL:
        SELECT DISTINCT cm.companyname, COALESCE(NULLIF(TRIM(cm.nsesymbol), ''), cm.bsecode) AS symbol ,yr.PE,yr.ROA,yr.type
        FROM company_master cm
        JOIN (
            SELECT CO_CODE,PE,ROA,type
            FROM yearly_ratios
            WHERE try_cast(PE AS DOUBLE) > 100 and YRC ='202303' OR try_cast(ROA AS DOUBLE) > 20 and YRC ='202303'
        ) yr
        ON cm.co_code = yr.CO_CODE    
        where cm.sectorname ='FMCG' and yr.type='C';

    - Example Query: give me list of companies whose pat growth is greater than 30  and mcap sales greater than 10 in 2023(metrics belongs to different table)
    - Example SQL:
        SELECT 
            cm.companyname, 
            COALESCE(NULLIF(TRIM(cm.nsesymbol), ''), cm.bsecode) AS symbol
            yr.PATGrowth, 
            vr.Mcap_Sales, 
            yr.type
        FROM 
            company_master cm
        JOIN (
            SELECT 
                CO_CODE, 
                PATGrowth, 
                type
            FROM 
                growth_ratios
            WHERE 
                try_cast(PATGrowth AS DOUBLE) > 30 
                AND YRC = '202303'
        ) yr 
        ON 
            cm.co_code = yr.CO_CODE
        JOIN (
            SELECT 
                CO_CODE, 
                Mcap_Sales, 
                type
            FROM 
                valuation_ratios
            WHERE 
                try_cast(Mcap_Sales AS DOUBLE) > 10 
                AND YRC = '202303'
        ) vr 
        ON 
            cm.co_code = vr.CO_CODE 
            AND yr.type = vr.type
        WHERE 
            yr.type = 'C';


    2. **Metrics from Balance Sheet, PnL Statement, Cashflow, Yearly Results Tables:**
    - Example Query: list all compnaies having total reveneue greater than 2500crs in banking sector.
    - Example SQL:
        SELECT DISTINCT 
        COALESCE(NULLIF(TRIM(cm.nsesymbol), ''), cm.bsecode) AS symbol 
        cm.nsesymbol, 
        pnl.Total_Revenue,
        pnl.type
        FROM 
            company_master cm 
        JOIN 
            ( 
                SELECT 
                    COCODE, 
                    MAX(CASE WHEN columnname = 'Total Revenue' THEN TRY_CAST(Y202303 AS DOUBLE) END) AS Total_Revenue,
                    type
                FROM 
                    pnLStatement 
                WHERE 
                    columnname = 'Total Revenue'
                GROUP BY 
                    COCODE, type
            ) AS pnl 
        ON 
            cm.CO_CODE = pnl.COCODE
        WHERE 
            pnl.Total_Revenue > 25000 
            AND cm.sectorname = 'Banks'
            AND pnl.type = 'C';
    
        

    3. **Metrics from Shareholding Pattern Table:**
   - Example Query: List companies with FIIs holding greater than 10% in 2023.
   - Example SQL:
    SELECT DISTINCT cm.companyname, COALESCE(NULLIF(TRIM(cm.nsesymbol), ''), cm.bsecode) AS symbol, sp.FIIs_Holding FROM company_master cm JOIN ( SELECT CO_CODE, FIIs_Holding FROM shareholdingpattern WHERE try_cast(FIIs_Holding AS DOUBLE) > 10 AND YRC = '202403' ) sp ON cm.co_code = sp.CO_CODE"
    
    Return only the sql query no need code 
    The output should be in json format:

    "query":generated query

    """


s_prompt = PromptTemplate(template=sql_prompt, input_variables=["data","query"])
llm_chain_sql= LLMChain(prompt=s_prompt, llm=llm_screener)
chain_s = s_prompt | llm_screener | JsonOutputParser()


def get_query(q):
    #with get_openai_callback() as cb:
    #q="give me list of companies whose mcap sales is greater than 300 and less than 500 in banking sector  "
    # q="give me list of companies whose reveune from operations is 565347 "
        #yr=get_completed_financial_year()
        # Example usage
    yr,qr=get_completed_financial_year_and_quarter()

    input_data = {
        "query": q,
        "year":yr,
        "qr":qr
    }
    #res=llm_chain_res.predict(query=q)
    res=chain.invoke(input_data)
    print(res)
    if len(res['data']['metrics'])==0:
        # return {"Response": "No Similar metrics found ",
        # "Total_Tokens": cb.total_tokens,
        # "Prompt_Tokens": cb.prompt_tokens,
        # "Completion_Tokens": cb.completion_tokens,
        # # "Total Cost (USD)": cb.total_cost}
        # }
        return None,None
                    

    sql_data={"data":res['data'],"query":q}

    sql=chain_s.invoke(sql_data)
    # print(res['data'])
    # print(f"Total Tokens: {cb.total_tokens}")
    # print(f"Prompt Tokens: {cb.prompt_tokens}")
    # print(f"Completion Tokens: {cb.completion_tokens}")
    # print(f"Total Cost (USD): ${cb.total_cost}")

    return sql,res


def final_query(q):
    sql,res=get_query(q)
    print(sql)
    if sql:
        query1=sql['query']
        # sec=res['data']['sector']
        # print(sec)
        # if sec == 1:
        #     pass
        # else:
        #     pass
            #query1 += f" WHERE cm.sectorname = '{sec}';"

        return query1
    else:
        return None

def validate(q):
    input_data = {
    "q":q
    }
    #res=llm_chain_res.predict(query=q)
    res=name_chain.invoke(input_data)
    return res['valid']


def screen_stocks(query, session_id,db):
    
    #similar_question = query_similar_questions(query,'screener_cache')
    #print(f"Similar Question: {similar_question}")
    similar_question=None
    if similar_question:
        cached_response = db.query(ChatbotResponse).filter(ChatbotResponse.question == similar_question).first()
        print(f"Cached Response: {cached_response}")

        if similar_question and cached_response:
            # looks up for the corresponding response for
            cached_response.count += 1
            db.commit()
            #add_to_memory(input_data.input_text, cached_response.response, session_id=session_id)
            return {
                "Response": cached_response.response,
                #"Popularity": cached_response.count,
                "Total_Tokens": cached_response.Total_Tokens,
                "Prompt_Tokens": cached_response.Prompt_Tokens,
                "Completion_Tokens": cached_response.Completion_Tokens,
            }
    else:
        with get_openai_callback() as cb:
            v=validate(query)
            if v==1:
                #return "The query who are asking does not met requrements for screener try asking in fundamental section"
                return {"Response": "This query does not align with the requirements for the screener. Try asking it in the fundamental section instead",
                        "Total_Tokens": cb.total_tokens,
                        "Prompt_Tokens": cb.prompt_tokens,
                        "Completion_Tokens": cb.completion_tokens,
                        # "Total Cost (USD)": cb.total_cost}
                        }
            else:
                q=final_query(query)
                if q:
                    database = 'athenadb'
                    response = execute_athena_query(q,database, S3_OUTPUT_LOCATION)
                    query_execution_id = response['QueryExecutionId']
                    result_response = get_query_results(query_execution_id)
                    #print(result_response)

                    #print(response)
                    # Specify the bucket name and CSV file key
                    bucket_name = s_bucket
                    file_key = f'output-data/{query_execution_id}.csv'
                    #print(file_key)
                    #file_key='output-data/152f5966-90f1-4af7-b758-d14708d686ff.csv'

                    # Download the file from S3 into memory
                    if result_response['QueryExecution']['Status']['State'] == 'SUCCEEDED':
                        try:
                            response = s3_client.get_object(Bucket=bucket_name, Key=file_key)
                            #print(response)
                            csv_content = response['Body'].read().decode('utf-8')
                            df = pd.read_csv(io.StringIO(csv_content))
                            #print(csv_content)
                            if df.empty:
                                #print("table")
                                #df = pd.read_csv(io.StringIO(csv_content))
                                #table=df.to_html().replace('\n', '')

                                #return table
                                return {"Response": "No stocks were found based on your input criteria. Please adjust your requirements and try again.",
                                        "Total_Tokens": cb.total_tokens,
                                        "Prompt_Tokens": cb.prompt_tokens,
                                        "Completion_Tokens": cb.completion_tokens,
                                        # "Total Cost (USD)": cb.total_cost}
                                        }
                            
                            # Use Pandas to read the CSV content into a DataFrame
                            #df = pd.read_csv(io.StringIO(csv_content))
                            #print(type(df))
                            else:
                                table=df.to_html().replace('\n', '')
                                new_response = ChatbotResponse(
                                    question=query,
                                    response=table,
                                    count=1,
                                    Total_Tokens=cb.total_tokens,  # Populate the new columns
                                    Prompt_Tokens=cb.prompt_tokens,
                                    Completion_Tokens=cb.completion_tokens,
                                )
                                #db.add(new_response)
                                #db.commit()
                                #add_questions(query,'screener_cache')
                                
                                #print("no table")
                                return {"Response": table,
                                    "Total_Tokens": cb.total_tokens,
                                    "Prompt_Tokens": cb.prompt_tokens,
                                    "Completion_Tokens": cb.completion_tokens,
                                    # "Total Cost (USD)": cb.total_cost}
                                    }
                            #df = pd.read_csv(io.StringIO(csv_content)


                        except s3_client.exceptions.NoSuchKey:
                            print(f"The specified key does not exist: {file_key}")


                    if result_response['QueryExecution']['Status']['State'] == 'FAILED':
                        return {"Response": "No stocks were found based on your input criteria. Please adjust your requirements and try again.",
                                    "Total_Tokens": cb.total_tokens,
                                    "Prompt_Tokens": cb.prompt_tokens,
                                    "Completion_Tokens": cb.completion_tokens,
                                    # "Total Cost (USD)": cb.total_cost}
                                    }
                else:
                    return {"Response": "The query lacks any financial metrics. Please revise the query to include a financial metric for screening",
                        "Total_Tokens": cb.total_tokens,
                        "Prompt_Tokens": cb.prompt_tokens,
                        "Completion_Tokens": cb.completion_tokens,
                        # "Total Cost (USD)": cb.total_cost}
                        }




# class queryCheckInput(BaseModel):
#     stockname: query = Field(..., description="return complete query user mentioned")
#     # year: str = Field(..., description="year mentioned in the query , if no year mentioned keep 2023")
#     # prompt: str = Field(..., description="give me complete prompt user asked")
    
# class screener(BaseTool):
#     name = "stock_screener"
#     description = "use this tool for stock screening functionality"

#     def _run(self, stockname: str):
#         print("i'm running")
#         print(query)
#         headlines_response = get_ans(query)

#         return headlines_response

#     def _arun(self, stockname: str):
#         raise NotImplementedError("This tool does not support async")

#     args_schema: Optional[Type[BaseModel]] = queryCheckInput
# print(screen_stocks('fmcg stocks having roe greater than 10',"qwert"))

# import streamlit as st

# st.title('screener')

# # Take text input from the user
# user_input = st.text_input('Enter prompt:')


# # Display the input back to the user
# if user_input:
#     res=screen_stocks(user_input,'123')
#     st.write(res)

# res=screen_stocks("list all stocks having pe greater than 5",'123')
# print(res)
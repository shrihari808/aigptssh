from dotenv import load_dotenv
from langchain.schema import HumanMessage, AIMessage, ChatMessage
import os
import json
import requests
import streamlit as st
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.memory import ConversationSummaryBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from datetime import datetime
import os
import pandas as pd
from fuzzywuzzy import process
import json
from langchain.agents import AgentExecutor
from langchain.agents import AgentType, initialize_agent, load_tools
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
import json
import openai
from openai import OpenAI
from fastapi import FastAPI, HTTPException,APIRouter
from pydantic import BaseModel
from dotenv import load_dotenv
from starlette.status import HTTP_403_FORBIDDEN
from fastapi import FastAPI, HTTPException,Depends, Header,Query


load_dotenv(override=True)

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.getenv("OPENAI_API_KEY"),
)



token=os.getenv("CMOTS_BEARER_TOKEN")

def find_stock_code(stock_name, threshold=90):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(r"C:\Users\nithi\Desktop\arr\newbranch\ai-gpt\app_service\csvdata\company_codes2.csv")

    # Extract the company names and codes from the DataFrame
    company_names = df["Company Name"].tolist()
    company_codes = df["Company Code"].tolist()

    # Use fuzzy matching to find the closest match to the input stock name
    match = process.extractOne(stock_name, company_names)

    if match and match[1] >= threshold:
        # Get the index of the closest match
        idx = company_names.index(match[0])
        # Return the corresponding stock code
        return company_codes[idx]
    else:
        return 0

    

def find_metric(metric, threshold=95):
    # Load the CSV file into a DataFrame
    df = pd.read_csv(r"C:\Users\nithi\Desktop\arr\newbranch\ai-gpt\app_service\csvdata\metrics1.csv")

    # Extract the metric names and descriptions from the DataFrame
    metric_name = df["metric_name"].tolist()
    metric_desc = df["metric_desc"].tolist()

    # Use fuzzy matching to find the closest match to the input metric name
    match = process.extractOne(metric, metric_name)

    if match and match[1] >= threshold:
        # Get the index of the closest match
        idx = metric_name.index(match[0])
        # Return the corresponding metric description
        return metric_desc[idx]
    else:
        return "metric not found"



async def get_year_ratio(stock_name, years, metric):
    # print(years)
    code = int(find_stock_code(stock_name))
    print(code)
    if code==1:
        return "stock not in nifty 50"
    # metric1 = get_metric(metric)
    metric1 = find_metric(metric)
    # print(code)
    # code=476
    url = f"http://airrchipapis.cmots.com/api/YearlyRatio/{code}/S"
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Print the response content
        data = response.json()
        if data['data']:
            financial_data = data['data'][:4]

            modified_financial_data = [
                {k: f"{float(v)}" if isinstance(v, (float, float)) and k != 'YearEnd' else v for k, v in entry.items()
                 if k not in ['CO_CODE']} for entry in financial_data]

            # Create a dictionary with modified 'YRC' values as keys
            result_dict = {str(int(entry['YearEnd'])): {k: v for k, v in entry.items() if k != 'YearEnd'} for entry in
                           modified_financial_data}
            updated_result_dict = {
                f"{entry[:4]}": values
                for entry, values in result_dict.items()
            }

            if years[0] == 1:
                output_dict = {
                    'stock_name': stock_name,
                    'metric': metric1,
                    'years': list(updated_result_dict.keys()),
                    'values': {updated_result_dict[year][metric1] for year in updated_result_dict},
                    'chart_type': 'line'
                }

                return output_dict
            else:
                output_dict = {
                    'stock_name': stock_name,
                    'metric': metric1,
                    'years': list(years),
                    'values': [updated_result_dict[year][metric1] for year in years],
                    'chart_type': 'bar'
                }

                return output_dict

                # return updated_result_dict[year][metric1]

    else:
        # Print an error message if the request was not successful
        print(f"Error: {response.status_code}")
        print(response.text)


async def years_result(stock_name, years, metric):
    # print(year)
    code = int(find_stock_code(stock_name))
    # metric1 = get_metric(metric)
    metric1 = find_metric(metric)
    url = f"http://airrchipapis.cmots.com/api/Yearly-Results/{code}/s"
    # description = "93.0"  # Replace with the actual description

    # Set up the headers with the authorization token
    headers = {"Authorization": f"Bearer {token}"}

    # Set up the payload with the description
    # data = {"description": description}

    # Make a GET request with headers and payload
    async with httpx.AsyncClient() as client:
        response = await client.get(url, headers=headers)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        data = response.json()
        print(data)
        yearwise_data = {}
        financial_data = data['data']

        # Process each data dictionary
        for data_dict in financial_data:
            asset_type = data_dict['COLUMNNAME'].rstrip(':').replace(' ', '_').lstrip('_') # Replace spaces with underscores

            # Create a new dictionary with keys replaced
            new_data_dict = {
                key.replace(' ', '_') if isinstance(key, str) else key: value
                for key, value in data_dict.items()
            }
            
            for key, value in new_data_dict.items():
                if key.startswith('Y20'):
                    # Extracting the year from the key
                    year = key[1:5]

                    # Creating year entry if it doesn't exist
                    if year not in yearwise_data:
                        yearwise_data[year] = {}

                    # Adding asset type and value for the current year
                    yearwise_data[year][asset_type] = value
        print(yearwise_data)

    if years[0] == 1:
        output_dict = {
            'stock_name': stock_name,
            'metric': metric1,
            'years': list(yearwise_data.keys()),
            'values': [yearwise_data[year][metric1] for year in years],
            'chart_type': 'line'
        }

    else:
        output_dict = {
            'stock_name': stock_name,
            'metric': metric1,
            'years': list(years),
            'values': {yearwise_data[year][metric1] for year in years},
            'chart_type': 'bar'
        }
    return output_dict


async def get_cashflow1(stock_name, years, metric):
    code = int(find_stock_code(stock_name))
    metric1 = find_metric(metric)
    
    url = f"http://airrchipapis.cmots.com/api/CashFlow/{code}/s"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
        response.raise_for_status()  # Check if the request was successful
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

    try:
        data = response.json()
    except ValueError as e:
        print(f"Failed to parse JSON: {e}")
        return None

    yearwise_data = {}
    financial_data = data.get('data', [])

    for data_dict in financial_data:
        asset_type = data_dict['COLUMNNAME']
        if not asset_type:
            continue
        asset_type = asset_type.rstrip(':').replace(' ', '_').lstrip('_')

        for key, value in data_dict.items():
            if key.startswith('Y20'):
                year = key[1:5]
                if year not in yearwise_data:
                    yearwise_data[year] = {}
                yearwise_data[year][asset_type] = value

    selected_years = [str(year) for year in years]

    if years[0] == 1:
        output_dict = {
            'stock_name': stock_name,
            'metric': metric1,
            'years': list(yearwise_data.keys()),
            'values': [yearwise_data[year][metric1] for year in yearwise_data.keys()],
            'chart_type': 'line'
        }
    else:
        output_dict = {
            'stock_name': stock_name,
            'metric': metric1,
            'years': selected_years,
            'values': [yearwise_data.get(year, {}).get(metric1, None) for year in selected_years],
            'chart_type': 'bar'
        }

    return output_dict


async def get_cashflow(stock_name, years, metric):
    code = int(find_stock_code(stock_name))
    metric1 = find_metric(metric)
    

    url = f"http://airrchipapis.cmots.com/api/CashFlow/{code}/S"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, headers=headers)
        response.raise_for_status()  # Check if the request was successful
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return None

    try:
        data = response.json()
    except ValueError as e:
        print(f"Failed to parse JSON: {e}")
        return None

    yearwise_data = {}
    financial_data = data.get('data', [])

    for data_dict in financial_data:
        if data_dict['COLUMNNAME'] is not None:
            asset_type = data_dict['COLUMNNAME'].rstrip(':')#.replace(' ', '_').lstrip('_')

            new_data_dict = {
                key.replace(' ', '_') if isinstance(key, str) else key: value
                for key, value in data_dict.items()
            }

            for key, value in new_data_dict.items():
                if key.startswith('Y20'):
                    year = key[1:5]
                    if year not in yearwise_data:
                        yearwise_data[year] = {}
                    yearwise_data[year][asset_type] = value

    selected_years = [str(year) for year in years]

    if years[0] == 1:
        output_dict = {
            'stock_name': stock_name,
            'metric': metric1,
            'years': list(yearwise_data.keys()),
            'values': [yearwise_data[year][metric1] for year in yearwise_data.keys()],
            'chart_type': 'line'
        }
    else:
        output_dict = {
            'stock_name': stock_name,
            'metric': metric1,
            'years': selected_years,
            'values': [yearwise_data[year].get(metric1, None) for year in selected_years],
            'chart_type': 'bar'
        }

    return output_dict

tools = [
    {
        "type": "function",
        "function":{
                "name": "get_year_ratio",
                "description": "Useful for when you need to get daily ,quaterly  and yearly ratios or  financial "
                              "statements/data  of  a stock." ,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "stockname": {
                            "type": "string",
                            "description": "list of stock names present in the prompt,return all stocks with comma mentioned in the prompt",
                        },
                        "year": {
                            "type": "string",
                            "description": "list of years mentioned in the query , if no year mentioned return year as 1"
                            ,
                        },
                        "metric": {
                            "type": "string",
                            "description": "list of metrics mentioned in the prompt ,if its "
                                        "EPS"
                                        "PE "
                                        "EBITDA "
                                        "ROE "
                                        "ROCE"
                                        "MCAP"
                                        "EV"
                                        "PBV"
                                        "DIVYIELD"
                                        "BookValue"
                                        "ROA"
                                        "EBIT"
                                        "EV_Sales"
                                        "EV_EBITDA"
                                        "NetIncomeMargin"
                                        "GrossIncomeMargin"
                                        "AssetTurnover"
                                        "CurrentRatio"
                                        "Debt_Equity"
                                        "Sales_TotalAssets"
                                        "NetDebt_EBITDA"
                                        "EBITDA_Margin"
                                        "ShorttermDebt"
                                        "LongtermDebt"
                                        "SharesOutstanding"
                                        "EPSDiluted"
                                        "NetSales"
                                        "Netprofit"
                                        "profit after tax"
                                        "PAT"
                                        "AnnualDividend"
                                        "COGS"
                                        "Other Adjustments After Tax"
                                        "Other Comprehensive Income"
                                        "Total Comprehensive Income"
                                        "Equity"
                                        "Gross_Sales/Income_from_operations"
                                        "Less:_Excise_duty"
                                        "Net_Sales/Income_from_operations"
                                        "Other_Operating_Income"
                                        "Total_Income_from_operations_(net)"
                                        "Total_Expenses"
                                        "____Cost_of_Sales"
                                        "____Employee_Cost"
                                        "____Depreciation,_amortization_and_depletion_expense"
                                        "____Provisions_&_Write_Offs"
                                        "____Administrative_and_Selling_Expenses"
                                        "____Other_Expenses"
                                        "____Pre_Operation_Expenses_Capitalised"
                                        "Profit_from_operations_before_other_income,_finance_costs_and_exceptional_items"
                                        "Other_Income"
                                        "Profit_from_ordinary_activities_before_finance_costs_and_exceptional_items"
                                        "Finance_Costs"
                                        "Profit_from_ordinary_activities_after_finance_costs_but_before_exceptional_items"
                                        "Exceptional_Items"
                                        "Other_Adjustments_Before_Tax"
                                        "Profit_from_ordinary_activities_before_tax"
                                        "Total_Tax"
                                        "Net_profit_from_Ordinary_Activities_After_Tax"
                                        "Profit_/_(_Loss)_from_Discontinued_Operations"
                                        "Net_profit_from_Ordinary_Activities/Discontinued_Operations_After_Tax"
                                        "Extraordinary_items"
                                        "Other_Adjustments_After_Tax"
                                        "Net_Profit_after_tax_for_the_Period"
                                        "Other_Comprehensive_Income"
                                        "Total_Comprehensive_Income"
                                        "Equity"
                                        "Reserve_&_Surplus"
                                        "Face_Value"
                                        "EPS"
                                        "EPS_before_Exceptional/Extraordinary_items-Basic"
                                        "EPS_before_Exceptional/Extraordinary_items-Diluted"
                                        "EPS_after_Exceptional/Extraordinary_items-Basic"
                                        "ßEPS_after_Exceptional/Extraordinary_items-Diluted"
                                        "Book_Value_(Unit_Curr.)"
                                        "Dividend_Per_Share(Rs.)"
                                        "Dividend_(%)"
                                        "No._of_Employees"
                                        "Debt_Equity_Ratio"
                                        "Debt_Service_Coverage_Ratio"
                                        "Interest_Service_Coverage_Ratio"
                                        "Debenture_Redemption_Reserve_(Rs_cr)"
                                        "Paid_up_Debt_Capital_(Rs_cr)"
                                        "Cash_and_Cash_Equivalents_at_Beginning_of_the_year"
                                        "Net_Profit_before_Tax_&Extraordinary_Items"
                                        "Depreciation"
                                        "Interest(Net)"
                                        "P_L_on_Sales_of_Assets"
                                        "Prov_&W_O(Net)"
                                        "P_L_in_Forex"
                                        "Total_Adjustments_(PBT_&Extraordinary_Items)"
                                        "Operating_Profit_before_Working_Capital_Changes"
                                        "Trade&0ther_Receivables"
                                        "Inventories"
                                        "Trade_Payables"
                                        "Total_Adjustments(OP_before_Working_Capital_Changes)"
                                        "Cash_Generated_from/(used_in)_Operations"
                                        "Direct_Taxes_Paid"
                                        "Total_Adjustments(Cash_Generated_from/(used_in)_Operations)"
                                        "Cash_Flow_before_Extraordinary_Items"
                                        "Net_Cash_from_Operating_Activities"
                                        "Purchased_of_Fixed_Assets"
                                        "Sale_of_Fixed_Assets"
                                        "Purchase_of_Investments"
                                        "Investments purchased"
                                        "Investments sold"
                                        "Sale_of_Investments"
                                        "Interest_Received"
                                        "Invest_In_Subsidiaires"
                                        "Loans_to_Subsidiaires"
                                        "Net_Cash_used_in_Investing_Activities"
                                        "Proceed_from_0ther_Long_Term_Borrowings"
                                        "Of_Financial_Liabilities"
                                        "Dividend_Paid"
                                        "Interest_Paid_in_Financing_Activities"
                                        "Net_Cash_used_in_Financing_Activities"
                                        "Net_Inc./(Dec.)_in_Cash_and_Cash_Equivalent"
                                        "Net Cash Flow"
                                        "Cash_and_Cash_Equivalents_at_End_of_the_year"

                                            "return all metric names mentioned in the prompt same provided above.",
                        },
                    },
                    "required": ["stockname", "year", "metric"],
                },
            }
        
    }
]

GPT_MODEL = "gpt-3.5-turbo-16k" #gpt-3.5-turbo-0613
def chat_completion_request(messages, tools=tools, tool_choice=None, model=GPT_MODEL):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
        )
        return response
    except Exception as e:
        print("Unable to generate ChatCompletion response")
        print(f"Exception: {e}")
        return e
    

def process_prompt(prompt):
    res=chat_completion_request(messages=[{"role": "user", "content": prompt}])
    op=res.choices[0].message

    metric_id = json.loads(op.tool_calls[0].function.arguments).get("metric")
    year = json.loads(op.tool_calls[0].function.arguments).get("year")
    stock_name = json.loads(op.tool_calls[0].function.arguments).get("stockname")

    metric_ = metric_id.strip().split(",")
    print(metric_)
    year_ = year.strip().split(",")
    print(year_)
    stock_ = stock_name.strip().split(",")
    print(stock_)

    metric_list = [value.strip() for value in metric_]
    year_list = [value.strip() for value in year_]
    stock_list = [value.strip() for value in stock_]

    if year_list==['1']:
        year_list=['2023','2022','2021','2020']
    else:
        year_list = [value.strip() for value in year_]



    # Print the lists to verify
    print(metric_list)
    print(year_list)
    print(stock_list)

    formatted_responses = []
    # all_metrics=[
    # 'Gross_Sales/Income_from_operations', 'Less:_Excise_duty', 'Net_Sales/Income_from_operations', 'Other_Operating_Income',
    # 'Total_Income_from_operations_(net)', 'Total_Expenses', '____Cost_of_Sales', '____Employee_Cost',
    # '____Depreciation,_amortization_and_depletion_expense', '____Provisions_&_Write_Offs', '____Administrative_and_Selling_Expenses',
    # '____Other_Expenses', '____Pre_Operation_Expenses_Capitalised', 'Profit_from_operations_before_other_income,_finance_costs_and_exceptional_items',
    # 'Other_Income', 'Profit_from_ordinary_activities_before_finance_costs_and_exceptional_items', 'Finance_Costs',
    # 'Profit_from_ordinary_activities_after_finance_costs_but_before_exceptional_items', 'Exceptional_Items', 'Other_Adjustments_Before_Tax',
    # 'Profit_from_ordinary_activities_before_tax', 'Total_Tax', 'Net_profit_from_Ordinary_Activities_After_Tax', 'Profit_/_(Loss)_from_Discontinued_Operations',
    # 'Net_profit_from_Ordinary_Activities/Discontinued_Operations_After_Tax', 'Extraordinary_items', 'Other_Adjustments_After_Tax',
    # 'Net_Profit_after_tax_for_the_Period', 'Other_Comprehensive_Income', 'Total_Comprehensive_Income', 'Equity', 'Reserve_&_Surplus',
    # 'Face_Value', '____EPS_before_Exceptional/Extraordinary_items-Basic', '____EPS_before_Exceptional/Extraordinary_items-Diluted',
    # '____EPS_after_Exceptional/Extraordinary_items-Basic', '____EPS_after_Exceptional/Extraordinary_items-Diluted', 'Book_Value_(Unit_Curr.)',
    # 'Dividend_Per_Share(Rs.)', 'Dividend_(%)','No._of_Employees', 'Debt_Equity_Ratio', 'Debt_Service_Coverage_Ratio',
    # 'Interest_Service_Coverage_Ratio', 'Debenture_Redemption_Reserve_(Rs_cr)', 'Paid_up_Debt_Capital_(Rs_cr)'
    # ]
    
    all_metrics = [
    "Gross Sales/Income from operations",
    "Less: Excise duty",
    "Net Sales/Income from operations",
    "Other Operating Income",
    "Total Income from operations (net)",
    "Total Expenses",
    "Cost of Sales",
    "Employee Cost",
    "Depreciation, amortization and depletion expense",
    "Provisions & Write Offs",
    "Administrative and Selling Expenses",
    "Other Expenses",
    "Pre Operation Expenses Capitalised",
    "Profit from operations before other income, finance costs and exceptional items",
    "Other Income",
    "Profit from ordinary activities before finance costs and exceptional items",
    "Finance Costs",
    "Profit from ordinary activities after finance costs but before exceptional items",
    "Exceptional Items",
    "Other Adjustments Before Tax",
    "Profit from ordinary activities before tax",
    "Total Tax",
    "Net profit from Ordinary Activities After Tax",
    "Profit / (Loss) from Discontinued Operations",
    "Net profit from Ordinary Activities/Discontinued Operations After Tax",
    "Extraordinary items",
    "Other Adjustments After Tax",
    "Net Profit after tax for the Period",
    "Other Comprehensive Income",
    "Total Comprehensive Income",
    "Equity",
    "Reserve & Surplus",
    "Face Value",
    "EPS:",
    "EPS before Exceptional/Extraordinary items-Basic",
    "EPS before Exceptional/Extraordinary items-Diluted",
    "EPS after Exceptional/Extraordinary items-Basic",
    "EPS after Exceptional/Extraordinary items-Diluted",
    "Book Value (Unit Curr.)",
    "Dividend Per Share(Rs.)",
    "Dividend (%)",
    "No. of Employees",
    "Debt Equity Ratio",
    "Debt Service Coverage Ratio",
    "Interest Service Coverage Ratio",
    "Debenture Redemption Reserve (Rs cr)",
    "Paid up Debt Capital (Rs cr)"
]

    balancesheet = [
    "Fixed Assets",
    "Property, Plant and Equipments",
    "Intangible Assets",
    "Non-current Investments",
    "Investments in Subsidiaries, Associates and Joint venture",
    "Investments - Long-term",
    "Long-term Loans and Advances",
    "Other Non-Current Assets",
    "Long-term Loans and Advances and Other Non-Current Assets",
    "Others Financial Assets - Long-term",
    "Current Tax Assets - Long-term",
    "Other Non-current Assets (LT)",
    "Total Non Current Assets",
    "Inventories",
    "Current Investments",
    "Cash and Cash Equivalents",
    "Cash and Cash Equivalents",
    "Bank Balances Other Than Cash and Cash Equivalents",
    "Trade Receivables",
    "Short-term Loans and Advances",
    "Other Current Assets",
    "Short-term Loans and Advances and Other Current Assets",
    "Loans - Short-term",
    "Others Financial Assets - Short-term",
    "Other Current Assets (ST)",
    "Total Current Assets",
    "TOTAL ASSETS",
    "Short term Borrowings",
    "Lease Liabilities (Current)",
    "Trade Payables",
    "Other Current Liabilities",
    "Others Financial Liabilities - Short-term",
    "Other Current Liabilities",
    "Provisions",
    "Total Current Liabilities",
    "Net Current Asset",
    "Long term Borrowings",
    "Borrowings",
    "Deposits",
    "Other Non-Current Liabilities",
    "Long term Provisions",
    "Current Tax Liabilities - Long-term",
    "Total Non Current Liabilities",
    "Shareholders’ Funds",
    "Share Capital",
    "Equity Capital",
    "Other Equity",
    "Total Equity",
    "TOTAL EQUITY AND LIABILITIES"
]
    cashflow=[
"Cash_and_Cash_Equivalents_at_Beginning_of_the_year",
"Net_Profit_before_Tax_&Extraordinary_Items",
"Depreciation",
"Interest(Net)",
"P_L_on_Sales_of_Assets",
"Prov_&W_O(Net)",
"P_L_in_Forex",
"Total_Adjustments_(PBT_&Extraordinary_Items)",
"Operating_Profit_before_Working_Capital_Changes",
"Trade&0ther_Receivables",
"Inventories",
"Trade_Payables",
"Total_Adjustments(OP_before_Working_Capital_Changes)",
"Cash_Generated_from/(used_in)_Operations",
"Direct_Taxes_Paid",
"Total_Adjustments(Cash_Generated_from/(used_in)_Operations)",
"Cash_Flow_before_Extraordinary_Items",
"Net_Cash_from_Operating_Activities",
"Purchased_of_Fixed_Assets",
"Sale_of_Fixed_Assets",
"Purchase_of_Investments",
"Sale_of_Investments",
"Interest_Received",
"Invest_In_Subsidiaires",
"Loans_to_Subsidiaires",
"Net_Cash_used_in_Investing_Activities",
"Proceed_from_0ther_Long_Term_Borrowings",
"Of_Financial_Liabilities",
"Dividend_Paid",
"Interest_Paid_in_Financing_Activities",
"Net_Cash_used_in_Financing_Activities",
"Net_Inc./(Dec.)_in_Cash_and_Cash_Equivalent",
"Net Cash Flow"
"Net Cash Flow"
"Cash_and_Cash_Equivalents_at_End_of_the_year"
]
    
    for metric in metric_list:
        metric1=find_metric(metric)
        response = []
        for stock in stock_list:
            if metric1  in all_metrics  :
                response.append(years_result(stock, year_list, metric1))
            elif metric1   in cashflow:
                response.append(get_cashflow1(stock, year_list, metric1))
            else:
                response.append(get_year_ratio(stock, year_list, metric1))

            # Create a formatted response dictionary
        formatted_response = {
            'metric_name': metric1,
            'chart_type': response[0]['chart_type'] if response else None,  # Assuming chart_type is consistent
            'xlabel': 'years',
            'ylabel': metric1,
            'data': []
        }

        # Populate data for the formatted response
        for entry in response:
            data_entry = {
                'company_name': entry['stock_name'].upper(),
                'x-axis': entry['years'],
                'y-axis': list(entry['values'])  # Convert the values to a list
            }
            formatted_response['data'].append(data_entry)

        formatted_responses.append(formatted_response)  # Append the formatted response to the list

    return formatted_responses


# def process_prompt(prompt):
#     completion = openai.ChatCompletion.create(
#         model="gpt-3.5-turbo-0613",
#         messages=[{"role": "user", "content": prompt}],
#         functions=function_descriptions,
#         function_call="auto",
#     )
#     output = completion.choices[0].message
#     print(output)
#     metric_id = json.loads(output.function_call.arguments).get("metric")
#     year = json.loads(output.function_call.arguments).get("year")
#     stock_name = json.loads(output.function_call.arguments).get("stockname")

#     metric_ = metric_id.strip().split(",")
#     year_ = year.strip().split(",")
#     stock_ = stock_name.strip().split(",")

#     metric_list = [value.strip() for value in metric_]
#     year_list = [value.strip() for value in year_]
#     stock_list = [value.strip() for value in stock_]

#     # Create a list of metrics available in get_yearly_results
#     yearly_results_metrics = [
#         "Gross Sales/Income from operations", "Less: Excise duty", "Net Sales/Income from operations",
#         "Other Operating Income", "Total Income from operations (net)", "Total Expenses",
#         "Profit from operations before other income, finance costs and exceptional items",
#         "Other Income", "Profit from ordinary activities before finance costs and exceptional items",
#         "Finance Costs", "Profit from ordinary activities after finance costs but before exceptional items",
#         "Exceptional Items", "Other Adjustments Before Tax", "Profit from ordinary activities before tax",
#         "Total Tax", "Net profit from Ordinary Activities After Tax", "Profit / (Loss) from Discontinued Operations",
#         "Net profit from Ordinary Activities/Discontinued Operations After Tax", "Extraordinary items",
#         "Other Adjustments After Tax", "Net Profit after tax for the Period", "Other Comprehensive Income",
#         "Total Comprehensive Income", "Equity", "Reserve & Surplus", "Face Value", "EPS:", "Book Value (Unit Curr.)",
#         "Dividend Per Share(Rs.)", "Dividend (%)", "No. of Employees", "Debt Equity Ratio",
#         "Debt Service Coverage Ratio", "Interest Service Coverage Ratio", "Debenture Redemption Reserve (Rs cr)",
#         "Paid up Debt Capital (Rs cr)"
#     ]

#     formatted_responses = []

#     for metric in metric_list:
#         response = []
#         if metric in yearly_results_metrics:
#             # Call get_yearly_results for this metric
#             for stock in stock_list:
#                 if year_list:
#                     response.append(get_yearly_results(stock, year_list[0]))
#                 else:
#                     response.append(get_yearly_results(stock))
#         else:
#             # Call get_year_ratio for other metrics
#             for stock in stock_list:
#                 response.append(get_year_ratio(stock, year_list, metric))

#         formatted_response = {
#             'metric_name': metric,  # Using the current metric in the loop
#             'chart_type': response[0]['chart_type'],  # Assuming the chart type is the same for all entries
#             'xlabel': 'years',
#             'ylabel': metric,  # Using the current metric in the loop
#             'data': []
#         }

#         for entry in response:
#             data_entry = {
#                 'company_name': entry['stock_name'].upper(),
#                 'x-axis': entry['years'],
#                 'y-axis': list(entry['values'])  # Convert the values to a list
#             }
#             formatted_response['data'].append(data_entry)

#         formatted_responses.append(formatted_response)

#     return formatted_responses

# st.title("GRAPH- GPT")
# prompt = st.text_input("Enter your prompt")

# if prompt:
#     # fs = prompt1.format_messages(input=prompt, format_instruction=format_instruction)
#     response = process_prompt(prompt)
#     # response = agent_executor.invoke({"input": prompt})
#     # print(response)
#     st.write(response)
#     # output = output_parser.parse(response)
#     # st.write(output)




router=APIRouter()
class PromptRequest(BaseModel):
    prompt: str


AI_KEY=os.getenv('AI_KEY')

async def authenticate_ai_key(x_api_key: str = Header(...)):
    if x_api_key != AI_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )
    
    
@router.post("/graph_api")
async def process_prompt_endpoint(prompt_request: PromptRequest,ai_key_auth: str = Depends(authenticate_ai_key)):
    try:
        prompt = prompt_request.prompt
        #prompt="mcap of relinace"
        response = process_prompt(prompt)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# def get_graph(prompt):
#     try:
#         response = process_prompt(prompt)
#         return response
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# get_graph("mcap of reliance")
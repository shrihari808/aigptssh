from fastapi import FastAPI, HTTPException, APIRouter
from pydantic import BaseModel
import tempfile
from pytube import YouTube
from googleapiclient.discovery import build
import os
import re
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from langchain import PromptTemplate, LLMChain
from langchain.output_parsers import StructuredOutputParser, ResponseSchema 
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.output_parsers import CommaSeparatedListOutputParser
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import YouTube
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.callbacks import get_openai_callback
from dotenv import load_dotenv

from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_community.document_loaders import YoutubeLoader

from starlette.status import HTTP_403_FORBIDDEN
from fastapi import FastAPI, HTTPException,Depends, Header,Query

from config import llm_screener
from streaming.yt_stream import get_video_length

load_dotenv(override=True)

router=APIRouter()


AI_KEY=os.getenv('AI_KEY')
yt_api_key=os.getenv('youtube_api_key')

async def authenticate_ai_key(x_api_key: str = Header(...)):
    if x_api_key != AI_KEY:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN,
            detail="Invalid or missing API Key",
        )

def get_video_metadata(video_url):
    try:
        # Create a YouTube object
        yt = YouTube(video_url)

        # Access video metadata
        title = yt.title
        # video_id = yt.video_id
        # iews = yt.views
        length = yt.length
        # author = yt.author
        # publish_date = yt.publish_date

        # Print or return the metadata
        print("Title:", title)
        # print("Video ID:", video_id)
        # print("Views:", views)
        print("Length:", length, "seconds")
        # print("Author:", author)
        # print("Publish Date:", publish_date)

        # You can return the metadata as a dictionary or any other desired format
        metadata = {
            "title": title,
            # "video_id": video_id,
            # "views": views,
            # "length": length,
            # "author": author,
            # "publish_date": publish_date
        }

        return metadata

    except Exception as e:
        print("An error occurred:", e)
        return None


def get_youtube_transcript(video_id):
    rapidapi_key=os.getenv('rapid_key')
    #print(video_id)
    try:
        # Step 1: Get video details to fetch the subtitles URL
        details_url = "https://youtube-media-downloader.p.rapidapi.com/v2/video/details"
        querystring_details = {"videoId": video_id}

        headers = {
            "x-rapidapi-key": rapidapi_key,
            "x-rapidapi-host": "youtube-media-downloader.p.rapidapi.com"
        }

        response = requests.get(details_url, headers=headers, params=querystring_details)
        response.raise_for_status()  # Ensure the request was successful
        data = response.json()

        # Step 2: Extract subtitle URL from the response
        if 'subtitles' in data and 'items' in data['subtitles'] and data['subtitles']['items']:
            cc_url = data['subtitles']['items'][0]['url']
            #print(cc_url)
        else:
            return "No subtitles available for this video."

        # Step 3: Get the transcript using the subtitle URL
        transcript_url = "https://youtube-media-downloader.p.rapidapi.com/v2/video/subtitles"
        querystring_transcript = {"subtitleUrl": cc_url, "format": "json"}

        trans = requests.get(transcript_url, headers=headers, params=querystring_transcript)
        trans.raise_for_status()  # Ensure the request was successful

        transcript_data = trans.json()

        # Step 4: Concatenate all the transcript text
        full_transcript = " ".join([item['text'] for item in transcript_data])
        #print(full_transcript)

        return full_transcript
    except Exception as e:
        return None


def get_youtube_title(video_id):
    # Build the YouTube service
    youtube = build('youtube', 'v3', developerKey=yt_api_key)

    try:
        # Make an API call to get the video details
        request = youtube.videos().list(part="snippet", id=video_id)
        response = request.execute()

        # Extract the video title
        if response['items']:
            title = response['items'][0]['snippet']['title']
            return title
        else:
            return "Video not found"
    except Exception as e:
        return f"Error: {str(e)}"



# title check
title_templete = '''Assume you have a strong understanding of the stock market, business, and finance. I will provide you with a YouTube video title. Your task is to analyze the title and determine if it mentions any stock or company name, or if it relates to the stock market, finance, or business.

Given title: {title}
Output the result in JSON format:

"valid": If the title includes relevant terms, return 1.If it does not, return 0.

'''
title_prompt = PromptTemplate(template=title_templete, input_variables=["title"])
#llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')
llm_chain_title = LLMChain(prompt=title_prompt, llm=llm_screener)
chain = title_prompt | llm_screener | JsonOutputParser()

response_schemas1 = [
    ResponseSchema(name="summary",
                   description="starting with a short highlight of the transcipt in one paragraph,"
                               "dont include any numbers"),
    ResponseSchema(
        name="Key_points",
        description="Your task is to summarize the text I have given you in up to concise bullet points , "
                    "\
                      dont cover all points covered in summary ,keep only important points that are not covered in summary  "
    ),
    ResponseSchema(
        name="sentiment",
        description="overall sentiment of the yotube transcript towards stock market and for the company."
                    " for ex:positive ,negative and neutral",
    ),
]
output_parser = StructuredOutputParser.from_response_schemas(response_schemas1)
format_instructions = output_parser.get_format_instructions()

templete = '''I trust in your exceptional summarization skills. I have a YouTube transcript that 
I would like you to distill into a concise summary. Please find the transcript provided below:

{transcript}

Your task is to extract the essential key points from the transcript. Please focus on capturing the most important
information related to the impact of the following **key business and stock market terms**:

- **Earnings Report**
- **Dividend**
- **Interest Rates**
- **Inflation**
- **GDP (Gross Domestic Product)**
- **Unemployment Rate**
- **Trade Tariffs**
- **Consumer Confidence**
- **Market Volatility**
- **Mergers and Acquisitions (M&A)**
- **Central Bank Policy**
- **Technological Innovation**
- **Political Instability**
- **Oil Prices**
- **Supply Chain Disruptions**
- **Environmental, Social, and Governance (ESG) Factors**
- **Cryptocurrency**
- **Consumer Spending**
- **Regulatory Changes**
- **Debt**
- **Economic Indicators**

Please highlight words related to these terms in bold, ensuring the summary is concise and highlights the core aspects.
'''

templete = '''Produce a concise summary capturing the essence of a YouTube transcript, emphasizing the high-level insights and numerical data discussed.
Additionally, generate detailed key points outlining specific information, figures, or strategies mentioned in the transcript.
 Lastly, conduct a sentiment analysis to gauge the overall sentiment expressed in the content, and highlight any notable trends or predictions.
 {transcript} {format_instructions}
 '''
templete2 = '''Produce a concise summary capturing the essence of a YouTube video, emphasizing the high-level insights and numerical data discussed.
Additionally, generate detailed key points outlining specific information, figures, or strategies mentioned in the transcript.
 Lastly, conduct a sentiment analysis to gauge the overall sentiment expressed in the content, and highlight any notable trends or predictions.
 {transcript} {format_instructions}
 
 Instructions:
 1.Dont mention word  transcript in summary 
 2.Dont Mention about speaker in the transcript.
 3.ALWAYS FOLLOW FORMAT INSTRUCTIONS.
 '''
summary = PromptTemplate(template=templete2, input_variables=["transcript"],
                         partial_variables={"format_instructions": format_instructions})
# llm = ChatOpenAI(temperature=0.5, model='gpt-4o-mini')
llm_chain = LLMChain(prompt=summary, llm=llm_screener)
app = FastAPI()


class Link(BaseModel):
    link: str

def extract_video_id(url):
    # Regular expression to match YouTube video IDs
    youtube_regex = r'(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([a-zA-Z0-9_-]{11})'
    
    match = re.search(youtube_regex, url)
    if match:
        return match.group(1)
    else:
        return None


@router.post("/get_yt_summary_key")
async def get_yt_summary(link_data: Link,ai_key_auth: str = Depends(authenticate_ai_key)):
    if get_video_length(link_data.link) > 900:
         raise HTTPException(status_code=400, detail="The video you're trying to upload exceeds the 15-minute limit. Please upload a shorter video.")

    with get_openai_callback() as cb:

        link = link_data.link
        _id=extract_video_id(link)
        title=get_youtube_title(_id)
        print(title)
        #print(link)
        #title = get_video_metadata(link)
        result = chain.invoke(input=title)
        print(result['valid'])

        if result['valid'] == 0:
           raise HTTPException(status_code=400, detail="Upload Failed: The video you are trying to upload does not appear to be related to the financial markets. Please ensure your content is focused on finance, investing, stocks, or related topics to maintain platform quality.")

        #_id = link.split("=")[1].split("&")[0]
        # _id=extract_video_id(link)
        # title=get_youtube_title(_id)
        #print(_id)
        try:
            # proxy='103.111.136.82:8080'
            # #print(proxy)
            # proxies = {
            #     'http': proxy,
            #     #'https': proxy
            # }
            # srt = YouTubeTranscriptApi.get_transcript(_id, languages=['hi', 'en'],proxies=proxies)
            # transcript_text = ' '.join(item['text'] for item in srt)
            # loader = YoutubeLoader.from_youtube_url(
            #     link,
            #     add_video_info=True,
            #     transcript_format=TranscriptFormat.CHUNKS,
            #     chunk_size_seconds=500000,
            # )
            # print(loader.load())
            # transcript_text=loader.load()[0].page_content
            # print(transcript_text)
            transcript_text=get_youtube_transcript(_id)
            # print(transcript_text)
            #print(len(transcript_text))
            if transcript_text and len(transcript_text) > 50:
                pass
            else:
                raise HTTPException(status_code=404, detail="No transcript available")
             
            
            #print(transcript_text)
        except Exception as e:
            print(f"Error: {e}")
            raise HTTPException(status_code=400, detail=str(e))
        #transcript_text = ' '.join(item['text'] for item in srt)
        #print(transcript_text)
        with tempfile.NamedTemporaryFile(mode="w", delete=False, encoding='utf-8') as temp_file:
            temp_file.write(transcript_text)
            temp_file_path = temp_file.name

        with open(temp_file_path, encoding='utf-8') as f:
            text = f.read()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=16000,
            chunk_overlap=20,
            length_function=len,
            is_separator_regex=False,
        )

        texts = text_splitter.split_text(text)
        all_results_list = []

        # total_tokens_all_texts = 0
        # prompt_tokens_all_texts = 0
        # completion_tokens_all_texts = 0
        # total_cost_all_texts = 0.1
        for text in texts:
        
            result = llm_chain.run(transcript=text)
            parsed_output = output_parser.parse(result)

            # total_tokens_all_texts += cb.total_tokens
            # prompt_tokens_all_texts += cb.prompt_tokens
            # completion_tokens_all_texts += cb.completion_tokens
            # total_cost_all_texts += cb.total_cost

            all_results_list.append(parsed_output)

        print(f"Total Tokens for All Texts: {cb.total_tokens}")
        print(f"Prompt Tokens for All Texts: {cb.total_tokens}")
        print(f"Completion Tokens for All Texts: {cb.total_tokens}")
        print(f"Total Cost for All Texts (USD): ${cb.total_tokens}")

        # print(all_results_list)
        os.remove(temp_file_path)
        merged_dict = {'summary': '', 'Key_points': [], 'sentiment': ''}

        print(all_results_list)
        for i, d in enumerate(all_results_list):
            merged_dict['summary'] += d['summary'] + '\n'
            # merged_dict['Key_points'] += d['Key_points']
            if 'Key_points' in merged_dict:
                merged_dict['Key_points'].extend(d['Key_points'])
            else:
                merged_dict['Key_points'] = d['Key_points']
            if i == 0:
                merged_dict['sentiment'] = d['sentiment']

        # return merged_dict, f"Total Tokens : {total_tokens_all_texts}", f"Total Cost : ${total_cost_all_texts}"
        # if title:
        #     yt_t=title['title']
        # else:
        #     yt_y=None
        return {"YT_title": title,
            "Response": merged_dict,
            "Total_Tokens": cb.total_tokens,
            "Prompt_Tokens": cb.prompt_tokens,
            "Completion_Tokens": cb.completion_tokens,
            "Total Cost (USD)": cb.total_cost
            }
                            


# yt_link = Link(link="https://www.youtube.com/watch?v=B7IwtoJS1Ik")  # Example YouTube URL

# # Call the get_yt_summary function and print the result
# summary = get_yt_summary(yt_link)
# if summary:
#     print(f"Video Summary: {summary}")
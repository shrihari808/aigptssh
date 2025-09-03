# aigptssh/api/dashboard/llm_generator.py
import json
import os
import asyncio
from config import GPT4o_mini as llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser

class LLMGenerator:
    """
    Uses an LLM to generate structured dashboard content based on pre-processed context for the general market.
    """
    def __init__(self, input_path):
        self.input_path = input_path
        self.data = self._load_data()
        if not self.data:
            raise ValueError("Failed to load or parse the input data file.")

    def _load_data(self):
        try:
            with open(self.input_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error loading data from {self.input_path}: {e}")
            return None

    async def _generate_section(self, section_name, context_docs, prompt_template, output_parser):
        """
        Asynchronously generates a single section of the dashboard content.
        """
        print(f"Generating content for section: {section_name}...")
        if not context_docs:
            print(f"No context available for {section_name}. Skipping.")
            return {"error": f"No context provided for {section_name}."}, []
            
        context_parts = []
        for doc in context_docs:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            url = metadata.get('url', 'N/A')
            age = metadata.get('age', 'N/A')
            context_parts.append(f"Source URL: {url}\nSource Age: {age}\nContent: {text}")
        
        context_str = "\n\n---\n\n".join(context_parts)
        source_urls = list(set([doc['metadata'].get('url') for doc in context_docs if doc['metadata'].get('url')]))
        
        chain = prompt_template | llm | output_parser
        
        try:
            # Use ainvoike for non-blocking LLM call
            response = await chain.ainvoke({"context": context_str})
            return response, source_urls
        except Exception as e:
            print(f"An error occurred during LLM generation for {section_name}: {e}")
            return {"error": "Failed to generate content from LLM."}, []

    async def generate_dashboard_content(self):
        final_output = {
            "last_updated_utc": self.data.get("last_updated_utc"),
            "market_summary": {},
            "latest_news": {},
            "sector_analysis": {},
            "standouts_analysis": {},
            "market_drivers": {}
        }
        
        contexts = self.data.get("llm_contexts", {})

        # --- Define all prompts and parsers ---
        summary_parser = JsonOutputParser()
        summary_prompt = ChatPromptTemplate.from_template(
            """Analyze the provided context about the Indian stock market. 
            Identify 5-6 distinct key themes or summary points for the day.
            For each point, create a title, a concise one-paragraph summary, and determine a representative 'age' based on the Source Age of the content you used.
            The output should be a JSON object containing a list called "summary_points".
            
            Context: {context}
            
            {format_instructions}
            """,
            partial_variables={"format_instructions": summary_parser.get_format_instructions()},
        )
        
        sectors_parser = JsonOutputParser()
        sectors_prompt = ChatPromptTemplate.from_template(
            """Based on the context, identify and summarize the performance of key sectors.
            List the top 2-3 performing sectors and the top 2-3 underperforming sectors.
            Provide a brief, one-sentence explanation for each.
            
            Context: {context}
            
            {format_instructions}
            """,
            partial_variables={"format_instructions": sectors_parser.get_format_instructions()},
        )

        standouts_parser = JsonOutputParser()
        standouts_prompt = ChatPromptTemplate.from_template(
            """From the provided context, identify the top 2-3 standout stock gainers and top 2-3 standout stock losers for the day.
            For each stock, provide a brief, one-sentence reason for its performance if mentioned in the text.

            Context: {context}

            {format_instructions}
            """,
            partial_variables={"format_instructions": standouts_parser.get_format_instructions()},
        )

        drivers_parser = JsonOutputParser()
        drivers_prompt = ChatPromptTemplate.from_template(
            """Analyze the context to determine the key drivers behind today's market performance.
            Summarize the main factors in a single narrative paragraph. Mention elements like GST reforms, global cues, institutional flows, or specific company news that influenced the market.

            Context: {context}

            {format_instructions}
            """,
            partial_variables={"format_instructions": drivers_parser.get_format_instructions()},
        )

        # --- Create a list of tasks to run concurrently ---
        tasks = [
            self._generate_section("market_summary", contexts.get("indices_context"), summary_prompt, summary_parser),
            self._generate_section("sector_analysis", contexts.get("sectors_context"), sectors_prompt, sectors_parser),
            self._generate_section("standouts_analysis", contexts.get("standouts_context"), standouts_prompt, standouts_parser),
            self._generate_section("market_drivers", contexts.get("market_drivers_context"), drivers_prompt, drivers_parser)
        ]

        # --- Run all LLM generation tasks in parallel ---
        results = await asyncio.gather(*tasks)

        # --- Unpack results ---
        (summary_content, summary_sources), (sector_content, _), (standouts_content, _), (drivers_content, _) = results

        # --- Populate final output ---
        final_output["market_summary"] = {
            "summary_points": summary_content.get("summary_points", []),
            "sources": summary_sources
        }
        final_output["latest_news"] = {"articles": self.data.get("latest_news_articles", [])}
        final_output["sector_analysis"] = sector_content
        final_output["standouts_analysis"] = standouts_content
        final_output["market_drivers"] = drivers_content

        print("--- LLM Content Generation Complete ---")
        return final_output

class PortfolioLLMGenerator(LLMGenerator):
    """
    A dedicated generator for creating portfolio-specific snapshots,
    including the new 'Key Issues' section.
    """
    async def generate_dashboard_content(self):
        """
        Overrides the parent method to generate the portfolio-specific layout.
        This is now an async method.
        """
        final_output = {
            "last_updated_utc": self.data.get("last_updated_utc"),
            "market_summary": {},
            "latest_news": {},
            "key_issues": [],
            "market_drivers": {}
        }
        
        contexts = self.data.get("llm_contexts", {})
        portfolio = self.data.get("portfolio", [])

        # 1. Market Summary (tailored to the portfolio)
        summary_parser = JsonOutputParser()
        summary_prompt = ChatPromptTemplate.from_template(
            """Analyze the provided context to generate a market summary for the following portfolio: {portfolio}.
            Identify 3-4 distinct key themes or summary points.
            For each point, create a title, a concise one-paragraph summary, and a representative 'age' based on the source content.
            The final output MUST be a JSON object containing a single key "summary_points", which is a list of these points.

            Context: {context}
            
            {format_instructions}
            """,
            partial_variables={
                "format_instructions": summary_parser.get_format_instructions(),
                "portfolio": ", ".join(portfolio)
             },
        )
        summary_content, summary_sources = await self._generate_section(
            "market_summary", contexts.get("indices_context"), summary_prompt, summary_parser
        )
        final_output["market_summary"] = {
            "summary_points": summary_content.get("summary_points", []),
            "sources": summary_sources
        }

        # 2. Latest News
        latest_news_articles = self.data.get("latest_news_articles", [])
        final_output["latest_news"] = {"articles": latest_news_articles}


        # 3. Key Issues (Concurrent Logic)
        key_issues_content = await self._generate_key_issues(
            contexts.get("key_issues_context"), 
            self.data.get("portfolio", [])
        )
        final_output["key_issues"] = key_issues_content.get("key_issues", [])

        # 4. Market Drivers (tailored to the portfolio)
        drivers_parser = JsonOutputParser()
        drivers_prompt = ChatPromptTemplate.from_template(
            """Analyze the context to determine the key drivers behind the performance of the stocks in this portfolio.
            Summarize the main factors in a single narrative paragraph.

            Context: {context}

            {format_instructions}
            """,
            partial_variables={"format_instructions": drivers_parser.get_format_instructions()},
        )
        drivers_content, _ = await self._generate_section(
            "market_drivers", contexts.get("market_drivers_context"), drivers_prompt, drivers_parser
        )
        final_output["market_drivers"] = drivers_content

        print("--- Portfolio LLM Content Generation Complete ---")
        return final_output

    def _generate_context_string(self, context_docs):
        """Helper to create a context string from document objects."""
        if not context_docs:
            return ""
        context_parts = []
        for doc in context_docs:
            metadata = doc.get('metadata', {})
            text = doc.get('text', '')
            url = metadata.get('url', 'N/A')
            age = metadata.get('age', 'N/A')
            context_parts.append(f"Source URL: {url}\nSource Age: {age}\nContent: {text}")
        return "\n\n---\n\n".join(context_parts)
        
    async def _analyze_stock_issues(self, stock, all_context_docs):
        """
        Asynchronously analyzes a single stock to find its key issues.
        """
        print(f"Analyzing key issues for: {stock}")
        
        stock_context_docs = [
            doc for doc in all_context_docs 
            if stock.lower() in doc.get('text', '').lower() or 
               stock.lower() in doc.get('metadata', {}).get('title', '').lower()
        ]

        if not stock_context_docs:
            print(f"No specific context found for {stock}. Skipping.")
            return []

        context_str = self._generate_context_string(stock_context_docs)
        
        issues_parser = JsonOutputParser()
        issues_prompt = ChatPromptTemplate.from_template(
            """Based on the provided news for {stock}, what are the up to 3 most important 'Key Issues' or themes?
            The output should be a JSON object with a key "issues" which is a list of concise titles.

            Context for {stock}:
            {context}

            {format_instructions}
            """,
            partial_variables={"format_instructions": issues_parser.get_format_instructions()},
        )
        issues_chain = issues_prompt | llm | issues_parser
        
        try:
            issues_result = await issues_chain.ainvoke({"stock": stock, "context": context_str})
            issue_titles = issues_result.get("issues", [])
            print(f"Identified issues for {stock}: {issue_titles}")
        except Exception as e:
            print(f"Error identifying issues for {stock}: {e}")
            return []

        stock_key_issues = []
        for issue_title in issue_titles:
            issue_context_docs = [doc for doc in stock_context_docs if issue_title.lower() in doc.get('text', '').lower()]
            if not issue_context_docs:
                issue_context_docs = stock_context_docs

            issue_context_str = self._generate_context_string(issue_context_docs)
            issue_source_urls = list(set([doc['metadata'].get('url') for doc in issue_context_docs if doc['metadata'].get('url')]))

            views_parser = JsonOutputParser()
            views_prompt = ChatPromptTemplate.from_template(
                """For the Key Issue: "{issue}" for the stock {stock}, analyze the provided context.
                Synthesize the information into a 'bullish_view' and a 'bearish_view'.
                **Each view must be a single paragraph and strictly under 500 characters.**
                The output should also include a 'sources' key with a list of the source URLs used.

                Context for {stock}:
                {context}

                Source URLs: {sources}

                {format_instructions}
                """,
                partial_variables={"format_instructions": views_parser.get_format_instructions()},
            )
            views_chain = views_prompt | llm | views_parser
            try:
                views = await views_chain.ainvoke({
                    "issue": issue_title,
                    "stock": stock,
                    "context": issue_context_str,
                    "sources": issue_source_urls
                })
                stock_key_issues.append({
                    "issue_title": issue_title,
                    "bullish_view": views.get("bullish_view"),
                    "bearish_view": views.get("bearish_view"),
                    "sources": views.get("sources", [])
                })
            except Exception as e:
                print(f"Error generating views for '{issue_title}': {e}")
                continue
        
        return stock_key_issues

    async def _generate_key_issues(self, all_context_docs, portfolio):
        """
        Generates 'Key Issues' by concurrently analyzing each stock.
        """
        print("Generating content for section: Key Issues...")
        if not all_context_docs:
            return {"error": "No context for Key Issues."}

        stocks_to_analyze = portfolio[:3] if len(portfolio) > 3 else portfolio
        
        tasks = [self._analyze_stock_issues(stock, all_context_docs) for stock in stocks_to_analyze]
        results_from_all_stocks = await asyncio.gather(*tasks)
        
        # Flatten the list of lists into a single list
        key_issues_content = [issue for stock_issues in results_from_all_stocks for issue in stock_issues]

        return {"key_issues": key_issues_content}
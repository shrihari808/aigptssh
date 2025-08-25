# /aigptssh/api/reddit_scraper.py
import asyncpraw
from config import REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT

class RedditScraper:
    def __init__(self):
        if not all([REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET, REDDIT_USER_AGENT]):
            raise ValueError(
                "Missing Reddit API credentials. Please ensure REDDIT_CLIENT_ID, "
                "REDDIT_CLIENT_SECRET, and REDDIT_USER_AGENT are set in your environment."
            )
        
        self.reddit = asyncpraw.Reddit(
            client_id=REDDIT_CLIENT_ID,
            client_secret=REDDIT_CLIENT_SECRET,
            user_agent=REDDIT_USER_AGENT,
        )

    async def scrape_post(self, article: dict) -> dict:
        """
        Scrapes a Reddit post, its comments, and replies asynchronously.
        """
        url = article.get("url")
        if not url:
            return None
            
        print(f"DEBUG: Scraping Reddit URL: {url}")
        try:
            submission = await self.reddit.submission(url=url)
            comments = []
            await submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                comments.append({
                    "body": comment.body,
                    "author": str(comment.author),
                    "score": comment.score,
                })

            scraped_data = {
                "title": submission.title,
                "score": submission.score,
                "selftext": submission.selftext,
                "comments": comments,
                "url": url,
                "page_age": article.get("page_age")
            }
            print(f"DEBUG: Successfully scraped '{submission.title}'. Found {len(comments)} comments.")
            return scraped_data
        except Exception as e:
            if "401" in str(e):
                print(f"CRITICAL ERROR: Reddit API returned a 401 Unauthorized error. "
                      f"Please check your REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET.")
            else:
                print(f"Error scraping Reddit URL {url}: {e}")
            return None
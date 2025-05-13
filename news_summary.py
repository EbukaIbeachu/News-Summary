import feedparser
import requests
import logging
import streamlit as st
import pandas as pd
from bs4 import BeautifulSoup
from transformers import pipeline
import nltk

# Setup
nltk.download('punkt', quiet=True)
logging.basicConfig(level=logging.INFO)

# Constants
KEYWORDS = ['oil', 'energy', 'Nigeria', 'inflation', 'Russia', 'war', 'OPEC', 'gas', 'election', 'politics']
NEWS_SITES = {
    'BBC': 'http://feeds.bbci.co.uk/news/rss.xml',
    'CNN': 'http://rss.cnn.com/rss/edition.rss',
    'Al Jazeera': 'https://www.aljazeera.com/xml/rss/all.xml',
    'Guardian Nigeria': 'https://guardian.ng/feed/',
    'Punch': 'https://punchng.com/feed/',
}
DATA_PATH = 'News.csv'

# Summarizer
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

# Functions
def summarize(text):
    try:
        return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    except:
        return "Could not summarize."

def get_article_text(url):
    try:
        res = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(res.content, 'html.parser')
        return ' '.join(p.text for p in soup.find_all('p'))[:2000]
    except:
        return ""

def fetch_articles():
    articles = []
    for name, feed_url in NEWS_SITES.items():
        feed = feedparser.parse(feed_url)
        for entry in feed.entries[:5]:
            if any(k in entry.title.lower() for k in KEYWORDS):
                text = get_article_text(entry.link)
                if len(text.split()) > 100:
                    summary = summarize(text)
                    articles.append({
                        'Source': name,
                        'Title': entry.title,
                        'Link': entry.link,
                        'Summary': summary
                    })
    return articles

# Streamlit UI
st.title("News Summary Dashboard")

if st.button("Get News Summaries"):
    with st.spinner("Fetching and summarizing..."):
        results = fetch_articles()
        if results:
            df = pd.DataFrame(results)
            df.to_csv(DATA_PATH, index=False)
            st.success(f"Fetched {len(df)} articles.")
            st.dataframe(df)
        else:
            st.warning("No articles found.")
elif st.checkbox("Show previous results") and os.path.exists(DATA_PATH):
    st.dataframe(pd.read_csv(DATA_PATH))
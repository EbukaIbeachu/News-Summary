import os
import csv
import difflib
import requests
import logging
import streamlit as st
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from newspaper import Article, Config, utils
from transformers import pipeline
import nltk

# ------------------------------
# Environment Setup
# ------------------------------
nltk.download('punkt', quiet=True)
utils.cache_disk.enabled = False  # Disable newspaper3k caching

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# ------------------------------
# Configurations
# ------------------------------
KEYWORDS = [
    'war', 'Russia', 'tariff', 'tariffs', 'inflation', 'Nigeria', 'boko haram',
    'Fulani', 'Fulani herdsmen', 'Tinubu', 'USA', 'America', 'China',
    'Trade', 'Trade war', 'Prices', 'Oil', 'Investments', 'Ukraine', 'Oil price',
    'Crude oil', 'Brent crude', 'WTI', 'Energy', 'Gasoline', 'Diesel',
    'Renewable energy', 'Electric vehicles', 'Solar power', 'Wind energy', 'Natural gas',
    'Energy crisis', 'OPEC', 'Fossil fuels', 'Sustainable energy', 'Climate change', 'Martins Otse',
    'VeryDarkMan', 'Senate President', 'Nigerian Senate', 'Nigerian Politics',
    '2023 Elections', 'Political Parties', 'Electoral Commission', 'INEC', 'Electoral Act', 'Voter Registration',
    'Voting Process', 'Election Results', 'Political Campaigns', 'Political Violence', 'Electoral Fraud',
    'Corruption in Politics', 'Political Accountability', 'Democracy in Nigeria',
    'Political Parties in Nigeria', 'Nigerian Constitution', 'Judiciary', 'Rule of Law',
    'Human Rights', 'Civil Society', 'Political Activism', 'Youth Participation', 'EFCC', 'Political Appointees',
    'Natasha Akpoti-Uduaghan', 'Senator Orji Uzor Kalu', 'PDP', 'APC', 'Labour Party', 'Social Democratic Party',
    'Nigerian Governors', 'Nigerian Senators', 'Nigerian House of Representatives', 'Nigerian Judiciary',
    'Godswill Obot Akpabio', 'Iran', 'Saudi Arabia', 'Venezuela', 'Iraq', 'Libya', 'Angola', 'United Kingdom', 'Turkey', 'Brazil',
    'Portugal', 'Spain', 'Italy', 'France', 'Germany', 'Netherlands', 'Belgium', 'Sweden', 'Norway',
    'Denmark', 'Finland', 'Switzerland', 'Austria', 'Czech Republic', 'Poland', 'Hungary', 'Romania', 'brent crude', 'oil demand', 'oil supply', 'crude oil', 'barrel',
    'strategic petroleum reserve', 'refinery', 'energy prices',
    'fossil fuels', 'global energy', 'gasoline prices', 'diesel prices',
    'energy inflation', 'oil market', 'oil production', 'oil exports',
    'oil imports', 'Russia oil', 'US shale', 'energy crisis',
    'IEA', 'EIA', 'Middle East tension'
]

NEWS_SITES = {
    'BBC News': 'http://feeds.bbci.co.uk/news/rss.xml',
    'CNN': 'http://rss.cnn.com/rss/edition.rss',
    'The Guardian': 'https://www.theguardian.com/world/rss',
    'Al Jazeera': 'https://www.aljazeera.com/xml/rss/all.xml',
    'Reuters': 'http://feeds.reuters.com/reuters/topNews',
    'The New York Times': 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml',
    'Fox News': 'http://feeds.foxnews.com/foxnews/latest',
    'NBC News': 'http://feeds.nbcnews.com/nbcnews/public/news',
    'Sky News': 'https://feeds.skynews.com/feeds/rss/home.xml',
    'The Independent': 'https://www.independent.co.uk/news/uk/rss',
    'USA Today': 'http://rssfeeds.usatoday.com/usatoday-NewsTopStories',
    'The Washington Post': 'http://feeds.washingtonpost.com/rss/national',
    'Bloomberg': 'https://www.bloomberg.com/feed/podcast/etf-report.xml',
    'Yahoo News': 'https://www.yahoo.com/news/rss',
    'Google News': 'https://news.google.com/rss',
    'Punch Newspapers': 'https://punchng.com/feed/',
    'The Guardian Nigeria': 'https://guardian.ng/feed/',
    'Vanguard': 'https://www.vanguardngr.com/feed/',
    'ThisDay Live': 'https://www.thisdaylive.com/index.php/feed/',
    'The Nation': 'https://thenationonlineng.net/feed/',
    'Daily Trust': 'https://dailytrust.com/feed',
    'Leadership News': 'https://leadership.ng/feed/',
    'BusinessDay': 'https://businessday.ng/feed/',
    'Nairametrics': 'https://nairametrics.com/feed/',
    'Proshare Nigeria': 'https://www.proshareng.com/rss',
    'Linda Ikeji’s Blog': 'https://www.lindaikejisblog.com/feeds/posts/default',
    'BellaNaija': 'https://www.bellanaija.com/feed/',
    'Pulse Nigeria': 'https://www.pulse.ng/news/rss',
    'Naijaloaded': 'https://www.naijaloaded.com.ng/feed',
    'OilPrice': 'https://www.oilprice.com',
    'Bloomberg Energy': 'https://www.bloomberg.com/energy',
    'Reuters Commodities': 'https://www.reuters.com/markets/commodities/',
    'EIA': 'https://www.eia.gov/',
    'S&P Global Commodity Insights': 'https://www.spglobal.com/commodityinsights',
    'WSJ Energy': 'https://www.wsj.com/news/energy-oil-gas',
    'CNBC Energy RSS': 'https://www.cnbc.com/id/15838368/device/rss/rss.html',
    'IEA News RSS': 'https://www.iea.org/rss/news'
}


DATA_PATH = "News.csv"
SIMILARITY_THRESHOLD = 0.85

# ------------------------------
# Summarizer Setup
# ------------------------------
summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# ------------------------------
# Helper Functions
# ------------------------------
def fetch_articles():
    articles = []
    headers = {'User-Agent': 'Mozilla/5.0'}
    for source, feed_url in NEWS_SITES.items():
        try:
            response = requests.get(feed_url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.content, 'xml')
            items = soup.find_all('item')
            for item in items:
                title = item.title.text if item.title else ''
                link = item.link.text if item.link else ''
                if not title or not link:
                    continue
                if any(keyword.lower() in title.lower() for keyword in KEYWORDS):
                    try:
                        article = Article(link)
                        article.download()
                        article.parse()
                        article.nlp()
                        summary = summarize_article(article.text)
                        pub_date = extract_publish_date(link) or datetime.utcnow()
                    except Exception:
                        summary = "Summary not available."
                        pub_date = datetime.utcnow()
                    articles.append({
                        'Title': title,
                        'Link': link,
                        'Source': source,
                        'Published': pub_date.strftime('%Y-%m-%d %H:%M:%S'),
                        'Summary': summary
                    })
        except Exception as e:
            logging.warning(f"Error fetching from {source}: {e}")
    return articles

def summarize_article(text):
    try:
        return summarizer(text[:1024], max_length=120, min_length=30, do_sample=False)[0]['summary_text']
    except Exception as e:
        logging.error(f"Summarization failed: {e}")
        return "Summary not available."

def is_similar(title, existing_titles):
    return any(
        difflib.SequenceMatcher(None, title.lower(), existing.lower()).ratio() >= SIMILARITY_THRESHOLD
        for existing in existing_titles
    )

def load_existing_titles():
    if not os.path.exists(DATA_PATH):
        return []
    with open(DATA_PATH, mode='r', encoding='utf-8') as f:
        return [row['Title'] for row in csv.DictReader(f)]

def extract_publish_date(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        meta = soup.find('meta', attrs={'property': 'article:published_time'})
        if meta and meta.get('content'):
            return datetime.fromisoformat(meta['content'].replace('Z', '+00:00'))
    except Exception as e:
        logging.warning(f"Could not extract publish date from {url}: {e}")
    return None

def save_articles_to_csv(articles):
    fieldnames = ['Title', 'Link', 'Source', 'Published', 'Summary']
    write_header = not os.path.exists(DATA_PATH)
    with open(DATA_PATH, mode='a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for article in articles:
            writer.writerow(article)

# ------------------------------
# Streamlit App Interface
# ------------------------------
st.title("📰 Global News Summary Dashboard")

if st.button("Fetch Latest News"):
    with st.spinner("Fetching and summarizing articles... Please wait."):
        existing_titles = load_existing_titles()
        new_articles = fetch_articles()
        filtered_articles = [
            a for a in new_articles if not is_similar(a['Title'], existing_titles)
        ]

        if filtered_articles:
            save_articles_to_csv(filtered_articles)
            df = pd.DataFrame(filtered_articles)
            st.success(f"✅ Fetched and summarized {len(df)} articles.")

            for _, row in df.iterrows():
                st.markdown(f"### {row['Title']}")
                st.markdown(f"**Source:** {row['Source']} | **Published:** {row['Published']}")
                st.markdown(f"**Summary:** {row['Summary']}")
                st.markdown(f"[Read more...]({row['Link']})", unsafe_allow_html=True)
                st.markdown("---")
        else:
            st.warning("No new articles matched your keywords or all were duplicates.")

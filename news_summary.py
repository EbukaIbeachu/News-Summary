import os
import csv
import difflib
import requests
import logging
import streamlit as st
import pandas as pd
from datetime import datetime
from bs4 import BeautifulSoup
from transformers import pipeline
import nltk

# ------------------------------
# Environment Setup
# ------------------------------
nltk.download('punkt', quiet=True)

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
summarizer = pipeline("summarization", model="knkarthick/MEETING_SUMMARY")

# ------------------------------
# Helper Functions
# ------------------------------

def summarize_article(text):
    try:
        return summarizer(text, max_length=130, min_length=30, do_sample=False)[0]['summary_text']
    except:
        return "Summary failed."

def extract_article_text(url):
    try:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join(p.get_text() for p in paragraphs if p.get_text())
        return text.strip()
    except Exception as e:
        logging.warning(f"Failed to extract article from {url}: {e}")
        return ""

def extract_publish_date(url):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        pub_date_tag = soup.find('meta', {'property': 'article:published_time'})
        if pub_date_tag and pub_date_tag.get('content'):
            return pub_date_tag.get('content')
    except:
        pass
    return None

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
                    text = extract_article_text(link)
                    if text:
                        summary = summarize_article(text)
                        pub_date = extract_publish_date(link) or datetime.now().isoformat()
                        articles.append({
                            'source': source,
                            'title': title,
                            'link': link,
                            'summary': summary,
                            'published': pub_date
                        })
        except Exception as e:
            logging.warning(f"Failed to fetch from {source}: {e}")
    return articles
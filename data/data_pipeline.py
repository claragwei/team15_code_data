#!/usr/bin/env python3
"""
FILMLYTICS DATA PIPELINE

Consolidated script that reproduces all data collection and preprocessing steps.

This script chains together:
1. TMDB API data collection (movie metadata from 2010-2025)
2. Data cleaning and filtering
3. Rotten Tomatoes URL mapping
4. RT review scraping
5. BERT sentiment analysis on reviews
6. Gender diversity metrics via TMDB API
7. YouTube trailer metrics via YouTube Data API
8. Final dataset merge

REQUIREMENTS:
    pip install pandas numpy requests beautifulsoup4 transformers torch tqdm

API KEYS REQUIRED:
    - TMDB Bearer Token (get from https://www.themoviedb.org/settings/api)
    - YouTube Data API Key (get from https://console.cloud.google.com/apis/credentials)

SETTING UP API KEYS:
    Option 1 - Environment Variables (Recommended):
        export TMDB_BEARER_TOKEN="your_tmdb_token_here"
        export YOUTUBE_API_KEY="your_youtube_key_here"
        python data_pipeline.py --step all

    Option 2 - .env File (Not committed to version control):
        1. Create a file named `.env` in the same directory as this script
        2. Add the following lines:
           TMDB_BEARER_TOKEN=your_tmdb_token_here
           YOUTUBE_API_KEY=your_youtube_key_here
        3. Install python-dotenv: pip install python-dotenv
        4. The script will automatically load from .env

OUTPUT FILES:
    data/tmdb_data.csv             - TMDB data from API
    data/tmdb_with_urls.csv        - TMDB + RT URLs
    data/rt_reviews.json           - Scraped RT reviews
    data/rt_sentiment.csv          - Sentiment scores from reviews
    data/tmdb_with_sentiment.csv   - TMDB + sentiment data
    data/complete_data.csv         - Final merged dataset w/ TMDB, RT, Youtube, sentiment analysis, & diversity

Authors: Team 15
Date: December 2025
"""

import os
import sys
import json
import time
import random
import argparse
from datetime import datetime
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse, parse_qs

import pandas as pd
import numpy as np
import requests
from tqdm import tqdm

# CONFIGURATION

# Try to load environment variables from .env file if python-dotenv is installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, will use os.environ directly

# API Keys - Loaded from environment variables
TMDB_BEARER_TOKEN = os.environ.get("TMDB_BEARER_TOKEN", "YOUR_TMDB_BEARER_TOKEN_HERE")
YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "YOUR_YOUTUBE_API_KEY_HERE")

# Validate API keys at startup

# Warn if API keys not set
if TMDB_BEARER_TOKEN == "YOUR_TMDB_BEARER_TOKEN_HERE":
    print("WARNING: Set TMDB_BEARER_TOKEN environment variable")
if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
    print("WARNING: Set YOUTUBE_API_KEY environment variable")

# Directory structure
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# TMDB API settings
TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_HEADERS = {
    "accept": "application/json",
    "Authorization": f"Bearer {TMDB_BEARER_TOKEN}"
}

# YouTube API settings
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"

# Date ranges for collection (2010-2025)
COLLECTION_YEARS = list(range(2010, 2026))
QUARTERS = [
    ('Q1', '01-01', '03-31'),
    ('Q2', '04-01', '06-30'),
    ('Q3', '07-01', '09-30'),
    ('Q4', '10-01', '12-31')
]

# High-volume quarters that need month-by-month collection (>500 pages per quarter)
QUARTERS_EXCEEDING_LIMIT = {
    (2019, 'Q4'), (2020, 'Q4'), (2021, 'Q4'),
    (2022, 'Q3'), (2022, 'Q4'),
    (2023, 'Q1'), (2023, 'Q2'), (2023, 'Q3'), (2023, 'Q4'),
    (2024, 'Q1'), (2024, 'Q2'), (2024, 'Q3'), (2024, 'Q4'),
    (2025, 'Q1'), (2025, 'Q2'), (2025, 'Q3')
}

MONTHS = {
    'Q1': [('Jan', '01-01', '01-31'), ('Feb', '02-01', '02-28'), ('Mar', '03-01', '03-31')],
    'Q2': [('Apr', '04-01', '04-30'), ('May', '05-01', '05-31'), ('Jun', '06-01', '06-30')],
    'Q3': [('Jul', '07-01', '07-31'), ('Aug', '08-01', '08-31'), ('Sep', '09-01', '09-30')],
    'Q4': [('Oct', '10-01', '10-31'), ('Nov', '11-01', '11-30'), ('Dec', '12-01', '12-31')]
}

# UTILITY FUNCTIONS

def print_stats(df: pd.DataFrame, name: str):
    """Print basic dataframe statistics"""
    print(f"{name}: {len(df)} rows")

def save_checkpoint(data: Any, filepath: str):
    """Save data checkpoint (JSON or CSV)"""
    if filepath.endswith('.json'):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    elif filepath.endswith('.csv'):
        if isinstance(data, pd.DataFrame):
            data.to_csv(filepath, index=False)
    pass  # checkpoint saved

def load_checkpoint(filepath: str) -> Any:
    """Load data checkpoint"""
    if not os.path.exists(filepath):
        return None
    if filepath.endswith('.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    elif filepath.endswith('.csv'):
        return pd.read_csv(filepath, low_memory=False)
    return None

def check_tmdb_api_key():
    """Check if TMDB API key is configured"""
    if TMDB_BEARER_TOKEN == "YOUR_TMDB_BEARER_TOKEN_HERE":
        print("ERROR: TMDB_BEARER_TOKEN not set")
        return False
    return True

def check_youtube_api_key():
    """Check if YouTube API key is configured"""
    if YOUTUBE_API_KEY == "YOUR_YOUTUBE_API_KEY_HERE":
        print("ERROR: YOUTUBE_API_KEY not set")
        return False
    return True

def extract_youtube_video_id(url: str) -> Optional[str]:
    """Extract video ID from YouTube URL"""
    if pd.isna(url) or not url:
        return None
    try:
        parsed_url = urlparse(str(url))
        if parsed_url.hostname in ["www.youtube.com", "youtube.com"]:
            return parse_qs(parsed_url.query).get("v", [None])[0]
        elif parsed_url.hostname == "youtu.be":
            return parsed_url.path[1:]
        return None
    except Exception:
        return None

def days_until_release(published_at: str, release_date: str) -> Optional[int]:
    """Calculate days from trailer publish date to movie release"""
    try:
        pub_date = datetime.strptime(published_at[:10], "%Y-%m-%d")
        rel_date = datetime.strptime(release_date[:10], "%Y-%m-%d")
        return (rel_date - pub_date).days
    except Exception:
        return None

# STEP 1: TMDB DATA COLLECTION

class TMDBCollector:
    """Collects movie data from TMDB API"""
    
    def __init__(self):
        self.request_count = 0
        self.last_request_time = 0
        self.min_interval = 1.0 / 20.0  # 20 requests per second max
        
    def _rate_limit(self):
        """Enforce rate limiting"""
        elapsed = time.time() - self.last_request_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request_time = time.time()
        self.request_count += 1
    
    def _request(self, url: str, params: dict, retries: int = 3) -> dict:
        """Make API request with retry logic"""
        self._rate_limit()
        for attempt in range(retries):
            try:
                response = requests.get(url, params=params, headers=TMDB_HEADERS, timeout=30)
                if response.status_code == 200:
                    return response.json()
                elif response.status_code == 429:
                    time.sleep(10)
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(5 * (attempt + 1))
                else:
                    pass
        return {}
    
    def get_discover_movies(self, page: int, date_from: str, date_to: str) -> dict:
        """Get movies for a date range"""
        url = f"{TMDB_BASE_URL}/discover/movie"
        params = {
            'region': 'US',
            'primary_release_date.gte': date_from,
            'primary_release_date.lte': date_to,
            'include_adult': False,
            'sort_by': 'release_date.desc',
            'page': page,
            'language': 'en-US',
            'vote_count.gte': 1
        }
        return self._request(url, params)
    
    def get_movie_details(self, movie_id: int) -> dict:
        """Get detailed info for a movie"""
        url = f"{TMDB_BASE_URL}/movie/{movie_id}"
        params = {
            'language': 'en-US',
            'append_to_response': 'credits,keywords,images,videos'
        }
        return self._request(url, params)
    
    def extract_features(self, data: dict) -> Optional[dict]:
        """Extract features from movie data"""
        try:
            features = {
                'id': data.get('id'),
                'title': data.get('title'),
                'release_date': data.get('release_date'),
                'budget': data.get('budget', 0),
                'runtime': data.get('runtime', 0),
                'vote_count': data.get('vote_count', 0),
                'vote_average': data.get('vote_average', 0),
                'popularity': data.get('popularity', 0),
                'tagline': data.get('tagline', ''),
                'overview': data.get('overview', ''),
            }
            
            # Genres
            genres = data.get('genres', [])
            features['genres'] = ', '.join([g['name'] for g in genres]) if genres else ''
            
            # Production companies and countries
            prod_companies = data.get('production_companies', [])
            features['production_companies'] = ', '.join([c['name'] for c in prod_companies]) if prod_companies else ''
            
            prod_countries = data.get('production_countries', [])
            features['production_countries'] = ', '.join([c['name'] for c in prod_countries]) if prod_countries else ''
            
            # Keywords
            keywords = data.get('keywords', {})
            keyword_list = keywords.get('keywords', []) if isinstance(keywords, dict) else []
            features['keywords'] = ', '.join([k['name'] for k in keyword_list]) if keyword_list else ''
            
            # Credits
            credits = data.get('credits', {})
            cast = credits.get('cast', [])
            crew = credits.get('crew', [])
            features['cast'] = ', '.join([c['name'] for c in cast[:10]])
            features['directors'] = ', '.join([c['name'] for c in crew if c.get('job') == 'Director'])
            features['producers'] = ', '.join([c['name'] for c in crew if c.get('job') == 'Producer'])
            
            # Images
            images = data.get('images', {})
            posters = images.get('posters', [])
            if posters:
                features['poster_url'] = f"https://image.tmdb.org/t/p/w500{posters[0].get('file_path')}"
            else:
                features['poster_url'] = None
            
            # Trailer
            videos = data.get('videos', {}).get('results', [])
            trailers = [v for v in videos if v.get('type') == 'Trailer']
            if trailers:
                features['trailer_url'] = f"https://www.youtube.com/watch?v={trailers[0].get('key')}"
            else:
                features['trailer_url'] = None
            
            return features
        except Exception as e:
            return None
    
    def collect_period(self, year: int, quarter: tuple, max_pages: int = 500) -> List[dict]:
        """Collect movies for a quarter or month"""
        q_name, q_start, q_end = quarter
        date_from = f"{year}-{q_start}"
        date_to = f"{year}-{q_end}"
        
        # Handle leap years for February
        if q_name == 'Feb' and year % 4 == 0 and (year % 100 != 0 or year % 400 == 0):
            date_to = f"{year}-02-29"
        
        print(f"   Collecting {year} {q_name}...")
        
        movies = []
        for page in range(1, max_pages + 1):
            discover = self.get_discover_movies(page, date_from, date_to)
            if not discover or 'results' not in discover:
                break
            
            results = discover['results']
            if not results:
                break
            
            for movie in results:
                details = self.get_movie_details(movie['id'])
                if details:
                    features = self.extract_features(details)
                    if features:
                        movies.append(features)
            
            if page % 10 == 0:
                pass
        
        print(f"   {year} {q_name}: {len(movies)} movies")
        return movies

def collect_tmdb(resume: bool = True):
    """Step 1: Collect movie data from TMDB API"""
    
    if not check_tmdb_api_key():
        return None
    
    output_file = os.path.join(DATA_DIR, "tmdb_data.csv")
    progress_file = os.path.join(DATA_DIR, "tmdb_progress.json")
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    all_movies = []
    completed_periods = set()
    
    if resume and os.path.exists(progress_file):
        progress = load_checkpoint(progress_file)
        if progress:
            all_movies = progress.get('movies', [])
            completed_periods = set(progress.get('completed', []))
    
    collector = TMDBCollector()
    
    for year in COLLECTION_YEARS:
        for quarter in QUARTERS:
            q_name = quarter[0]
            
            # Check if this quarter needs month-by-month collection
            if (year, q_name) in QUARTERS_EXCEEDING_LIMIT:
                # Collect by month
                for month in MONTHS[q_name]:
                    period_key = f"{year}_{q_name}_{month[0]}"
                    if period_key in completed_periods:
                        continue
                    
                    movies = collector.collect_period(year, month)
                    all_movies.extend(movies)
                    completed_periods.add(period_key)
                    
                    save_checkpoint({
                        'movies': all_movies,
                        'completed': list(completed_periods)
                    }, progress_file)
            else:
                # Collect whole quarter
                period_key = f"{year}_{q_name}"
                if period_key in completed_periods:
                    continue
                
                movies = collector.collect_period(year, quarter)
                all_movies.extend(movies)
                completed_periods.add(period_key)
                
                save_checkpoint({
                    'movies': all_movies,
                    'completed': list(completed_periods)
                }, progress_file)
    
    # Remove duplicates by movie ID
    seen_ids = set()
    unique_movies = []
    for movie in all_movies:
        if movie['id'] not in seen_ids:
            seen_ids.add(movie['id'])
            unique_movies.append(movie)
    
    df = pd.DataFrame(unique_movies)
    df = df.sort_values('release_date', ascending=True)
    df.to_csv(output_file, index=False)
    print_stats(df, "TMDB Raw Data")
    print(f"Saved: {output_file}")
    
    return df

# STEP 2: DATA CLEANING

def clean_data():
    """Step 2: Clean and filter TMDB data"""
    
    input_file = os.path.join(DATA_DIR, "tmdb_data.csv")
    output_file = os.path.join(DATA_DIR, "tmdb_data.csv")
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return None
    
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Loaded {len(df)} movies")
    
    # Filter rows with missing critical data
    initial_count = len(df)
    df = df[df['title'].notna()]
    
    initial_count = len(df)
    df = df[df['vote_average'] > 0]
    
    initial_count = len(df)
    df = df[df['vote_count'] >= 5]
    
    initial_count = len(df)
    df = df[df['genres'].notna() & (df['genres'].str.strip() != '')]
    
    initial_count = len(df)
    df = df[df['overview'].notna() & (df['overview'].str.strip() != '')]
    
    initial_count = len(df)
    df = df[df['release_date'].notna()]
    
    # Drop unnecessary columns (as per clean_tmdb.py)
    drop_cols = ['popularity', 'tagline', 'keywords', 'producers']
    drop_cols = [col for col in drop_cols if col in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    
    # Create target variable
    df['is_successful'] = (df['vote_average'] >= 6.0).astype(int)
    success_rate = df['is_successful'].mean() * 100
    
    df = df.reset_index(drop=True)
    df.to_csv(output_file, index=False)
    print_stats(df, "Cleaned Data")
    print(f"Saved: {output_file}")
    
    return df

# STEP 3: ROTTEN TOMATOES URL MAPPING

def map_rt_urls():
    """Step 3: Map TMDB movies to Rotten Tomatoes URLs"""
    
    input_file = os.path.join(DATA_DIR, "tmdb_data.csv")
    output_file = os.path.join(DATA_DIR, "tmdb_with_urls.csv")
    rt_mapping_file = os.path.join(BASE_DIR, "Dataset", "url mapping", "movie_info.csv")
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return None
    
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Loaded {len(df)} movies")
    
    if os.path.exists(rt_mapping_file):
        rt_df = pd.read_csv(rt_mapping_file)
        print(f"Loaded {len(rt_df):} RT URLs")
        
        df['title_clean'] = df['title'].str.lower().str.strip()
        rt_df['title_clean'] = rt_df['title'].str.lower().str.strip()
        
        df = pd.merge(df, rt_df[['title_clean', 'url']], on='title_clean', how='left')
        df['rt_url'] = df['url'].fillna('')
        df['has_rt_url'] = (df['rt_url'] != '').astype(int)
        df = df.drop(columns=['title_clean', 'url'])
        
        matched = df['has_rt_url'].sum()
        print(f"\nMatched {matched:} movies with RT URLs ({matched/len(df)*100:.1f}%)")
    else:
        print(f"RT mapping file not found: {rt_mapping_file}")
        df['rt_url'] = ''
        df['has_rt_url'] = 0
    
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")
    
    return df

# STEP 4: ROTTEN TOMATOES REVIEW SCRAPING

def scrape_rt_reviews():
    """Step 4: Scrape reviews from Rotten Tomatoes"""
    
    from bs4 import BeautifulSoup
    
    input_file = os.path.join(DATA_DIR, "tmdb_with_urls.csv")
    output_file = os.path.join(DATA_DIR, "rt_reviews.json")
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return None
    
    df = pd.read_csv(input_file, low_memory=False)
    urls = df[df['rt_url'].str.startswith('http', na=False)]['rt_url'].tolist()
    print(f"Found {len(urls):} movies with RT URLs")
    
    results = []
    scraped_urls = set()
    if os.path.exists(output_file):
        results = load_checkpoint(output_file)
        scraped_urls = {r.get('url') for r in results if r.get('url')}
    
    headers = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"}
    
    for url in tqdm(urls, desc="Scraping RT"):
        if url in scraped_urls:
            continue
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            soup = BeautifulSoup(response.text, "html.parser")
            
            # Extract scores
            script_tag = soup.select_one("media-scorecard-manager script")
            critic_score = audience_score = "N/A"
            
            if script_tag:
                try:
                    data = json.loads(script_tag.text.strip())
                    critic_score = data.get("tomatometer", {}).get("score", "N/A")
                    audience_score = data.get("audienceScore", {}).get("score", "N/A")
                except:
                    pass
            
            # Fetch reviews
            reviews = []
            review_url = f"{url}/reviews"
            res = requests.get(f"{review_url}?type=top_critics", headers=headers, timeout=10)
            if res.status_code == 200:
                review_soup = BeautifulSoup(res.text, "html.parser")
                for block in review_soup.select("review-speech-balloon, div.review-row, div.review-table-row"):
                    quote = block.select_one(".the_review, p")
                    if quote and quote.text.strip():
                        reviews.append(quote.text.strip())
                        if len(reviews) >= 20:
                            break
            
            results.append({
                "url": url,
                "critic_score": critic_score,
                "audience_score": audience_score,
                "reviews": reviews
            })
            scraped_urls.add(url)
            
            if len(results) % 10 == 0:
                save_checkpoint(results, output_file)
            
            time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            continue
    
    save_checkpoint(results, output_file)
    print(f"\nScraped {len(results):} movies")
    print(f"   Saved to: {output_file}")
    
    return results

# STEP 5: SENTIMENT ANALYSIS

def sentiment_analysis():
    """Step 5: Run BERT sentiment analysis on reviews"""
    
    try:
        from transformers import pipeline
    except ImportError:
        print("transformers library not installed")
        return None
    
    input_file = os.path.join(DATA_DIR, "rt_reviews.json")
    output_file = os.path.join(DATA_DIR, "rt_sentiment.csv")
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return None
    
    reviews_data = load_checkpoint(input_file)
    print(f"Loaded {len(reviews_data):} movies with reviews")
    
    print("Loading sentiment model...")
    sentiment_pipe = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=-1
    )
    
    results = []
    for movie in tqdm(reviews_data, desc="Analyzing sentiment"):
        reviews = movie.get('reviews', [])
        
        if not reviews:
            avg_sentiment = None
        else:
            scores = []
            for review in reviews[:20]:
                try:
                    pred = sentiment_pipe(review[:512], truncation=True)[0]
                    score = pred['score'] if pred['label'] == 'POSITIVE' else -pred['score']
                    scores.append(score)
                except:
                    continue
            avg_sentiment = sum(scores) / len(scores) if scores else None
        
        results.append({
            'rt_url': movie.get('url'),
            'critic_score': movie.get('critic_score'),
            'audience_score': movie.get('audience_score'),
            'review_sentiment': avg_sentiment,
            'num_reviews': len(reviews)
        })
    
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    print(f"\nAnalyzed {len(results):} movies")
    print(f"   Saved to: {output_file}")
    
    return df

# STEP 6: GENDER DIVERSITY METRICS

def add_diversity():
    """Step 6: Add gender diversity metrics via TMDB API"""
    
    if not check_tmdb_api_key():
        return None
    
    input_file = os.path.join(DATA_DIR, "tmdb_with_urls.csv")
    cache_file = os.path.join(DATA_DIR, "diversity_cache.json")
    output_file = os.path.join(DATA_DIR, "tmdb_with_sentiment.csv")
    
    if not os.path.exists(input_file):
        print(f"Input file not found: {input_file}")
        return None
    
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Loaded {len(df)} movies")
    
    cache = {}
    if os.path.exists(cache_file):
        cache = load_checkpoint(cache_file)
    
    def get_gender(name: str) -> int:
        """Get gender from TMDB API (0=unknown, 1=female, 2=male)"""
        if name in cache:
            return cache[name]
        
        try:
            url = f"{TMDB_BASE_URL}/search/person"
            params = {"query": name, "include_adult": "false"}
            response = requests.get(url, params=params, headers=TMDB_HEADERS, timeout=5)
            if response.status_code == 200:
                results = response.json().get('results', [])
                if results:
                    gender = results[0].get('gender', 0)
                    cache[name] = gender
                    return gender
            elif response.status_code == 429:
                time.sleep(10)
                return get_gender(name)  # Retry
        except:
            pass
        cache[name] = 0
        return 0
    
    # Initialize diversity columns
    df['female_cast_count'] = 0
    df['male_cast_count'] = 0
    df['female_cast_percentage'] = 0.0
    df['gender_balance_score'] = 0.0
    df['director_gender'] = 0
    df['female_director'] = False
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Fetching gender data"):
        cast_str = row.get('cast', '')
        if pd.notna(cast_str) and cast_str:
            cast_list = [c.strip() for c in str(cast_str).split(',')[:10]]
            genders = [get_gender(name) for name in cast_list]
            
            female = genders.count(1)
            male = genders.count(2)
            total = len(genders)
            
            df.at[idx, 'female_cast_count'] = female
            df.at[idx, 'male_cast_count'] = male
            female_pct = (female / total * 100) if total > 0 else 0
            df.at[idx, 'female_cast_percentage'] = female_pct
            # Gender balance: 100 = perfect 50/50, 0 = all one gender
            df.at[idx, 'gender_balance_score'] = max(0, 100 - abs(50 - female_pct) * 2)
        
        directors_str = row.get('directors', '')
        if pd.notna(directors_str) and directors_str:
            director = str(directors_str).split(',')[0].strip()
            director_gender = get_gender(director)
            df.at[idx, 'director_gender'] = director_gender
            df.at[idx, 'female_director'] = (director_gender == 1)
        
        if idx % 100 == 0:
            save_checkpoint(cache, cache_file)
        
        time.sleep(0.25)
    
    save_checkpoint(cache, cache_file)
    df.to_csv(output_file, index=False)
    
    # Print stats
    female_directors = df['female_director'].sum()
    avg_female_cast = df[df['female_cast_percentage'] > 0]['female_cast_percentage'].mean()
    
    print(f"\nAdded diversity data to {len(df):} movies")
    print(f"   Saved to: {output_file}")
    
    return df

# STEP 7: YOUTUBE TRAILER METRICS

def youtube_metrics():
    """Step 7: Fetch YouTube trailer metrics (views, likes, comments)"""
    
    if not check_youtube_api_key():
        return None
    
    # Find best input file
    input_file = os.path.join(DATA_DIR, "tmdb_with_sentiment.csv")
    if not os.path.exists(input_file):
        input_file = os.path.join(DATA_DIR, "tmdb_with_urls.csv")
    if not os.path.exists(input_file):
        input_file = os.path.join(DATA_DIR, "tmdb_data.csv")
    
    if not os.path.exists(input_file):
        print(f"Input file not found")
        return None
    
    output_file = os.path.join(DATA_DIR, "youtube_cache_results.csv")
    cache_file = os.path.join(DATA_DIR, "youtube_cache.json")
    
    df = pd.read_csv(input_file, low_memory=False)
    print(f"Loaded {len(df):} movies from {os.path.basename(input_file)}")
    
    if 'trailer_url' not in df.columns:
        print("No trailer_url column found")
        return None
    
    movies_with_trailers = df[df['trailer_url'].notna() & (df['trailer_url'] != '')]
    print(f"   Found {len(movies_with_trailers):} movies with trailer URLs")
    
    cache = {}
    if os.path.exists(cache_file):
        cache = load_checkpoint(cache_file)
    
    def get_youtube_metrics(video_id: str) -> dict:
        """Fetch YouTube video statistics"""
        if video_id in cache:
            return cache[video_id]
        
        try:
            params = {
                "part": "snippet,statistics",
                "id": video_id,
                "key": YOUTUBE_API_KEY
            }
            response = requests.get(YOUTUBE_API_URL, params=params, timeout=10)
            
            if response.status_code != 200:
                cache[video_id] = {}
                return {}
            
            data = response.json()
            if not data.get("items"):
                cache[video_id] = {}
                return {}
            
            item = data["items"][0]
            snippet = item.get("snippet", {})
            stats = item.get("statistics", {})
            
            metrics = {
                "yt_title": snippet.get("title", ""),
                "yt_description": snippet.get("description", "")[:500],
                "yt_tags": ", ".join(snippet.get("tags", [])[:10]) if snippet.get("tags") else "",
                "yt_category_id": snippet.get("categoryId", ""),
                "yt_published_at": snippet.get("publishedAt", ""),
                "yt_view_count": int(stats.get("viewCount", 0)) if stats.get("viewCount") else 0,
                "yt_like_count": int(stats.get("likeCount", 0)) if stats.get("likeCount") else 0,
                "yt_favorite_count": int(stats.get("favoriteCount", 0)) if stats.get("favoriteCount") else 0,
                "yt_comment_count": int(stats.get("commentCount", 0)) if stats.get("commentCount") else 0,
            }
            
            cache[video_id] = metrics
            return metrics
            
        except Exception as e:
            print(f"   Error fetching {video_id}: {e}")
            cache[video_id] = {}
            return {}
    
    results = []
    
    for idx, row in tqdm(movies_with_trailers.iterrows(), total=len(movies_with_trailers), 
                         desc="Fetching YouTube data"):
        video_id = extract_youtube_video_id(row['trailer_url'])
        
        if not video_id:
            continue
        
        metrics = get_youtube_metrics(video_id)
        
        if metrics:
            # Calculate days until release
            days_diff = None
            if metrics.get('yt_published_at') and pd.notna(row.get('release_date')):
                days_diff = days_until_release(metrics['yt_published_at'], row['release_date'])
            
            results.append({
                'id': row.get('id'),
                'title': row.get('title'),
                'trailer_url': row.get('trailer_url'),
                'video_id': video_id,
                'days_until_release': days_diff,
                **metrics
            })
        
        if len(results) % 50 == 0:
            save_checkpoint(cache, cache_file)
        
        time.sleep(0.1)
    
    save_checkpoint(cache, cache_file)
    
    if results:
        result_df = pd.DataFrame(results)
        result_df.to_csv(output_file, index=False)
        print(f"\nFetched YouTube data for {len(results):} trailers")
        print(f"   Saved to: {output_file}")
        
        if 'yt_view_count' in result_df.columns:
            avg_views = result_df['yt_view_count'].mean()
        
        return result_df
    else:
        print("\n No YouTube data collected")
        return None

# STEP 8: FINAL MERGE

def merge_all():
    """Step 8: Merge all data sources into final dataset"""
    
    output_file = os.path.join(DATA_DIR, "complete_data.csv")
    
    # Find best base file
    base_file = os.path.join(DATA_DIR, "tmdb_with_sentiment.csv")
    if not os.path.exists(base_file):
        base_file = os.path.join(DATA_DIR, "tmdb_with_urls.csv")
    
    if not os.path.exists(base_file):
        print(f"Base file not found")
        return None
    
    df = pd.read_csv(base_file, low_memory=False)
    print(f"Loaded base: {len(df):} movies")
    
    # Merge sentiment data
    sentiment_file = os.path.join(DATA_DIR, "rt_sentiment.csv")
    if os.path.exists(sentiment_file):
        sentiment_df = pd.read_csv(sentiment_file)
        df = df.merge(sentiment_df, on='rt_url', how='left', suffixes=('', '_sent'))
        print(f"   Merged sentiment data: {len(sentiment_df):} rows")
    
    # Merge YouTube data
    youtube_file = os.path.join(DATA_DIR, "youtube_cache_results.csv")
    if os.path.exists(youtube_file):
        youtube_df = pd.read_csv(youtube_file)
        if 'id' in df.columns and 'id' in youtube_df.columns:
            youtube_cols = ['id', 'yt_view_count', 'yt_like_count', 'yt_comment_count', 
                           'yt_published_at', 'yt_tags', 'days_until_release']
            youtube_cols = [c for c in youtube_cols if c in youtube_df.columns]
            df = df.merge(youtube_df[youtube_cols], on='id', how='left', suffixes=('', '_yt'))
            print(f"   Merged YouTube data: {len(youtube_df):} rows")
    
    # Add description sentiment
    try:
        from transformers import pipeline
        print("Computing description sentiment...")
        sentiment_pipe = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=-1
        )
        
        def get_description_sentiment(text):
            if pd.isna(text) or not text:
                return None
            try:
                pred = sentiment_pipe(str(text)[:512], truncation=True)[0]
                return pred['score'] if pred['label'] == 'POSITIVE' else -pred['score']
            except:
                return None
        
        df['description_sentiment_score'] = df['overview'].apply(get_description_sentiment)
    except Exception as e:
        pass
    
    df.to_csv(output_file, index=False)
    print_stats(df, "Final Dataset")
    print(f"Saved: {output_file}")
    
    return df

# MAIN

STEPS = {
    1: ("TMDB Data Collection", collect_tmdb),
    2: ("Data Cleaning", clean_data),
    3: ("RT URL Mapping", map_rt_urls),
    4: ("RT Review Scraping", scrape_rt_reviews),
    5: ("Sentiment Analysis", sentiment_analysis),
    6: ("Gender Diversity", add_diversity),
    7: ("YouTube Trailer Metrics", youtube_metrics),
    8: ("Final Merge", merge_all),
}

def main():
    parser = argparse.ArgumentParser(description="Filmlytics Data Pipeline")
    parser.add_argument('--step', type=str, default='all',
                        help='Step(s) to run: "all", single number, or comma-separated (e.g., "1,2,3")')
    parser.add_argument('--skip-api', action='store_true',
                        help='Skip API-dependent steps (use cached data)')
    args = parser.parse_args()
    
    print("\n" + "=" * 80)
    print("FILMLYTICS DATA PIPELINE")
    print("=" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {DATA_DIR}")
    
    if args.step.lower() == 'all':
        steps_to_run = list(STEPS.keys())
    else:
        steps_to_run = [int(s.strip()) for s in args.step.split(',')]
    
    print(f"Steps to run: {steps_to_run}\n")
    
    for step_num in steps_to_run:
        if step_num not in STEPS:
            print(f"Unknown step: {step_num}")
            continue
        
        step_name, step_func = STEPS[step_num]
        
        # Skip API steps if requested
        if args.skip_api and step_num in [1, 4, 6, 7]:
            print(f"\nSkipping Step {step_num}: {step_name} (--skip-api)")
            continue
        
        try:
            step_func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Progress saved.")
            break
        except Exception as e:
            print(f"\nError in Step {step_num}: {e}")
            import traceback
            traceback.print_exc()
            break
    
    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()

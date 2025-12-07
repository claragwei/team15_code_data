
import requests
from datetime import datetime, timedelta
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi


TMDB_BEARER_TOKEN = "eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiJiMDAzMDhjZWMzZWZiMWEwYzUzYzMxYWNmOWZkNzA1NyIsIm5iZiI6MTc2MTA2ODg3NS40NzksInN1YiI6IjY4ZjdjNzRiMjVlMzNjMTdiYzg3ZDViYyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.Q5JNWorEJL9w9rwop3QVePSSzqx90pGN6VB04YAwGAY"
TMDB_BASE_URL = "https://api.themoviedb.org/3"
MONGODB_URI = "mongodb+srv://cinemaniacs:filmlytics@filmlytics.1emhcue.mongodb.net/?appName=filmlytics"


def fetch_upcoming_movies(weeks_ahead=6, include_announed=False):
   
   headers = {
       "accept": "application/json",
       "Authorization": f"Bearer {TMDB_BEARER_TOKEN}"
   }
  
   print(f"\n{'='*70}")
   print(f"Fetching upcoming movies for the next {weeks_ahead} weeks...")
   print(f"{'='*70}")


   today = datetime.now()
   end_date = today + timedelta(weeks=weeks_ahead)


   url = f"{TMDB_BASE_URL}/discover/movie"
   params = {
       "primary_release_date.gte": today.strftime("%Y-%m-%d"),
       "primary_release_date.lte": end_date.strftime("%Y-%m-%d"),
       "sort_by": "popularity.desc",
       "language": "en-US",
       "page": 1
   }


   all_movies = []


   first = requests.get(url, params=params, headers=headers)
   if first.status_code != 200:
       print(f"❌ Error: {first.status_code}")
       return []


   data = first.json()


   total_pages = min(data.get("total_pages", 1), 500)
   all_movies.extend(data.get("results", []))


   print(f"Page 1: {len(data.get('results', []))} movies")
   print(f"TMDB reports {total_pages} total pages for this window.")


   for page in range(2, total_pages + 1):
       params["page"] = page
       res = requests.get(url, params=params, headers=headers)


       if res.status_code != 200:
           print(f"❌ Error on page {page}: {res.status_code}")
           break


       results = res.json().get("results", [])
       if not results:
           break


       all_movies.extend(results)
       print(f"Page {page}: {len(results)} movies")


   print(f"\nTOTAL MOVIES RETURNED: {len(all_movies)}\n")
   return all_movies


def fetch_movie_details(movie_id, headers):
   url = f"{TMDB_BASE_URL}/movie/{movie_id}"
   params = {
       'language': 'en-US',
       'append_to_response': 'credits,keywords,videos'
   }
  
   response = requests.get(url, params=params, headers=headers)
   if response.status_code == 200:
       return response.json()
   return None


def format_for_mongodb(movie_data):
  
   # Extract credits
   credits = movie_data.get('credits', {})
   cast = credits.get('cast', [])
   crew = credits.get('crew', [])
  
   directors = [c['name'] for c in crew if c['job'] == 'Director']
   cast_list = [c['name'] for c in cast[:10]]
  
   # Extract cast genders
   cast_genders = [c.get('gender', 0) for c in cast[:10]]
   female_count = sum(1 for g in cast_genders if g == 1)
   male_count = sum(1 for g in cast_genders if g == 2)
   total = female_count + male_count
  
   # Format document
   doc = {
       'tmdb_id': movie_data['id'],
       'title': movie_data['title'],
      
       'release_info': {
           'tmdb_release_date': movie_data.get('release_date'),
           'days_until_release': None  # Will be calculated later
       },
      
       'production': {
           'budget': movie_data.get('budget', 0),
           'runtime': movie_data.get('runtime'),
           'genres': [g['name'] for g in movie_data.get('genres', [])],
           'production_companies': [c['name'] for c in movie_data.get('production_companies', [])],
           'production_countries': [c['name'] for c in movie_data.get('production_countries', [])]
       },
      
       'people': {
           'cast': cast_list,
           'directors': directors
       },
      
       'tmdb_metrics': {
           'vote_count': movie_data.get('vote_count', 0),
           'vote_average': movie_data.get('vote_average', 0),
           'is_successful': None  # Unknown for upcoming movies
       },
      
       'content': {
           'overview': movie_data.get('overview', ''),
           'poster_url': f"https://image.tmdb.org/t/p/w500{movie_data['poster_path']}" if movie_data.get('poster_path') else None
       },
      
       'diversity': {
           'female_cast_count': female_count,
           'male_cast_count': male_count,
           'female_cast_percentage': (female_count / total * 100) if total > 0 else 0,
           'gender_balance_score': None,
           'director_gender': crew[0].get('gender', 0) if directors and crew else 0,
           'female_director': any(c.get('gender') == 1 and c['job'] == 'Director' for c in crew),
           'cast_genders': cast_genders
       },
      
       'rotten_tomatoes': {
           'has_rt_url': False,
           'critic_score': None,
           'audience_score': None
       },
      
       'sentiment': {
           'description_sentiment_score': None
       },
      
       'trailer': {
           'trailer_url_youtube': None,
           'metrics': {
               'view_count': None,
               'like_count': None,
               'comment_count': None
           },
           'published_at': None,
           'official': False
       },
      
       'is_upcoming': True  # Flag to identify upcoming movies
   }
  
   return doc


def main():
   print("="*70)
   print("ADDING UPCOMING MOVIES TO MONGODB")
   print("="*70)
  
   # Connect to MongoDB
   client = MongoClient(MONGODB_URI, server_api=ServerApi('1'), tlsCAFile=certifi.where())
   db = client['cinemaniacs']
   collection = db['movies']
  
   headers = {
       "accept": "application/json",
       "Authorization": f"Bearer {TMDB_BEARER_TOKEN}"
   }
  
   # Fetch upcoming movies
   print("\nFetching upcoming movies from TMDB...")
   upcoming = fetch_upcoming_movies(weeks_ahead=6)
   print(f"✅ Found {len(upcoming)} upcoming movies\n")
  
   added_count = 0
   skipped_count = 0
  
   for i, movie_basic in enumerate(upcoming, 1):
       movie_id = movie_basic['id']
       title = movie_basic['title']
      
       print(f"[{i}/{len(upcoming)}] {title}")
      
       # Check if already in database
       existing = collection.find_one({'tmdb_id': movie_id})
       if existing:
           print(f"  ⏭️  Already in database, skipping\n")
           skipped_count += 1
           continue
      
       # Fetch full details
       movie_details = fetch_movie_details(movie_id, headers)
       if not movie_details:
           print(f"  ❌ Could not fetch details\n")
           continue
      
       # Format for MongoDB
       doc = format_for_mongodb(movie_details)
      
       # Insert into database
       try:
           collection.insert_one(doc)
           print(f"  ✅ Added to database\n")
           added_count += 1
       except Exception as e:
           print(f"  ❌ Error: {e}\n")
  

if __name__ == "__main__":
   main()
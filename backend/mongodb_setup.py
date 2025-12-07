import pymongo
from pymongo import MongoClient
import pandas as pd
import numpy as np
from datetime import datetime
import json
import os
from typing import Dict, List, Any

class MovieDatabaseManager:
    
    def __init__(self, connection_string: str, database_name: str = "cinemaniacs"):
        
        self.client = MongoClient(connection_string)
        self.db = self.client[database_name]
        print(f"✓ Connected to MongoDB database: {database_name}")
    
    def prepare_movie_document(self, row: pd.Series) -> Dict[str, Any]:
        
        
        # Helper function to parse lists from strings
        def parse_list(value):
            if pd.isna(value) or value == '':
                return []
            if isinstance(value, str):
                # Handle comma-separated values
                return [item.strip() for item in value.split(',') if item.strip()]
            return value if isinstance(value, list) else []
        
        # Helper function to safely convert to float
        def safe_float(value):
            if pd.isna(value) or value == '':
                return None
            try:
                return float(value)
            except:
                return None
        
        # Helper function to safely convert to int
        def safe_int(value):
            if pd.isna(value) or value == '':
                return None
            try:
                return int(value)
            except:
                return None
        
        # Build the document structure
        document = {
            # Core Identifiers
            "tmdb_id": safe_int(row.get('id')),
            "title": row.get('title', ''),
            
            # Release Information
            "release_info": {
                "tmdb_release_date": row.get('release_date_x'),
                "youtube_release_date": row.get('release_date_y'),
                "days_until_release": safe_int(row.get('days_until_release')),
                "before_release": bool(row.get('before_release')) if pd.notna(row.get('before_release')) else None
            },
            
            # Production Details
            "production": {
                "budget": safe_float(row.get('budget')),
                "runtime": safe_int(row.get('runtime')),
                "genres": parse_list(row.get('genres')),
                "production_companies": parse_list(row.get('production_companies')),
                "production_countries": parse_list(row.get('production_countries'))
            },
            
            # Cast & Crew
            "people": {
                "cast": parse_list(row.get('cast')),
                "directors": parse_list(row.get('directors'))
            },
            
            # TMDB Ratings & Engagement
            "tmdb_metrics": {
                "vote_count": safe_int(row.get('vote_count')),
                "vote_average": safe_float(row.get('vote_average')),
                "is_successful": bool(row.get('is_successful')) if pd.notna(row.get('is_successful')) else None
            },
            
            # Rotten Tomatoes Data
            "rotten_tomatoes": {
                "has_rt_url": bool(row.get('has_rt_url')) if pd.notna(row.get('has_rt_url')) else False,
                "rt_url": row.get('rt_url') if pd.notna(row.get('rt_url')) else None,
                "critic_score": row.get('critic_score') if pd.notna(row.get('critic_score')) else None,
                "audience_score": row.get('audience_score') if pd.notna(row.get('audience_score')) else None,
                "review_sentiment": safe_float(row.get('review_sentiment'))
            },
            
            # Sentiment Analysis
            "sentiment": {
                "description_sentiment_score": safe_float(row.get('description_sentiment_score'))
            },
            
            # YouTube Trailer Data
            "trailer": {
                "trailer_url_tmdb": row.get('trailer_url_x') if pd.notna(row.get('trailer_url_x')) else None,
                "trailer_url_youtube": row.get('trailer_url_y') if pd.notna(row.get('trailer_url_y')) else None,
                "video_name": row.get('video_name') if pd.notna(row.get('video_name')) else None,
                "published_at": row.get('published_at') if pd.notna(row.get('published_at')) else None,
                "official": bool(row.get('official')) if pd.notna(row.get('official')) else None,
                "tags": parse_list(row.get('tags')),
                "description": row.get('description') if pd.notna(row.get('description')) else None,
                "category_id": safe_int(row.get('category_id')),
                "metrics": {
                    "view_count": safe_int(row.get('view_count')),
                    "like_count": safe_int(row.get('like_count')),
                    "comment_count": safe_int(row.get('comment_count')),
                    "favorite_count": safe_int(row.get('favorite_count'))
                }
            },
            
            # Content
            "content": {
                "overview": row.get('overview', ''),
                "poster_url": row.get('poster_url') if pd.notna(row.get('poster_url')) else None
            },
            
            # Metadata
            "metadata": {
                "created_at": datetime.utcnow(),
                "last_updated": datetime.utcnow()
            }
        }
        
        return document
    
    def load_csv_to_mongodb(self, csv_path: str, batch_size: int = 1000):
        """
        Load CSV data into MongoDB with progress tracking
        
        Args:
            csv_path: Path to your complete CSV file
            batch_size: Number of documents to insert at once (for performance)
        """
        print(f"\n📂 Loading data from: {csv_path}")
        
        # Read CSV
        df = pd.read_csv(csv_path, low_memory=False)
        total_rows = len(df)
        print(f"✓ Found {total_rows:,} movies in CSV")
        
        # Get or create the movies collection
        collection = self.db.movies
        
        # Clear existing data (optional - comment out if you want to keep old data)
        print("\n⚠️  Clearing existing data...")
        collection.delete_many({})
        
        # Process and insert in batches
        print(f"\n📥 Inserting documents in batches of {batch_size}...")
        documents = []
        inserted_count = 0
        
        for idx, row in df.iterrows():
            try:
                doc = self.prepare_movie_document(row)
                documents.append(doc)
                
                # Insert when batch is full
                if len(documents) >= batch_size:
                    collection.insert_many(documents)
                    inserted_count += len(documents)
                    print(f"   Inserted {inserted_count:,}/{total_rows:,} documents ({(inserted_count/total_rows)*100:.1f}%)")
                    documents = []
                    
            except Exception as e:
                print(f"⚠️  Error processing row {idx}: {e}")
                continue
        
        # Insert remaining documents
        if documents:
            collection.insert_many(documents)
            inserted_count += len(documents)
        
        print(f"\n✓ Successfully inserted {inserted_count:,} documents!")
        return inserted_count
    
    def create_indexes(self):
        """
        Create indexes for efficient querying
        Critical for GNN, website queries, and model training
        """
        print("\n🔍 Creating indexes for optimized queries...")
        collection = self.db.movies
        
        # Core indexes
        indexes = [
            ("tmdb_id", pymongo.ASCENDING),  # Primary lookup
            ("title", pymongo.TEXT),  # Text search for website
            ("production.genres", pymongo.ASCENDING),  # Genre filtering
            ("tmdb_metrics.is_successful", pymongo.ASCENDING),  # Success filtering
            ("release_info.tmdb_release_date", pymongo.ASCENDING),  # Time-based queries
            ("rotten_tomatoes.has_rt_url", pymongo.ASCENDING),  # RT data filtering
            ("people.directors", pymongo.ASCENDING),  # For GNN edges (director connections)
            ("people.cast", pymongo.ASCENDING),  # For GNN edges (actor connections)
            ("production.production_companies", pymongo.ASCENDING),  # Company connections
        ]
        
        for field, direction in indexes:
            try:
                collection.create_index([(field, direction)])
                print(f"   ✓ Created index on: {field}")
            except Exception as e:
                print(f"   ⚠️  Could not create index on {field}: {e}")
        
        # Compound index for XGBoost training (train/test split by date)
        try:
            collection.create_index([
                ("release_info.tmdb_release_date", pymongo.ASCENDING),
                ("tmdb_metrics.is_successful", pymongo.ASCENDING)
            ])
            print("   ✓ Created compound index for train/test split")
        except Exception as e:
            print(f"   ⚠️  Could not create compound index: {e}")
        
        print("\n✓ Index creation complete!")
    
    def get_database_stats(self):
        """
        Print useful statistics about your database
        """
        print("\n" + "="*60)
        print("📊 DATABASE STATISTICS")
        print("="*60)
        
        collection = self.db.movies
        
        # Total count
        total = collection.count_documents({})
        print(f"\nTotal Movies: {total:,}")
        
        # Successful movies
        successful = collection.count_documents({"tmdb_metrics.is_successful": True})
        print(f"Successful Movies (vote_avg >= 6): {successful:,} ({(successful/total)*100:.1f}%)")
        
        # Movies with RT data
        with_rt = collection.count_documents({"rotten_tomatoes.has_rt_url": True})
        print(f"Movies with Rotten Tomatoes URL: {with_rt:,} ({(with_rt/total)*100:.1f}%)")
        
        # Movies with trailer data
        with_trailer = collection.count_documents({"trailer.trailer_url_youtube": {"$ne": None}})
        print(f"Movies with YouTube Trailer: {with_trailer:,} ({(with_trailer/total)*100:.1f}%)")
        
        # Genre distribution (top 5)
        print("\nTop 5 Genres:")
        pipeline = [
            {"$unwind": "$production.genres"},
            {"$group": {"_id": "$production.genres", "count": {"$sum": 1}}},
            {"$sort": {"count": -1}},
            {"$limit": 5}
        ]
        for doc in collection.aggregate(pipeline):
            print(f"   {doc['_id']}: {doc['count']:,}")
        
        # Date range
        oldest = collection.find_one(
            {"release_info.tmdb_release_date": {"$ne": None}},
            sort=[("release_info.tmdb_release_date", pymongo.ASCENDING)]
        )
        newest = collection.find_one(
            {"release_info.tmdb_release_date": {"$ne": None}},
            sort=[("release_info.tmdb_release_date", pymongo.DESCENDING)]
        )
        if oldest and newest:
            print(f"\nRelease Date Range: {oldest['release_info']['tmdb_release_date']} to {newest['release_info']['tmdb_release_date']}")
        
        print("\n" + "="*60 + "\n")
    
    def example_queries(self):
        """
        Demonstrate some useful queries for your project
        """
        print("\n💡 EXAMPLE QUERIES FOR YOUR PROJECT\n")
        collection = self.db.movies
        
        # 1. Get all successful action movies for training
        print("1. Query: Get successful Action movies")
        query = {
            "production.genres": "Action",
            "tmdb_metrics.is_successful": True,
            "tmdb_metrics.vote_count": {"$gte": 10}  # Minimum votes
        }
        count = collection.count_documents(query)
        print(f"   Found {count:,} movies\n")
        
        # 2. Movies for GNN - those with cast/director info
        print("2. Query: Movies with cast and director data (for GNN)")
        query = {
            "people.cast": {"$ne": []},
            "people.directors": {"$ne": []}
        }
        count = collection.count_documents(query)
        print(f"   Found {count:,} movies\n")
        
        # 3. Movies with complete feature set (for XGBoost)
        print("3. Query: Movies with complete features")
        query = {
            "tmdb_metrics.vote_average": {"$ne": None},
            "production.budget": {"$ne": None},
            "production.runtime": {"$ne": None},
            "sentiment.description_sentiment_score": {"$ne": None}
        }
        count = collection.count_documents(query)
        print(f"   Found {count:,} movies\n")
        
        # 4. Recent movies for testing
        print("4. Query: Movies from 2020 onwards (test set)")
        query = {
            "release_info.tmdb_release_date": {"$gte": "2020-01-01"}
        }
        count = collection.count_documents(query)
        print(f"   Found {count:,} movies\n")
        
        # 5. Example: Get a sample movie with all data
        print("5. Sample Movie Document:")
        sample = collection.find_one(
            {"rotten_tomatoes.has_rt_url": True},
            {"title": 1, "production.genres": 1, "tmdb_metrics": 1, "rotten_tomatoes": 1}
        )
        if sample:
            print(f"   Title: {sample['title']}")
            print(f"   Genres: {sample['production']['genres']}")
            print(f"   TMDB Rating: {sample['tmdb_metrics']['vote_average']}")
            print(f"   RT Critic Score: {sample['rotten_tomatoes']['critic_score']}")


def main():
    """
    Main setup function - follow these steps!
    """
    print("\n" + "="*60)
    print("🎬 CINEMANIACS MONGODB SETUP")
    print("="*60 + "\n")
    
    # STEP 1: Set your MongoDB connection string
    # For MongoDB Atlas: Get this from your Atlas dashboard
    # Format: mongodb+srv://<username>:<password>@<cluster>.mongodb.net/?retryWrites=true&w=majority
    
    CONNECTION_STRING = "connection_string"
    CSV_PATH = ""  # Update this path
    
    
    try:
        # Initialize database manager
        db_manager = MovieDatabaseManager(CONNECTION_STRING)
        
        # Load CSV data
        db_manager.load_csv_to_mongodb(CSV_PATH)
        
        # Create indexes
        db_manager.create_indexes()
        
        # Show statistics
        db_manager.get_database_stats()
        
        # Show example queries
        db_manager.example_queries()
        
        
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("\nTroubleshooting:")
        print("- Check your MongoDB connection string")
        print("- Verify CSV file path")
        print("- Ensure MongoDB Atlas IP whitelist includes your IP")


if __name__ == "__main__":
    main()
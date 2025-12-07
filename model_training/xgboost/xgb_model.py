import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import certifi
import os
warnings.filterwarnings('ignore')

# Create output folder
output_folder = 'visualizations'
os.makedirs(output_folder, exist_ok=True)

# Connect to MongoDB
client = MongoClient(
    "mongodb+srv://cinemaniacs:filmlytics@filmlytics.1emhcue.mongodb.net/?appName=filmlytics",
    server_api=ServerApi('1'),
    tlsCAFile=certifi.where()
)

db = client['cinemaniacs']
collection = db['movies']

# Load data
data = list(collection.find())

# Flatten nested structure into DataFrame
records = []
for doc in data:
    record = {
        'tmdb_id': doc.get('tmdb_id'),
        'title': doc.get('title'),
        
        # Release info
        'release_date': doc.get('release_info', {}).get('tmdb_release_date'),
        'days_until_release': doc.get('release_info', {}).get('days_until_release'),
        
        # Production
        'budget': doc.get('production', {}).get('budget', 0),
        'runtime': doc.get('production', {}).get('runtime'),
        'genres': doc.get('production', {}).get('genres', []),
        'production_companies': doc.get('production', {}).get('production_companies', []),
        'production_countries': doc.get('production', {}).get('production_countries', []),
        
        # People
        'cast': doc.get('people', {}).get('cast', []),
        'directors': doc.get('people', {}).get('directors', []),
        
        # TMDB metrics
        'vote_count': doc.get('tmdb_metrics', {}).get('vote_count'),
        'vote_average': doc.get('tmdb_metrics', {}).get('vote_average'),
        'is_successful': doc.get('tmdb_metrics', {}).get('is_successful'),
        
        # Rotten Tomatoes
        'has_rt_url': doc.get('rotten_tomatoes', {}).get('has_rt_url', False),
        'critic_score': doc.get('rotten_tomatoes', {}).get('critic_score'),
        'audience_score': doc.get('rotten_tomatoes', {}).get('audience_score'),
        
        # Sentiment
        'description_sentiment': doc.get('sentiment', {}).get('description_sentiment_score'),
        
        # Trailer metrics
        'view_count': doc.get('trailer', {}).get('metrics', {}).get('view_count'),
        'like_count': doc.get('trailer', {}).get('metrics', {}).get('like_count'),
        'comment_count': doc.get('trailer', {}).get('metrics', {}).get('comment_count'),
        'trailer_published_at': doc.get('trailer', {}).get('published_at'),
        'is_official_trailer': doc.get('trailer', {}).get('official'),

        # Diversity metrics
        'female_cast_count': doc.get('diversity', {}).get('female_cast_count'),
        'male_cast_count': doc.get('diversity', {}).get('male_cast_count'),
        'female_cast_percentage': doc.get('diversity', {}).get('female_cast_percentage'),
        'gender_balance_score': doc.get('diversity', {}).get('gender_balance_score'),
        'director_gender': doc.get('diversity', {}).get('director_gender'),
        'female_director': doc.get('diversity', {}).get('female_director'),
    }
    records.append(record)

df = pd.DataFrame(records)

# Data preprocessing
df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
df['trailer_published_at'] = pd.to_datetime(df['trailer_published_at'], errors='coerce')

# Feature engineering

# Basic features
df['budget_log'] = np.log1p(df['budget'])
df['has_budget'] = (df['budget'] > 0).astype(int)
df['has_runtime'] = df['runtime'].notna().astype(int)

# Temporal features
df['release_year'] = df['release_date'].dt.year
df['release_month'] = df['release_date'].dt.month
df['release_quarter'] = df['release_date'].dt.quarter
df['is_summer_release'] = df['release_month'].isin([6, 7, 8]).astype(int)
df['is_holiday_release'] = df['release_month'].isin([11, 12]).astype(int)

# Genre features
df['genre_count'] = df['genres'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['has_genre'] = (df['genre_count'] > 0).astype(int)

# Get top genres
all_genres = []
for genres in df['genres']:
    if isinstance(genres, list):
        all_genres.extend(genres)
top_genres = pd.Series(all_genres).value_counts().head(5).index.tolist()

for genre in top_genres:
    df[f'is_{genre.lower().replace(" ", "_")}'] = df['genres'].apply(
        lambda x: 1 if isinstance(x, list) and genre in x else 0
    )

# People features - Basic counts
df['cast_count'] = df['cast'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['director_count'] = df['directors'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['has_cast'] = (df['cast_count'] > 0).astype(int)
df['has_director'] = (df['director_count'] > 0).astype(int)

# Advanced Cast Features
# Extract all cast members and calculate their average vote_average
cast_performance = {}
for idx, row in df.iterrows():
    if isinstance(row['cast'], list) and pd.notna(row['vote_average']):
        for actor in row['cast']:
            if actor not in cast_performance:
                cast_performance[actor] = []
            cast_performance[actor].append(row['vote_average'])

# Calculate average performance for each actor
cast_avg_rating = {actor: np.mean(ratings) for actor, ratings in cast_performance.items() if len(ratings) >= 3}

# Get top 20 actors by number of appearances
cast_counts = pd.Series([actor for cast in df['cast'] if isinstance(cast, list) for actor in cast]).value_counts()
top_actors = cast_counts.head(20).index.tolist()

# Create features for top actors
for actor in top_actors:
    safe_name = actor.lower().replace(' ', '_').replace('.', '').replace('-', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')[:40]
    col_name = f'actor_{safe_name}'
    df[col_name] = df['cast'].apply(lambda x: 1 if isinstance(x, list) and actor in x else 0)

# Cast quality metrics
df['cast_avg_rating'] = df['cast'].apply(
    lambda x: np.mean([cast_avg_rating.get(actor, 0) for actor in x])
    if isinstance(x, list) and len(x) > 0 else 0
)
df['cast_max_rating'] = df['cast'].apply(
    lambda x: max([cast_avg_rating.get(actor, 0) for actor in x])
    if isinstance(x, list) and len(x) > 0 else 0
)
df['has_star_actor'] = df['cast'].apply(
    lambda x: 1 if isinstance(x, list) and any(actor in top_actors for actor in x) else 0
)
df['star_actor_count'] = df['cast'].apply(
    lambda x: sum(1 for actor in x if actor in top_actors) if isinstance(x, list) else 0
)

# Advanced Director Features
# Extract all directors and calculate their average vote_average
director_performance = {}
for idx, row in df.iterrows():
    if isinstance(row['directors'], list) and pd.notna(row['vote_average']):
        for director in row['directors']:
            if director not in director_performance:
                director_performance[director] = []
            director_performance[director].append(row['vote_average'])

# Calculate average performance for each director
director_avg_rating = {director: np.mean(ratings) for director, ratings in director_performance.items() if len(ratings) >= 2}

# Get top 15 directors by number of movies
director_counts = pd.Series([director for directors in df['directors'] if isinstance(directors, list) for director in directors]).value_counts()
top_directors = director_counts.head(15).index.tolist()

# Create features for top directors
for director in top_directors:
    safe_name = director.lower().replace(' ', '_').replace('.', '').replace('-', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')[:40]
    col_name = f'director_{safe_name}'
    df[col_name] = df['directors'].apply(lambda x: 1 if isinstance(x, list) and director in x else 0)

# Director quality metrics
df['director_avg_rating'] = df['directors'].apply(
    lambda x: np.mean([director_avg_rating.get(director, 0) for director in x])
    if isinstance(x, list) and len(x) > 0 else 0
)
df['director_max_rating'] = df['directors'].apply(
    lambda x: max([director_avg_rating.get(director, 0) for director in x])
    if isinstance(x, list) and len(x) > 0 else 0
)
df['has_top_director'] = df['directors'].apply(
    lambda x: 1 if isinstance(x, list) and any(director in top_directors for director in x) else 0
)
df['is_solo_director'] = (df['director_count'] == 1).astype(int)
df['is_multi_director'] = (df['director_count'] > 1).astype(int)

# Production Country Features
# Extract all production countries
all_countries = []
for countries in df['production_countries']:
    if isinstance(countries, list):
        all_countries.extend(countries)

# Get top 10 production countries
country_counts = pd.Series(all_countries).value_counts()
top_countries = country_counts.head(10).index.tolist()

# Create features for top countries
for country in top_countries:
    safe_name = country.lower().replace(' ', '_').replace('.', '').replace('-', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')[:40]
    col_name = f'country_{safe_name}'
    df[col_name] = df['production_countries'].apply(
        lambda x: 1 if isinstance(x, list) and country in x else 0
    )

# Production features
df['production_company_count'] = df['production_companies'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['production_country_count'] = df['production_countries'].apply(lambda x: len(x) if isinstance(x, list) else 0)
df['is_us_production'] = df['production_countries'].apply(
    lambda x: 1 if isinstance(x, list) and 'United States of America' in x else 0
)
df['is_international_coproduction'] = (df['production_country_count'] > 1).astype(int)
df['is_single_country'] = (df['production_country_count'] == 1).astype(int)

# Diversity Features
# Fill missing diversity values with 0
df['female_cast_count'] = df['female_cast_count'].fillna(0)
df['male_cast_count'] = df['male_cast_count'].fillna(0)
df['female_cast_percentage'] = df['female_cast_percentage'].fillna(0)
df['gender_balance_score'] = df['gender_balance_score'].fillna(0)
df['director_gender'] = df['director_gender'].fillna(0)
df['female_director'] = df['female_director'].fillna(False).astype(int)

# Has diversity data indicator
df['has_diversity_data'] = ((df['female_cast_count'] > 0) | (df['male_cast_count'] > 0)).astype(int)

# Gender diversity categories
df['male_dominated_cast'] = (df['female_cast_percentage'] < 30).astype(int)
df['female_dominated_cast'] = (df['female_cast_percentage'] > 70).astype(int)
df['balanced_cast'] = ((df['female_cast_percentage'] >= 40) & (df['female_cast_percentage'] <= 60)).astype(int)

# Cast size from diversity data
df['total_cast_from_diversity'] = df['female_cast_count'] + df['male_cast_count']

# Gender ratio (male to female)
df['male_to_female_ratio'] = np.where(
    df['female_cast_count'] > 0,
    df['male_cast_count'] / df['female_cast_count'],
    0
)

# High gender balance indicator
df['high_gender_balance'] = (df['gender_balance_score'] >= 70).astype(int)
df['low_gender_balance'] = (df['gender_balance_score'] < 40).astype(int)

# Director gender categories (0=unknown, 1=female, 2=male based on typical encoding)
df['has_known_director_gender'] = (df['director_gender'] > 0).astype(int)
df['male_director'] = (df['director_gender'] == 2).astype(int)

# Sentiment features (expanded)
df['has_sentiment'] = df['description_sentiment'].notna().astype(int)
df['sentiment_positive'] = (df['description_sentiment'] > 0).astype(int)
df['sentiment_negative'] = (df['description_sentiment'] < 0).astype(int)

# Sentiment magnitude and strength
df['sentiment_magnitude'] = np.abs(df['description_sentiment'].fillna(0))
df['sentiment_strength'] = df['description_sentiment'].fillna(0) ** 2

# Sentiment categories
df['very_positive'] = (df['description_sentiment'] > 0.5).astype(int)
df['very_negative'] = (df['description_sentiment'] < -0.5).astype(int)
df['neutral_sentiment'] = (np.abs(df['description_sentiment'].fillna(0)) < 0.1).astype(int)
df['moderate_positive'] = ((df['description_sentiment'] > 0.1) & (df['description_sentiment'] <= 0.5)).astype(int)
df['moderate_negative'] = ((df['description_sentiment'] < -0.1) & (df['description_sentiment'] >= -0.5)).astype(int)

# Trailer features
df['has_trailer_data'] = df['view_count'].notna().astype(int)
df['view_count'] = df['view_count'].fillna(0)
df['like_count'] = df['like_count'].fillna(0)
df['comment_count'] = df['comment_count'].fillna(0)

# Engagement rates
df['like_rate'] = np.where(df['view_count'] > 0, df['like_count'] / df['view_count'], 0)
df['comment_rate'] = np.where(df['view_count'] > 0, df['comment_count'] / df['view_count'], 0)

# Rotten Tomatoes features
df['has_critic_score'] = df['critic_score'].notna().astype(int)
df['has_audience_score'] = df['audience_score'].notna().astype(int)

# Binning features
# Budget categories
budget_bins = [0, 1e6, 10e6, 50e6, 100e6, np.inf]
budget_labels = ['micro', 'low', 'medium', 'high', 'blockbuster']
df['budget_tier'] = pd.cut(df['budget'], bins=budget_bins, labels=budget_labels)

for tier in budget_labels:
    df[f'budget_{tier}'] = (df['budget_tier'] == tier).astype(int)

# Runtime categories
runtime_bins = [0, 80, 100, 120, 150, np.inf]
runtime_labels = ['short', 'standard', 'long', 'very_long', 'marathon']
df['runtime_category'] = pd.cut(df['runtime'].fillna(0), bins=runtime_bins, labels=runtime_labels)

for cat in runtime_labels:
    df[f'runtime_{cat}'] = (df['runtime_category'] == cat).astype(int)

# Engagement tiers (only for movies with trailer data)
df['engagement_tier'] = 'none'
has_views = df['view_count'] > 0
if has_views.sum() > 0:
    view_quantiles = df.loc[has_views, 'view_count'].quantile([0.2, 0.4, 0.6, 0.8])
    df.loc[has_views, 'engagement_tier'] = pd.cut(
        df.loc[has_views, 'view_count'],
        bins=[0] + view_quantiles.tolist() + [np.inf],
        labels=['very_low', 'low', 'medium', 'high', 'viral']
    ).astype(str)

for tier in ['none', 'very_low', 'low', 'medium', 'high', 'viral']:
    df[f'engagement_{tier}'] = (df['engagement_tier'] == tier).astype(int)

# Release year bins (by decade)
df['release_decade'] = (df['release_year'] // 10) * 10
decade_dummies = pd.get_dummies(df['release_decade'], prefix='decade')
df = pd.concat([df, decade_dummies], axis=1)

# Extract all production companies
all_companies = []
for companies in df['production_companies']:
    if isinstance(companies, list):
        all_companies.extend(companies)

# Get top 15 production companies by frequency
top_companies = pd.Series(all_companies).value_counts().head(15)

# Create dummy variables for top companies
for company in top_companies.index:
    # Create safe column name
    safe_name = company.lower().replace(' ', '_').replace('.', '').replace('-', '_')
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c == '_')[:40]  # Limit length
    col_name = f'company_{safe_name}'

    df[col_name] = df['production_companies'].apply(
        lambda x: 1 if isinstance(x, list) and company in x else 0
    )

# Time decay features
def exponential_decay(value, days_before_release, half_life=30):
    """Apply exponential decay based on days before release"""
    if pd.isna(value) or pd.isna(days_before_release) or days_before_release <= 0:
        return 0
    decay_rate = np.log(2) / half_life
    return value * np.exp(-decay_rate * days_before_release)

def recency_weight(days_before_release, peak_days=14):
    """Give more weight to trailers released near optimal timing"""
    if pd.isna(days_before_release):
        return 0
    return np.exp(-((days_before_release - peak_days) ** 2) / (2 * (peak_days ** 2)))

# Apply time decay only to movies with trailer data
df['views_exp_decay_30'] = df.apply(
    lambda row: exponential_decay(row['view_count'], row['days_until_release'], 30)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) else 0, axis=1
)

df['likes_exp_decay_30'] = df.apply(
    lambda row: exponential_decay(row['like_count'], row['days_until_release'], 30)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) else 0, axis=1
)

df['views_recency_weighted'] = df.apply(
    lambda row: row['view_count'] * recency_weight(row['days_until_release'], 14)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) else 0, axis=1
)

df['likes_recency_weighted'] = df.apply(
    lambda row: row['like_count'] * recency_weight(row['days_until_release'], 14)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) else 0, axis=1
)

df['views_per_day'] = df.apply(
    lambda row: row['view_count'] / (np.abs(row['days_until_release']) + 1)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) and row['days_until_release'] != 0 else 0, axis=1
)

df['likes_per_day'] = df.apply(
    lambda row: row['like_count'] / (np.abs(row['days_until_release']) + 1)
    if row['has_trailer_data'] and pd.notna(row['days_until_release']) and row['days_until_release'] != 0 else 0, axis=1
)

# Select features for modeling
feature_columns = [
    # Basic
    'budget', 'budget_log', 'has_budget', 'runtime', 'has_runtime',

    # Temporal
    'release_year', 'release_month', 'release_quarter',
    'is_summer_release', 'is_holiday_release',

    # Genre
    'genre_count', 'has_genre',

    # People - Basic
    'cast_count', 'director_count', 'has_cast', 'has_director',

    # Cast - Advanced
    'cast_avg_rating', 'cast_max_rating', 'has_star_actor', 'star_actor_count',

    # Director - Advanced
    'director_avg_rating', 'director_max_rating', 'has_top_director',
    'is_solo_director', 'is_multi_director',

    # Production
    'production_company_count', 'production_country_count', 'is_us_production',
    'is_international_coproduction', 'is_single_country',

    # Diversity
    'has_diversity_data', 'female_cast_count', 'male_cast_count',
    'female_cast_percentage', 'gender_balance_score',
    'female_director', 'male_director', 'has_known_director_gender',
    'male_dominated_cast', 'female_dominated_cast', 'balanced_cast',
    'total_cast_from_diversity', 'male_to_female_ratio',
    'high_gender_balance', 'low_gender_balance',

    # Sentiment (expanded)
    'has_sentiment', 'sentiment_positive', 'sentiment_negative',
    'sentiment_magnitude', 'sentiment_strength',
    'very_positive', 'very_negative', 'neutral_sentiment',
    'moderate_positive', 'moderate_negative',

    # Trailer (original)
    'has_trailer_data', 'view_count', 'like_count', 'comment_count',
    'like_rate', 'comment_rate',

    # Time decay features
    'views_exp_decay_30', 'likes_exp_decay_30',
    'views_recency_weighted', 'likes_recency_weighted',
    'views_per_day', 'likes_per_day',

    # Rotten Tomatoes
    'has_rt_url', 'has_critic_score', 'has_audience_score',
]

# Add genre dummy variables
for genre in top_genres:
    col_name = f'is_{genre.lower().replace(" ", "_")}'
    if col_name in df.columns:
        feature_columns.append(col_name)

# Add budget tier dummies
for tier in ['micro', 'low', 'medium', 'high', 'blockbuster']:
    if f'budget_{tier}' in df.columns:
        feature_columns.append(f'budget_{tier}')

# Add runtime category dummies
for cat in ['short', 'standard', 'long', 'epic', 'marathon']:
    if f'runtime_{cat}' in df.columns:
        feature_columns.append(f'runtime_{cat}')

# Add engagement tier dummies
for tier in ['none', 'very_low', 'low', 'medium', 'high', 'viral']:
    if f'engagement_{tier}' in df.columns:
        feature_columns.append(f'engagement_{tier}')

# Add decade dummies
decade_cols = [col for col in df.columns if col.startswith('decade_')]
feature_columns.extend(decade_cols)

# Add production company dummies
company_cols = [col for col in df.columns if col.startswith('company_')]
feature_columns.extend(company_cols)

# Add actor dummies
actor_cols = [col for col in df.columns if col.startswith('actor_')]
feature_columns.extend(actor_cols)

# Add director dummies
director_cols = [col for col in df.columns if col.startswith('director_')]
feature_columns.extend(director_cols)

# Add country dummies
country_cols = [col for col in df.columns if col.startswith('country_')]
feature_columns.extend(country_cols)

# Prepare data first
df_model = df[df['is_successful'].notna() & df['release_date'].notna()].copy()
df_model = df_model.sort_values('release_date').reset_index(drop=True)

# Validate feature columns exist in df_model
valid_feature_columns = []
for col in feature_columns:
    if col in df_model.columns:
        valid_feature_columns.append(col)
    else:
        print(f"Warning: Feature '{col}' not found in dataframe")

feature_columns = valid_feature_columns

# Check for duplicate feature names
if len(feature_columns) != len(set(feature_columns)):
    from collections import Counter
    duplicates = [item for item, count in Counter(feature_columns).items() if count > 1]
    # Remove duplicates
    feature_columns = list(dict.fromkeys(feature_columns))

X = df_model[feature_columns].copy()
y = df_model['vote_average'].copy()  # Regression target: predict vote_average

# Check for non-numeric columns
object_cols = X.select_dtypes(include=['object']).columns.tolist()
if object_cols:
    print(f"Warning: {len(object_cols)} object columns found, converting to numeric:")
    print(f"  {object_cols[:5]}")
    for col in object_cols:
        X[col] = pd.to_numeric(X[col], errors='coerce')

# Fill NaN and convert to float
X = X.fillna(0).astype(float)

# Reset index to avoid any index-related issues
X = X.reset_index(drop=True)
y = y.reset_index(drop=True)

# Random train-test split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Improved hyperparameter search

param_distributions = {
    'n_estimators': [400, 500, 600, 700, 800],
    'learning_rate': [0.005, 0.01, 0.02, 0.03, 0.05],
    'max_depth': [4, 5, 6, 7, 8, 9],
    'min_child_weight': [1, 2, 3, 5, 7],
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bylevel': [0.6, 0.7, 0.8, 0.9],
    'gamma': [0, 0.01, 0.05, 0.1, 0.2],
    'reg_alpha': [0, 0.01, 0.1, 0.5, 1],
    'reg_lambda': [0.5, 1, 1.5, 2, 3]
}

base_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42
)

tscv = TimeSeriesSplit(n_splits=3)

random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=50,
    scoring='neg_mean_squared_error',
    cv=tscv,
    verbose=1,
    random_state=42,
    n_jobs=-1
)

random_search.fit(X_train, y_train)

# Use best model
model = random_search.best_estimator_
model.fit(X_train, y_train)

# Evaluation
y_pred_test = model.predict(X_test)

# Calculate regression metrics
test_mse = mean_squared_error(y_test, y_pred_test)
test_rmse = np.sqrt(test_mse)
test_mae = mean_absolute_error(y_test, y_pred_test)
test_r2 = r2_score(y_test, y_pred_test)

print("\n" + "="*60)
print("MODEL EVALUATION METRICS")
print("="*60)
print(f"  RMSE (Root Mean Squared Error): {test_rmse:.4f}")
print(f"  MAE (Mean Absolute Error):      {test_mae:.4f}")
print(f"  R² (R-squared):                 {test_r2:.4f}")
print(f"  MSE (Mean Squared Error):       {test_mse:.4f}")
print("="*60)

# Feature importance
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Analyze time decay feature importance
time_decay_features = ['views_exp_decay_30', 'likes_exp_decay_30',
                       'views_recency_weighted', 'likes_recency_weighted',
                       'views_per_day', 'likes_per_day']
decay_importance = feature_importance[feature_importance['feature'].isin(time_decay_features)]
total_decay_importance = decay_importance['importance'].sum()

original_engagement = ['view_count', 'like_count', 'comment_count']
original_importance = feature_importance[feature_importance['feature'].isin(original_engagement)]
total_original_importance = original_importance['importance'].sum()

# Save results
feature_importance.to_csv('feature_importance.csv', index=False)

# Visualizations

# Plot 1: Top 20 Features
plt.figure(figsize=(10, 8))
top_features = feature_importance.head(20)
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(top_features)))
bars = plt.barh(range(len(top_features)), top_features['importance'], color=colors)
plt.yticks(range(len(top_features)), top_features['feature'], fontsize=10)
plt.xlabel('Importance', fontsize=12)
plt.title('Top 20 Most Important Features', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'top_features.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 2: Predicted vs Actual
plt.figure(figsize=(10, 8))
plt.scatter(y_test, y_pred_test, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Vote Average', fontsize=12)
plt.ylabel('Predicted Vote Average', fontsize=12)
plt.title('Predicted vs Actual Vote Average', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'predicted_vs_actual.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 3: Feature Categories
feature_categories = {
    'Trailer': ['has_trailer_data', 'view_count', 'like_count', 'comment_count', 'like_rate', 'comment_rate'],
    'Time Decay': time_decay_features,
    'Production': ['budget', 'budget_log', 'has_budget', 'runtime', 'production_company_count', 'production_country_count'],
    'Budget Bins': [f'budget_{tier}' for tier in ['micro', 'low', 'medium', 'high', 'blockbuster']],
    'Runtime Bins': [f'runtime_{cat}' for cat in ['short', 'standard', 'long', 'epic', 'marathon']],
    'Engagement Bins': [f'engagement_{tier}' for tier in ['none', 'very_low', 'low', 'medium', 'high', 'viral']],
    'Temporal': ['release_year', 'release_month', 'release_quarter', 'is_summer_release', 'is_holiday_release'] + decade_cols,
    'People - Basic': ['cast_count', 'director_count', 'has_cast', 'has_director'],
    'Cast - Quality': ['cast_avg_rating', 'cast_max_rating', 'has_star_actor', 'star_actor_count'],
    'Cast - Individual': actor_cols,
    'Director - Quality': ['director_avg_rating', 'director_max_rating', 'has_top_director', 'is_solo_director', 'is_multi_director'],
    'Director - Individual': director_cols,
    'Genre': ['genre_count', 'has_genre'] + [f'is_{g.lower().replace(" ", "_")}' for g in top_genres],
    'Sentiment': ['has_sentiment', 'sentiment_positive', 'sentiment_negative', 'sentiment_magnitude',
                  'sentiment_strength', 'very_positive', 'very_negative', 'neutral_sentiment',
                  'moderate_positive', 'moderate_negative'],
    'Production Companies': company_cols,
    'Production Countries': country_cols + ['is_us_production', 'is_international_coproduction', 'is_single_country'],
    'Diversity': ['has_diversity_data', 'female_cast_count', 'male_cast_count', 'female_cast_percentage',
                  'gender_balance_score', 'female_director', 'male_director', 'has_known_director_gender',
                  'male_dominated_cast', 'female_dominated_cast', 'balanced_cast',
                  'total_cast_from_diversity', 'male_to_female_ratio', 'high_gender_balance', 'low_gender_balance'],
    'External': ['has_rt_url', 'has_critic_score', 'has_audience_score']
}

category_importance = {}
for category, features in feature_categories.items():
    cat_features = [f for f in features if f in feature_importance['feature'].values]
    importance_sum = feature_importance[feature_importance['feature'].isin(cat_features)]['importance'].sum()
    category_importance[category] = importance_sum

plt.figure(figsize=(10, 6))
categories = list(category_importance.keys())
importances = list(category_importance.values())
colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))
bars = plt.bar(categories, importances, color=colors, edgecolor='black', linewidth=1.5)
plt.ylabel('Total Importance', fontsize=12)
plt.title('Feature Importance by Category', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars, importances):
    plt.text(bar.get_x() + bar.get_width()/2, val, f'{val:.3f}',
             ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'category_importance.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 4: Residuals Plot
residuals = y_test - y_pred_test
plt.figure(figsize=(10, 6))
plt.scatter(y_pred_test, residuals, alpha=0.5, s=30, edgecolors='k', linewidth=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('Predicted Vote Average', fontsize=12)
plt.ylabel('Residuals (Actual - Predicted)', fontsize=12)
plt.title('Residuals Plot', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'residuals_plot.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 5: Residuals Distribution
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
plt.axvline(x=0, color='r', linestyle='--', lw=2, label='Zero Error')
plt.xlabel('Residuals (Actual - Predicted)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Residuals', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'residuals_distribution.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 6: Performance by Release Year
test_df = df_model.loc[X_test.index].copy()
test_df['prediction'] = y_pred_test
test_df['actual'] = y_test.values

yearly_performance = test_df.groupby('release_year').agg({
    'actual': 'mean',
    'prediction': 'mean'
}).reset_index()

plt.figure(figsize=(12, 6))
plt.plot(yearly_performance['release_year'], yearly_performance['actual'],
         marker='o', linewidth=2, markersize=8, label='Actual Avg Vote', color='blue')
plt.plot(yearly_performance['release_year'], yearly_performance['prediction'],
         marker='s', linewidth=2, markersize=8, label='Predicted Avg Vote', color='orange')
plt.xlabel('Release Year', fontsize=12)
plt.ylabel('Average Vote', fontsize=12)
plt.title('Model Performance by Release Year (Test Set)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'performance_by_year.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 7: Feature Distribution (Top Features)
top_5_features = feature_importance.head(5)['feature'].tolist()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, feature in enumerate(top_5_features):
    if feature in df_model.columns:
        ax = axes[idx]
        feature_data = df_model[feature]

        # Convert boolean columns to numeric for histogram plotting
        if feature_data.dtype == bool:
            feature_data = feature_data.astype(int)

        ax.hist(feature_data, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
        ax.set_xlabel(feature, fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

# Remove extra subplot
axes[-1].axis('off')

plt.suptitle('Distribution of Top 5 Features', fontsize=14, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig(os.path.join(output_folder, 'feature_distributions.png'), dpi=300, bbox_inches='tight')
plt.close()

# Plot 8: Budget vs Average Vote
if 'budget' in df_model.columns:
    budget_bins = pd.qcut(df_model[df_model['budget'] > 0]['budget'], q=10, duplicates='drop')
    budget_vote_avg = df_model[df_model['budget'] > 0].groupby(budget_bins)['vote_average'].mean()

    plt.figure(figsize=(12, 6))
    x_pos = range(len(budget_vote_avg))
    plt.bar(x_pos, budget_vote_avg.values, color='steelblue', edgecolor='black', linewidth=1.5)
    plt.xlabel('Budget Range', fontsize=12)
    plt.ylabel('Average Vote', fontsize=12)
    plt.title('Average Vote by Budget Range', fontsize=14, fontweight='bold')
    plt.xticks(x_pos, [f'${int(interval.left/1e6)}-{int(interval.right/1e6)}M' for interval in budget_vote_avg.index],
               rotation=45, ha='right', fontsize=9)
    plt.grid(True, alpha=0.3, axis='y')
    plt.axhline(y=df_model['vote_average'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Overall avg: {df_model["vote_average"].mean():.2f}')
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'budget_vs_vote.png'), dpi=300, bbox_inches='tight')
    plt.close()

# ============================================================================
# DIVERSITY METRICS EFFECTIVENESS VISUALIZATIONS
# ============================================================================

# Plot 9: Gender Balance Score vs Vote Average
if 'gender_balance_score' in df_model.columns and df_model['has_diversity_data'].sum() > 0:
    diversity_data = df_model[df_model['has_diversity_data'] == 1].copy()

    # Create bins for gender balance score
    diversity_data['balance_bin'] = pd.cut(
        diversity_data['gender_balance_score'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['Very Low\n(0-20)', 'Low\n(20-40)', 'Medium\n(40-60)', 'High\n(60-80)', 'Very High\n(80-100)']
    )

    balance_stats = diversity_data.groupby('balance_bin', observed=True).agg({
        'vote_average': ['mean', 'count', 'std']
    }).reset_index()
    balance_stats.columns = ['balance_bin', 'avg_vote', 'count', 'std']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Left plot: Average vote by balance score
    colors = ['#d62728', '#ff7f0e', '#ffdd57', '#2ca02c', '#1f77b4']
    bars = ax1.bar(range(len(balance_stats)), balance_stats['avg_vote'],
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax1.axhline(y=df_model['vote_average'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Overall avg: {df_model["vote_average"].mean():.2f}')
    ax1.set_xlabel('Gender Balance Score', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Average Vote', fontsize=12, fontweight='bold')
    ax1.set_title('Vote Average by Gender Balance Score', fontsize=14, fontweight='bold')
    ax1.set_xticks(range(len(balance_stats)))
    ax1.set_xticklabels(balance_stats['balance_bin'], fontsize=10)
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, val, cnt) in enumerate(zip(bars, balance_stats['avg_vote'], balance_stats['count'])):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}\n(n={int(cnt)})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Right plot: Sample size distribution
    ax2.bar(range(len(balance_stats)), balance_stats['count'],
            color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax2.set_xlabel('Gender Balance Score', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Number of Movies', fontsize=12, fontweight='bold')
    ax2.set_title('Sample Size by Gender Balance Category', fontsize=14, fontweight='bold')
    ax2.set_xticks(range(len(balance_stats)))
    ax2.set_xticklabels(balance_stats['balance_bin'], fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'diversity_gender_balance.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Plot 10: Female Cast Percentage vs Vote Average
if 'female_cast_percentage' in df_model.columns and df_model['has_diversity_data'].sum() > 0:
    diversity_data = df_model[df_model['has_diversity_data'] == 1].copy()

    # Create bins for female cast percentage
    diversity_data['female_pct_bin'] = pd.cut(
        diversity_data['female_cast_percentage'],
        bins=[0, 20, 40, 60, 80, 100],
        labels=['0-20%', '20-40%', '40-60%', '60-80%', '80-100%']
    )

    female_pct_stats = diversity_data.groupby('female_pct_bin', observed=True).agg({
        'vote_average': ['mean', 'count', 'std']
    }).reset_index()
    female_pct_stats.columns = ['female_pct_bin', 'avg_vote', 'count', 'std']

    plt.figure(figsize=(12, 6))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.8, len(female_pct_stats)))
    bars = plt.bar(range(len(female_pct_stats)), female_pct_stats['avg_vote'],
                   color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    plt.axhline(y=df_model['vote_average'].mean(), color='red', linestyle='--',
                linewidth=2, label=f'Overall avg: {df_model["vote_average"].mean():.2f}')
    plt.xlabel('Female Cast Percentage', fontsize=12, fontweight='bold')
    plt.ylabel('Average Vote', fontsize=12, fontweight='bold')
    plt.title('Vote Average by Female Cast Percentage', fontsize=14, fontweight='bold')
    plt.xticks(range(len(female_pct_stats)), female_pct_stats['female_pct_bin'], fontsize=10)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, val, cnt) in enumerate(zip(bars, female_pct_stats['avg_vote'], female_pct_stats['count'])):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}\n(n={int(cnt)})', ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'diversity_female_cast_pct.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Plot 11: Director Gender Comparison
if 'female_director' in df_model.columns and df_model['has_diversity_data'].sum() > 0:
    diversity_data = df_model[df_model['has_known_director_gender'] == 1].copy()

    if len(diversity_data) > 0:
        director_stats = diversity_data.groupby('female_director').agg({
            'vote_average': ['mean', 'count', 'std']
        }).reset_index()
        director_stats.columns = ['female_director', 'avg_vote', 'count', 'std']
        director_stats['director_type'] = director_stats['female_director'].map({0: 'Male Director', 1: 'Female Director'})

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Left plot: Average vote comparison
        colors = ['#1f77b4', '#e377c2']
        bars = ax1.bar(range(len(director_stats)), director_stats['avg_vote'],
                      color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
        ax1.axhline(y=df_model['vote_average'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f'Overall avg: {df_model["vote_average"].mean():.2f}')
        ax1.set_xlabel('Director Gender', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Average Vote', fontsize=12, fontweight='bold')
        ax1.set_title('Vote Average by Director Gender', fontsize=14, fontweight='bold')
        ax1.set_xticks(range(len(director_stats)))
        ax1.set_xticklabels(director_stats['director_type'], fontsize=11)
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, axis='y')

        # Add value labels
        for bar, val, cnt in zip(bars, director_stats['avg_vote'], director_stats['count']):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f'{val:.2f}\n(n={int(cnt)})', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # Right plot: Distribution over time
        if 'release_year' in diversity_data.columns:
            yearly_gender = diversity_data.groupby(['release_year', 'female_director']).size().unstack(fill_value=0)
            yearly_gender_pct = yearly_gender.div(yearly_gender.sum(axis=1), axis=0) * 100

            if 1 in yearly_gender_pct.columns:
                ax2.plot(yearly_gender_pct.index, yearly_gender_pct[1],
                        marker='o', linewidth=2, markersize=6, color='#e377c2', label='Female Directors')
            if 0 in yearly_gender_pct.columns:
                ax2.plot(yearly_gender_pct.index, yearly_gender_pct[0],
                        marker='s', linewidth=2, markersize=6, color='#1f77b4', label='Male Directors')

            ax2.set_xlabel('Release Year', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Percentage (%)', fontsize=12, fontweight='bold')
            ax2.set_title('Director Gender Distribution Over Time', fontsize=14, fontweight='bold')
            ax2.legend(fontsize=11)
            ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'diversity_director_gender.png'), dpi=300, bbox_inches='tight')
        plt.close()

# ============================================================================
# TIME DECAY FEATURES EFFECTIVENESS VISUALIZATIONS
# ============================================================================

# Plot 12: Time Decay Impact - Views
if 'has_trailer_data' in df_model.columns and df_model['has_trailer_data'].sum() > 0:
    trailer_data = df_model[df_model['has_trailer_data'] == 1].copy()

    # Create bins for days until release
    trailer_data['days_bin'] = pd.cut(
        trailer_data['days_until_release'],
        bins=[-np.inf, 0, 7, 14, 30, 60, 90, np.inf],
        labels=['After Release', '0-7 days', '8-14 days', '15-30 days', '31-60 days', '61-90 days', '90+ days']
    )

    days_stats = trailer_data.groupby('days_bin', observed=True).agg({
        'vote_average': ['mean', 'count'],
        'view_count': 'mean',
        'views_exp_decay_30': 'mean',
        'views_recency_weighted': 'mean'
    }).reset_index()
    days_stats.columns = ['days_bin', 'avg_vote', 'count', 'avg_views', 'avg_decay_views', 'avg_recency_views']

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Vote Average by Trailer Release Timing
    ax = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(days_stats)))
    bars = ax.bar(range(len(days_stats)), days_stats['avg_vote'],
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.axhline(y=df_model['vote_average'].mean(), color='red', linestyle='--',
               linewidth=2, label=f'Overall avg: {df_model["vote_average"].mean():.2f}')
    ax.set_xlabel('Days Before Release', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Vote', fontsize=11, fontweight='bold')
    ax.set_title('Vote Average by Trailer Release Timing', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(days_stats)))
    ax.set_xticklabels(days_stats['days_bin'], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar, val, cnt in zip(bars, days_stats['avg_vote'], days_stats['count']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.2f}\n(n={int(cnt)})', ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Plot 2: Raw Views vs Time Decay Views
    ax = axes[0, 1]
    x = np.arange(len(days_stats))
    width = 0.35
    bars1 = ax.bar(x - width/2, days_stats['avg_views']/1000, width,
                   label='Raw Views', color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)
    bars2 = ax.bar(x + width/2, days_stats['avg_decay_views']/1000, width,
                   label='Decay-Weighted Views', color='orange', edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Days Before Release', fontsize=11, fontweight='bold')
    ax.set_ylabel('Average Views (thousands)', fontsize=11, fontweight='bold')
    ax.set_title('Raw Views vs Exponential Decay Views', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(days_stats['days_bin'], rotation=45, ha='right', fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Recency-Weighted Views
    ax = axes[1, 0]
    bars = ax.bar(range(len(days_stats)), days_stats['avg_recency_views']/1000,
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Days Before Release', fontsize=11, fontweight='bold')
    ax.set_ylabel('Recency-Weighted Views (thousands)', fontsize=11, fontweight='bold')
    ax.set_title('Recency-Weighted Views (Peak at 14 days)', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(days_stats)))
    ax.set_xticklabels(days_stats['days_bin'], rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Sample Size Distribution
    ax = axes[1, 1]
    bars = ax.bar(range(len(days_stats)), days_stats['count'],
                  color=colors, edgecolor='black', linewidth=1.5, alpha=0.8)
    ax.set_xlabel('Days Before Release', fontsize=11, fontweight='bold')
    ax.set_ylabel('Number of Movies', fontsize=11, fontweight='bold')
    ax.set_title('Sample Size by Trailer Release Timing', fontsize=13, fontweight='bold')
    ax.set_xticks(range(len(days_stats)))
    ax.set_xticklabels(days_stats['days_bin'], rotation=45, ha='right', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'time_decay_effectiveness.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Plot 13: Correlation Heatmap - Time Decay Features
if 'has_trailer_data' in df_model.columns and df_model['has_trailer_data'].sum() > 0:
    trailer_data = df_model[df_model['has_trailer_data'] == 1].copy()

    time_decay_cols = [
        'view_count', 'like_count',
        'views_exp_decay_30', 'likes_exp_decay_30',
        'views_recency_weighted', 'likes_recency_weighted',
        'views_per_day', 'likes_per_day',
        'vote_average'
    ]

    # Filter to only existing columns
    time_decay_cols = [col for col in time_decay_cols if col in trailer_data.columns]

    if len(time_decay_cols) > 2:
        corr_matrix = trailer_data[time_decay_cols].corr()

        plt.figure(figsize=(12, 10))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f', cmap='coolwarm',
                   center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                   vmin=-1, vmax=1)
        plt.title('Correlation Matrix: Time Decay Features vs Vote Average',
                 fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'time_decay_correlation.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Plot 14: Feature Importance - Diversity vs Time Decay
diversity_features = [f for f in feature_importance['feature'].values
                     if any(keyword in f for keyword in ['female', 'male', 'gender', 'balance', 'diversity'])]
time_decay_features_list = [f for f in feature_importance['feature'].values
                           if any(keyword in f for keyword in ['decay', 'recency', 'per_day'])]

if len(diversity_features) > 0 or len(time_decay_features_list) > 0:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Diversity features importance
    if len(diversity_features) > 0:
        diversity_imp = feature_importance[feature_importance['feature'].isin(diversity_features)].head(10)
        colors_div = plt.cm.RdPu(np.linspace(0.4, 0.9, len(diversity_imp)))
        bars = ax1.barh(range(len(diversity_imp)), diversity_imp['importance'],
                       color=colors_div, edgecolor='black', linewidth=1.5)
        ax1.set_yticks(range(len(diversity_imp)))
        ax1.set_yticklabels(diversity_imp['feature'], fontsize=10)
        ax1.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax1.set_title('Top Diversity Features Importance', fontsize=13, fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        ax1.invert_yaxis()

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, diversity_imp['importance'])):
            ax1.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    else:
        ax1.text(0.5, 0.5, 'No diversity features\nin top features',
                ha='center', va='center', fontsize=12, transform=ax1.transAxes)
        ax1.set_title('Top Diversity Features Importance', fontsize=13, fontweight='bold')

    # Time decay features importance
    if len(time_decay_features_list) > 0:
        time_decay_imp = feature_importance[feature_importance['feature'].isin(time_decay_features_list)].head(10)
        colors_time = plt.cm.YlGnBu(np.linspace(0.4, 0.9, len(time_decay_imp)))
        bars = ax2.barh(range(len(time_decay_imp)), time_decay_imp['importance'],
                       color=colors_time, edgecolor='black', linewidth=1.5)
        ax2.set_yticks(range(len(time_decay_imp)))
        ax2.set_yticklabels(time_decay_imp['feature'], fontsize=10)
        ax2.set_xlabel('Importance', fontsize=12, fontweight='bold')
        ax2.set_title('Top Time Decay Features Importance', fontsize=13, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='x')
        ax2.invert_yaxis()

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, time_decay_imp['importance'])):
            ax2.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9, fontweight='bold')
    else:
        ax2.text(0.5, 0.5, 'No time decay features\nin top features',
                ha='center', va='center', fontsize=12, transform=ax2.transAxes)
        ax2.set_title('Top Time Decay Features Importance', fontsize=13, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'diversity_vs_time_decay_importance.png'), dpi=300, bbox_inches='tight')
    plt.close()

# Plot 15: Diversity Trends Over Time
if 'has_diversity_data' in df_model.columns and df_model['has_diversity_data'].sum() > 0:
    diversity_data = df_model[df_model['has_diversity_data'] == 1].copy()

    if 'release_year' in diversity_data.columns and len(diversity_data) > 0:
        yearly_diversity = diversity_data.groupby('release_year').agg({
            'female_cast_percentage': 'mean',
            'gender_balance_score': 'mean',
            'female_director': 'mean',
            'vote_average': 'mean'
        }).reset_index()

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Plot 1: Female Cast Percentage Over Time
        ax = axes[0, 0]
        ax.plot(yearly_diversity['release_year'], yearly_diversity['female_cast_percentage'],
               marker='o', linewidth=2.5, markersize=6, color='#e377c2', label='Female Cast %')
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='50% (Parity)')
        ax.set_xlabel('Release Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Female Cast Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Female Cast Representation Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        # Plot 2: Gender Balance Score Over Time
        ax = axes[0, 1]
        ax.plot(yearly_diversity['release_year'], yearly_diversity['gender_balance_score'],
               marker='s', linewidth=2.5, markersize=6, color='#2ca02c', label='Balance Score')
        ax.set_xlabel('Release Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Gender Balance Score', fontsize=11, fontweight='bold')
        ax.set_title('Gender Balance Score Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        # Plot 3: Female Director Percentage Over Time
        ax = axes[1, 0]
        ax.plot(yearly_diversity['release_year'], yearly_diversity['female_director'] * 100,
               marker='^', linewidth=2.5, markersize=6, color='#9467bd', label='Female Directors %')
        ax.axhline(y=50, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='50% (Parity)')
        ax.set_xlabel('Release Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Female Director Percentage (%)', fontsize=11, fontweight='bold')
        ax.set_title('Female Director Representation Over Time', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        # Plot 4: Vote Average Over Time (for context)
        ax = axes[1, 1]
        ax.plot(yearly_diversity['release_year'], yearly_diversity['vote_average'],
               marker='D', linewidth=2.5, markersize=6, color='#ff7f0e', label='Avg Vote')
        ax.set_xlabel('Release Year', fontsize=11, fontweight='bold')
        ax.set_ylabel('Average Vote', fontsize=11, fontweight='bold')
        ax.set_title('Vote Average Over Time (Movies with Diversity Data)', fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_folder, 'diversity_trends_over_time.png'), dpi=300, bbox_inches='tight')
        plt.close()

# Print final results
print("\n" + "="*80)
print("XGBOOST MODEL RESULTS")
print("="*80)
print(f"\nDataset: {len(df_model):,} movies")
print(f"Features: {len(feature_columns)}")
print(f"Train/Test Split: {len(X_train):,} / {len(X_test):,}")

print(f"\nTarget Distribution (vote_average):")
print(f"  Mean: {y_train.mean():.2f} ± {y_train.std():.2f}")
print(f"  Range: [{y_train.min():.2f}, {y_train.max():.2f}]")

print(f"\nBest Hyperparameters:")
for param, value in random_search.best_params_.items():
    print(f"  {param:20s}: {value}")

print(f"\nCross-Validation Performance:")
print(f"  CV RMSE: {np.sqrt(-random_search.best_score_):.4f}")

print(f"\nTest Set Performance:")
print(f"  RMSE: {test_rmse:.4f}")
print(f"  MAE:  {test_mae:.4f}")
print(f"  R²:   {test_r2:.4f}")

print(f"\nTop 15 Most Important Features:")
print(feature_importance.head(15).to_string(index=False))

print(f"\nFeature Category Importance:")
print(f"  Time Decay Features:     {total_decay_importance*100:.2f}%")
print(f"  Raw Engagement Features: {total_original_importance*100:.2f}%")

print(f"\nDiversity Metrics Coverage:")
print(f"  Movies with diversity data: {df['has_diversity_data'].sum():,} ({df['has_diversity_data'].mean()*100:.1f}%)")
print(f"  Movies with female directors: {df['female_director'].sum():,} ({df['female_director'].mean()*100:.1f}%)")
print(f"  Average female cast percentage: {df['female_cast_percentage'].mean():.1f}%")

print(f"\nTrailer Data Coverage:")
print(f"  Movies with trailer data: {df['has_trailer_data'].sum():,} ({df['has_trailer_data'].mean()*100:.1f}%)")
print(f"  Movies with RT scores: {df['has_rt_url'].sum():,} ({df['has_rt_url'].mean()*100:.1f}%)")

print(f"\nVisualizations saved to: {output_folder}/")
print("  • feature_importance.png")
print("  • predicted_vs_actual.png")
print("  • feature_importance_by_category.png")
print("  • residuals.png")
print("  • residuals_distribution.png")
print("  • performance_by_year.png")
print("  • feature_distributions.png")
print("  • budget_vs_vote.png")
print("  • diversity_gender_balance.png")
print("  • diversity_female_cast_pct.png")
print("  • diversity_director_gender.png")
print("  • diversity_trends_over_time.png")
print("  • time_decay_effectiveness.png")
print("  • time_decay_correlation.png")
print("  • diversity_vs_time_decay_importance.png")

print("\nResults saved to: feature_importance.csv")
print("="*80)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
import os
import joblib  # For loading XGBoost model and scaler
import json    # For loading feature list / metadata
import pickle  # For loading stacking model
import networkx as nx  # For graph visualization


# =============================================================================
# CONFIGURATION
# =============================================================================

# MongoDB Connection
MONGODB_URI = st.secrets.get(
    "MONGODB_URI",
    "mongodb+srv://cinemaniacs:filmlytics@filmlytics.1emhcue.mongodb.net/?appName=filmlytics"
)

# Page Configuration
st.set_page_config(
    page_title="Filmlytics - Movie Audience Score Prediction",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

@st.cache_resource
def get_database_connection():
    """Connect to MongoDB and cache the connection."""
    try:
        client = MongoClient(
            MONGODB_URI,
            server_api=ServerApi('1'),
            tlsCAFile=certifi.where()
        )
        db = client['cinemaniacs']
        # Test connection
        db.movies.count_documents({})
        return db
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


# =============================================================================
# ARTIFACT LOADING AND ENSEMBLE PREDICTION SETUP
# =============================================================================

@st.cache_resource
def load_ensemble_artifacts():
    """
    Load all models, scalers, prediction dataframes, and stacking model/metadata.
    This mirrors your teammate's working ensemble setup.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    artifact_dir = os.path.join(script_dir, 'model_artifacts')

    artifacts = {}

    required_files = {
        # XGBoost
        'xg_model': 'xgboost_base_model.pkl',
        'xg_features': 'xg_feature_columns.json',
        # GNN/KGCN shared
        'scaler': 'movie_feature_scaler_diversity.pkl',
        # GNN Predictions (Lookup)
        'gnn_preds': 'gnn_preds_all_movies.csv',
        # KGCN Predictions (Lookup)
        'kgcn_preds': 'kgcn_preds_all_movies.csv',
        # XGBoost Predictions (Lookup)
        'xgb_preds': 'xgb_preds_all_movies.csv',
    }

    for key, filename in required_files.items():
        path = os.path.join(artifact_dir, filename)
        if not os.path.exists(path):
            st.warning(f"Missing required file: {filename}. Prediction will be incomplete.")
            artifacts[key] = None
            continue

        try:
            if filename.endswith('.pkl'):
                artifacts[key] = joblib.load(path)
            elif filename.endswith('.json'):
                with open(path, 'r') as f:
                    artifacts[key] = json.load(f)
            elif filename.endswith('.csv'):
                df = pd.read_csv(path)
                df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce').fillna(0).astype(int)

                if 'pred_audience_score' in df.columns:
                    artifacts[key] = df.set_index('tmdb_id')['pred_audience_score']
                elif 'predicted_audience_score' in df.columns:
                    artifacts[key] = df.set_index('tmdb_id')['predicted_audience_score']
                else:
                    artifacts[key] = df.set_index('tmdb_id').iloc[:, 0]
        except Exception as e:
            return None, f"Error loading {filename}: {e}"

    # Stacking meta-model
    stacking_path = os.path.join(artifact_dir, 'stacking_meta_model.pkl')
    if os.path.exists(stacking_path):
        try:
            with open(stacking_path, 'rb') as f:
                artifacts['stacking_model'] = pickle.load(f)
        except Exception as e:
            st.warning(f"Could not load stacking model: {e}")
            artifacts['stacking_model'] = None
    else:
        artifacts['stacking_model'] = None

    # Ensemble metadata
    meta_path = os.path.join(artifact_dir, 'ensemble_weights.json')
    if os.path.exists(meta_path):
        try:
            with open(meta_path, 'r') as f:
                artifacts['ensemble_meta'] = json.load(f)
        except Exception:
            artifacts['ensemble_meta'] = {}
    else:
        artifacts['ensemble_meta'] = {}

    # Fallback ensemble weights
    artifacts['ensemble_weights'] = {'gnn': 0.33, 'kgcn': 0.34, 'xg': 0.33}

    return artifacts, None


def parse_pct_string(s):
    if isinstance(s, str) and s.endswith("%"):
        try:
            return float(s[:-1].strip()) / 100.0
        except Exception:
            return np.nan
    return np.nan


def generate_xgboost_features(movie_data, artifacts):
    """
    Placeholder for a feature generator matching the XGBoost training pipeline.
    Currently unused; predictions come from precomputed lookup tables.
    """
    return None


def safe_get_prediction(preds, tmdb_id):
    """Safely get a prediction value from a pandas Series."""
    if preds is None:
        return np.nan
    val = preds.get(tmdb_id, np.nan)
    if isinstance(val, pd.Series):
        val = val.iloc[0] if len(val) > 0 else np.nan
    if pd.isna(val):
        return np.nan
    return float(val)


def predict_ensemble(movie_data, artifacts):
    """
    Uses stacking meta-learner to combine GNN, KGCN, and XGBoost predictions.
    Falls back to weighted average if stacking model is not available.

    Returns: ensemble_prediction (0-1), prediction_breakdown (dict)
    """
    tmdb_id = movie_data.get('tmdb_id')
    if tmdb_id is None:
        return np.nan, {}

    gnn_pred = safe_get_prediction(artifacts.get('gnn_preds'), tmdb_id)
    kgcn_pred = safe_get_prediction(artifacts.get('kgcn_preds'), tmdb_id)
    xgb_pred = safe_get_prediction(artifacts.get('xgb_preds'), tmdb_id)

    predictions = {
        'gnn': gnn_pred,
        'kgcn': kgcn_pred,
        'xg': xgb_pred
    }

    stacking_model = artifacts.get('stacking_model')
    has_all_preds = not (np.isnan(gnn_pred) or np.isnan(kgcn_pred) or np.isnan(xgb_pred))

    if stacking_model is not None and has_all_preds:
        X = np.array([[gnn_pred, kgcn_pred, xgb_pred]])
        ensemble_pred = float(np.clip(stacking_model.predict(X)[0], 0, 1))
        return ensemble_pred, predictions

    # Fallback: weighted average
    weights = artifacts.get('ensemble_weights', {'gnn': 0.33, 'kgcn': 0.34, 'xg': 0.33})
    valid_preds = {k: v for k, v in predictions.items() if not np.isnan(v)}

    if not valid_preds:
        tmdb_avg = movie_data.get('tmdb_metrics', {}).get('vote_average')
        if tmdb_avg is not None:
            return float(tmdb_avg) / 10.0, predictions
        return np.nan, predictions

    valid_keys = valid_preds.keys()
    total_valid_weight = sum(weights.get(k, 0) for k in valid_keys)

    if total_valid_weight == 0:
        ensemble_pred = np.mean(list(valid_preds.values()))
    else:
        ensemble_pred = sum(
            valid_preds[k] * (weights[k] / total_valid_weight)
            for k in valid_keys
        )

    ensemble_pred = np.clip(ensemble_pred, 0.0, 1.0)
    return ensemble_pred, predictions


# =============================================================================
# DATA QUERY FUNCTIONS
# =============================================================================

@st.cache_data
def get_all_movie_titles(_db):
    """Get a list of UNIQUE movie titles, sorted by vote count (popularity)."""
    try:
        pipeline = [
            {"$match": {
                "title": {"$ne": None},
                "tmdb_id": {"$ne": None},
                "tmdb_metrics.vote_count": {"$gte": 1}
            }},
            {"$group": {
                "_id": "$tmdb_id",
                "title": {"$first": "$title"},
                "vote_count": {"$max": "$tmdb_metrics.vote_count"}
            }},
            {"$sort": {"vote_count": -1}},
            {"$project": {"_id": 0, "title": 1}}
        ]
        unique_titles = list(_db.movies.aggregate(pipeline))
        return [doc['title'] for doc in unique_titles]
    except Exception as e:
        st.error(f"Error fetching unique movie titles: {e}")
        return []


def search_movie(db, title):
    movie = db.movies.find_one({"title": {"$regex": f"^{title}$", "$options": "i"}})
    if not movie:
        movie = db.movies.find_one({"title": {"$regex": title, "$options": "i"}})
    return movie


def get_top_movies(db, limit=50, min_votes=1000):
    query = {
        "tmdb_metrics.vote_count": {"$gte": min_votes},
        "tmdb_metrics.vote_average": {"$ne": None}
    }
    return list(db.movies.find(query).sort("tmdb_metrics.vote_average", -1).limit(limit))


def get_similar_movies(db, tmdb_id, limit=10, min_votes=1000):
    movie = db.movies.find_one({"tmdb_id": tmdb_id})
    if not movie:
        return []

    genres = movie['production'].get('genres', [])
    if not genres:
        return []

    genre_count = len(genres)

    pipeline = [
        {"$match": {
            "production.genres": {
                "$all": genres,
                "$size": genre_count
            },
            "tmdb_id": {"$ne": tmdb_id},
            "tmdb_metrics.vote_count": {"$gte": min_votes},
            "tmdb_metrics.vote_average": {"$ne": None}
        }},
        {"$group": {
            "_id": "$tmdb_id",
            "unique_movie": {"$first": "$$ROOT"}
        }},
        {"$replaceRoot": {"newRoot": "$unique_movie"}},
        {"$sort": {"tmdb_metrics.vote_average": -1}},
        {"$limit": limit}
    ]
    return list(db.movies.aggregate(pipeline))


def get_database_stats(db):
    total = db.movies.count_documents({})
    successful = db.movies.count_documents({"tmdb_metrics.is_successful": True})
    with_rt = db.movies.count_documents({"rotten_tomatoes.has_rt_url": True})
    with_trailer = db.movies.count_documents({"trailer.trailer_url_youtube": {"$ne": None}})

    return {
        "total": total,
        "successful": successful,
        "with_rotten_tomatoes": with_rt,
        "with_trailers": with_trailer
    }


def get_all_genres(db):
    genres = db.movies.distinct("production.genres")
    return sorted([g for g in genres if g])


def get_movies_by_genre(db, genre, limit=20, min_votes=1000):
    pipeline = [
        {"$match": {
            "production.genres": genre,
            "tmdb_metrics.vote_count": {"$gte": min_votes},
            "tmdb_metrics.vote_average": {"$ne": None}
        }},
        {"$group": {
            "_id": "$tmdb_id",
            "unique_movie": {"$first": "$$ROOT"}
        }},
        {"$replaceRoot": {"newRoot": "$unique_movie"}},
        {"$sort": {"tmdb_metrics.vote_average": -1}},
        {"$limit": limit}
    ]
    return list(db.movies.aggregate(pipeline))


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_genre_distribution_chart(db):
    pipeline = [
        {"$unwind": "$production.genres"},
        {"$group": {"_id": "$production.genres", "count": {"$sum": 1}}},
        {"$sort": {"count": -1}},
        {"$limit": 15}
    ]
    results = list(db.movies.aggregate(pipeline))
    df = pd.DataFrame(results)
    if df.empty:
        return None
    df.columns = ['Genre', 'Count']
    fig = px.bar(
        df,
        x='Genre',
        y='Count',
        title='Top 15 Movie Genres',
        color='Count',
        color_continuous_scale='viridis'
    )
    fig.update_layout(xaxis_tickangle=-45)
    return fig


def create_rating_distribution(db):
    movies = list(db.movies.find(
        {"tmdb_metrics.vote_average": {"$ne": None}},
        {"tmdb_metrics.vote_average": 1}
    ).limit(5000))

    ratings = [m['tmdb_metrics']['vote_average'] for m in movies]
    if not ratings:
        return None

    fig = px.histogram(
        ratings,
        nbins=50,
        title='Distribution of Movie Ratings',
        labels={'value': 'Rating', 'count': 'Number of Movies'}
    )
    fig.update_layout(showlegend=False)
    return fig


def create_success_over_time(db):
    pipeline = [
        {"$match": {"release_info.tmdb_release_date": {"$ne": None, "$regex": "^[0-9]{4}"}}},
        {"$project": {
            "year": {"$substr": ["$release_info.tmdb_release_date", 0, 4]},
            "is_successful": "$tmdb_metrics.is_successful"
        }},
        {"$group": {
            "_id": "$year",
            "total": {"$sum": 1},
            "successful": {"$sum": {"$cond": ["$is_successful", 1, 0]}}
        }},
        {"$project": {
            "year": "$_id",
            "success_rate": {"$multiply": [{"$divide": ["$successful", "$total"]}, 100]}
        }},
        {"$sort": {"year": 1}}
    ]

    results = list(db.movies.aggregate(pipeline))
    df = pd.DataFrame(results)

    if not df.empty and len(df) > 10:
        df = df[df['year'].astype(int) >= 2000]
        if df.empty:
            return None

        fig = px.line(
            df,
            x='year',
            y='success_rate',
            title='Movie Success Rate Over Time',
            labels={'year': 'Year', 'success_rate': 'Success Rate (%)'},
            markers=True
        )
        fig.update_layout(hovermode='x unified')
        return fig
    return None


# =============================================================================
# PAGE FUNCTIONS
# =============================================================================

def home_page(db):

    # ===============================
    # HERO HEADER (Large, Professional)
    # ===============================

    st.markdown(
        """
        <style>
        .hero-title {
            font-size: 3rem;
            font-weight: 800;
            text-align: center;
            margin-top: 0.2em;
            margin-bottom: 0.1em;
            color: #111827;
        }
        .hero-subtitle {
            font-size: 1.4rem;
            font-weight: 400;
            text-align: center;
            color: #4b5563;
            margin-bottom: 1.5em;
        }
        .section-header {
            font-size: 1.8rem;
            font-weight: 700;
            margin-top: 1.4em;
            color: #1f2937;
        }
        .content-text {
            font-size: 1.05rem;
            line-height: 1.55;
            color: #374151;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='hero-title'>Predicting Movie Audience Scores Using Graph-Based Modeling</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='hero-subtitle'>Filmytics — A Graph-Driven, Multi-Model Framework for Movie Audience Prediction</div>",
        unsafe_allow_html=True
    )

    st.markdown("---")

    # ===============================
    # INTRO SECTION
    # ===============================

    st.markdown("<div class='section-header'>What this project is and why it matters</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='content-text'>
        Filmytics predicts audience scores for movies <b>before release</b>.  
        Studios rely on early audience insights to guide marketing strategy, streaming placement,
        and financial forecasting. For fans and researchers, these predictions uncover what types
        of films resonate and how factors such as <b>representation</b> influence audience reception.
        <br><br>
        Our system combines large public datasets, builds rich feature representations,
        and uses advanced graph-based modeling to understand how films relate to each other.
        The result is a robust ensemble prediction framework for new and upcoming movies.
        </div>
        """,
        unsafe_allow_html=True
    )

    # ===============================
    # APPROACH SECTION
    # ===============================

    st.markdown("<div class='section-header'>Our Approach</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='content-text'>
        We collect data from three major public sources:
        <ul>
            <li><b>TMDB</b> — metadata such as cast, genres, runtime, budget, popularity</li>
            <li><b>Rotten Tomatoes</b> — critic & audience scores and review excerpts</li>
            <li><b>YouTube</b> — trailer views, likes, comments, and recency metrics</li>
        </ul>

        After cleaning and joining these sources, we build a unified dataset of
        <b>~66,000 films (2010–2025)</b> and engineer over <b>150 features</b>
        covering metadata, engagement metrics, sentiment, and representation indicators.
        <br><br>
        We then train three complementary models:
        <ul>
            <li><b>GNN</b> — captures similarity across films via shared attributes</li>
            <li><b>KGCN</b> — learns semantic relationships in the film knowledge graph</li>
            <li><b>XGBoost</b> — strong feature-based baseline using engineered predictors</li>
        </ul>

        Finally, a <b>stacking meta-learner</b> integrates these components into a single
        high-accuracy audience score predictor.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "Workflow: TMDB + RottenTomatoes + YouTube → Clean/Merge → Feature Engineering → {GNN, KGCN, XGBoost} → Stacking Meta-Learner → Audience Score Prediction"
    )

    # ===============================
    # RESULTS SECTION
    # ===============================

    st.markdown("<div class='section-header'>Main Findings and Results</div>", unsafe_allow_html=True)
    st.markdown(
        """
        <div class='content-text'>
        <ul>
            <li><b>XGBoost RMSE:</b> 0.110</li>
            <li><b>KGCN RMSE:</b> 0.171</li>
            <li><b>GNN RMSE:</b> 0.195</li>
            <li><b>Stacking Ensemble RMSE:</b> 0.1085 (best overall)</li>
            <li><b>80.2%</b> of predictions fall within ±10 percentage points</li>
        </ul>
        Combining graph-based and tabular models yields the strongest practical performance.
        </div>
        """,
        unsafe_allow_html=True
    )





def movie_search_page(db, artifacts):
    st.title("Movie Search and Prediction")
    st.markdown("Search for a film, view its metadata, and see the ensemble audience score prediction.")

    all_titles = get_all_movie_titles(db)

    selected_title = st.selectbox(
        "Select or type a movie title:",
        options=["-- Select a Movie --"] + all_titles,
        index=0
    )

    search_query = selected_title if selected_title != "-- Select a Movie --" else None

    if search_query:
        movie = search_movie(db, search_query)
        if movie:
            st.success(f"Found: {movie['title']}")

            ensemble_score, breakdown = predict_ensemble(movie, artifacts)

            st.markdown("---")
            st.subheader("Ensemble Audience Score Prediction")

            if not np.isnan(ensemble_score):
                st.success(f"Predicted Audience Score: {ensemble_score * 100:.1f}%")

                st.markdown("#### Model Breakdown")
                b1, b2, b3 = st.columns(3)
                with b1:
                    score = breakdown['gnn']
                    st.metric("GNN Prediction", f"{score * 100:.1f}%" if not np.isnan(score) else "N/A")
                with b2:
                    score = breakdown['kgcn']
                    st.metric("KGCN Prediction", f"{score * 100:.1f}%" if not np.isnan(score) else "N/A")
                with b3:
                    score = breakdown['xg']
                    st.metric("XGBoost Prediction", f"{score * 100:.1f}%" if not np.isnan(score) else "N/A")
            else:
                st.warning("Cannot generate ensemble prediction (missing GNN/KGCN/XGB data for this movie).")

            st.markdown("---")

            c1, c2 = st.columns([1, 2])
            with c1:
                poster = movie.get('content', {}).get('poster_url')
                if poster:
                    st.image(poster, width=250)
            with c2:
                st.subheader(movie['title'])
                st.write(f"**TMDB ID:** {movie['tmdb_id']}")
                st.write(f"**Genres:** {', '.join(movie['production'].get('genres', []))}")
                st.write(f"**Runtime:** {movie['production'].get('runtime', 'N/A')} minutes")
                st.write(f"**Budget:** ${movie['production'].get('budget', 0):,.0f}")
                st.write(f"**Release Date:** {movie['release_info'].get('tmdb_release_date', 'N/A')}")

                st.markdown("### Ratings")
                r1, r2, r3 = st.columns(3)
                with r1:
                    st.metric("TMDB Rating", f"{movie['tmdb_metrics'].get('vote_average', 'N/A')}/10")
                with r2:
                    rt_crit = movie.get('rotten_tomatoes', {}).get('critic_score')
                    st.metric("RT Critics", rt_crit if rt_crit else "N/A")
                with r3:
                    rt_aud = movie.get('rotten_tomatoes', {}).get('audience_score')
                    st.metric("RT Audience", rt_aud if rt_aud else "N/A")

                if movie['tmdb_metrics'].get('is_successful'):
                    st.success("Classified as: SUCCESSFUL")
                else:
                    st.error("Classified as: NOT SUCCESSFUL")

            st.markdown("### Similar Movies")
            similar = get_similar_movies(db, movie['tmdb_id'], limit=5)
            if similar:
                cols = st.columns(5)
                for idx, sim_movie in enumerate(similar):
                    with cols[idx]:
                        st.write(f"**{sim_movie['title']}**")
                        st.write(f"{sim_movie['tmdb_metrics']['vote_average']}/10")
            else:
                st.write("No similar movies found")
        else:
            st.error(f"No movie found matching '{search_query}'")

    st.markdown("---")
    st.subheader("Browse by Genre")
    genres = get_all_genres(db)
    if genres:
        selected_genre = st.selectbox("Select a genre:", genres)
        if selected_genre:
            genre_movies = get_movies_by_genre(db, selected_genre, limit=10)
            st.write(f"Showing top {len(genre_movies)} {selected_genre} movies:")
            for movie in genre_movies:
                st.write(f"**{movie['title']}** — {movie['tmdb_metrics']['vote_average']}/10")
    else:
        st.info("No genres available in the database.")


def compare_movies_page(db, artifacts):
    st.title("Compare Movies")
    st.markdown("Compare basic features and ensemble predictions for two films.")

    titles = get_all_movie_titles(db)
    if not titles:
        st.info("No titles available to compare.")
        return

    c1, c2 = st.columns(2)
    movie1_title = c1.selectbox("Select first movie:", titles, key="compare_movie_1")
    movie2_title = c2.selectbox("Select second movie:", titles, key="compare_movie_2")

    if movie1_title and movie2_title:
        m1 = search_movie(db, movie1_title)
        m2 = search_movie(db, movie2_title)

        if m1 is None or m2 is None:
            st.error("One or both selected movies could not be found in the database.")
            return

        st.subheader("Basic Feature Comparison")
        comparison = pd.DataFrame({
            "Feature": ["TMDB Score", "Vote Count", "Runtime (min)", "Budget ($)"],
            movie1_title: [
                m1["tmdb_metrics"]["vote_average"],
                m1["tmdb_metrics"]["vote_count"],
                m1["production"].get("runtime", None),
                m1["production"].get("budget", None),
            ],
            movie2_title: [
                m2["tmdb_metrics"]["vote_average"],
                m2["tmdb_metrics"]["vote_count"],
                m2["production"].get("runtime", None),
                m2["production"].get("budget", None),
            ]
        })
        st.table(comparison)

        st.subheader("Ensemble Prediction Comparison")
        pred1, _ = predict_ensemble(m1, artifacts)
        pred2, _ = predict_ensemble(m2, artifacts)

        c1, c2 = st.columns(2)
        c1.metric(movie1_title, f"{pred1 * 100:.1f}%" if not np.isnan(pred1) else "N/A")
        c2.metric(movie2_title, f"{pred2 * 100:.1f}%" if not np.isnan(pred2) else "N/A")


def analytics_page(db):
    st.title("Analytics Dashboard")
    st.markdown("Explore high-level patterns in the Filmlytics dataset.")

    tab1, tab2, tab3 = st.tabs(["Genre Analysis", "Rating Distribution", "Success Trends"])

    with tab1:
        st.subheader("Genre Distribution")
        fig = create_genre_distribution_chart(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for genre distribution.")

    with tab2:
        st.subheader("Rating Distribution (TMDB)")
        fig = create_rating_distribution(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for rating distribution.")

    with tab3:
        st.subheader("Success Rate Over Time")
        fig = create_success_over_time(db)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Insufficient data for time series analysis.")


def modeling_page(artifacts):
    st.title("Modeling and Ensemble Overview")

    stacking_model = artifacts.get('stacking_model') if artifacts else None
    ensemble_meta = artifacts.get('ensemble_meta', {}) if artifacts else {}

    tab1, tab2, tab3 = st.tabs(["Modeling Overview", "Ensemble Model", "Data Pipeline"])

    # TAB 1: Modeling Overview
    with tab1:
        st.subheader("Modeling Framework")

        st.markdown("#### Graph Neural Network (GNN)")
        st.write(
            """
            - Nodes: 66k+ films from 2010–2025  
            - Edges: Shared genres, directors, production countries, and diversity similarity  
            - Architecture: Two-layer GraphSAGE with batch normalization, dropout, and residual connections  
            - Objective: Predict audience score using neighborhood aggregation on the movie graph  
            """
        )

        st.markdown("#### Knowledge Graph Convolutional Network (KGCN)")
        st.write(
            """
            - Nodes: films, directors, actors, genres, production companies  
            - Edges: semantic relations (e.g., *directed_by*, *has_genre*, *produced_by*)  
            - Learns relation-specific embeddings that capture structured metadata interactions  
            """
        )

        st.markdown("#### XGBoost")
        st.write(
            """
            - 150+ engineered features across:  
              - TMDB metadata (runtime, budget, popularity, release lag)  
              - Rotten Tomatoes critic signals  
              - YouTube trailer engagement metrics (views, likes, comment ratio, recency)  
              - Gender diversity indicators (female cast share, representation alignment)  
            - Strong baseline for tabular prediction of audience scores  
            """
        )

        st.markdown("#### Individual Model Performance (on validation set)")
        perf_df = pd.DataFrame({
            "Model": ["GNN", "KGCN", "XGBoost"],
            "RMSE": [0.1929, 0.1709, 0.1097]
        })
        fig = px.bar(
            perf_df,
            x="Model",
            y="RMSE",
            title="Individual Model RMSE",
            text="RMSE"
        )
        fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
        fig.update_layout(yaxis_title="RMSE", uniformtext_minsize=10, uniformtext_mode='hide')
        st.plotly_chart(fig, use_container_width=True)

    # TAB 2: Ensemble Model
    with tab2:
        st.subheader("Stacking Ensemble Model")

        if stacking_model is not None:
            st.success("Stacking meta-learner loaded successfully.")
        else:
            st.warning("Stacking model not found. The app falls back to a weighted average ensemble.")

        st.write(
            """
            The final audience score prediction is produced by a stacking ensemble that combines:
            - GNN predictions (graph-based similarity)  
            - KGCN predictions (relational knowledge graph)  
            - XGBoost predictions (tabular features)  

            Instead of a fixed average, the meta-learner is trained to learn how much to trust
            each base model in different regions of the prediction space.
            """
        )

        if stacking_model is not None:
            meta_cols = st.columns(3)
            with meta_cols[0]:
                model_name = ensemble_meta.get('meta_model', 'Gradient Boosting')
                st.metric("Meta-Model", model_name)
            with meta_cols[1]:
                rmse = ensemble_meta.get('stacking_rmse', 0.1085)
                st.metric("Stacking RMSE", f"{rmse:.4f}")
            with meta_cols[2]:
                st.metric("Accuracy (±10%)", "80.2%")

            st.info(
                """
                Workflow:
                1. Train GNN, KGCN, and XGBoost on the same set of labeled films.  
                2. Collect their predictions and use them as features in a new model.  
                3. Train the meta-learner (e.g., Gradient Boosting) on these predictions vs. true scores.  
                4. Use this meta-learner at inference time for new movies.  
                """
            )

        st.markdown("---")
        st.subheader("Coverage of Base Predictions")

        cov1, cov2, cov3 = st.columns(3)
        with cov1:
            gnn_preds_len = len(artifacts['gnn_preds']) if artifacts and artifacts.get('gnn_preds') is not None else 0
            st.metric("GNN Predictions", f"{gnn_preds_len:,}")
        with cov2:
            kgcn_preds_len = len(artifacts['kgcn_preds']) if artifacts and artifacts.get('kgcn_preds') is not None else 0
            st.metric("KGCN Predictions", f"{kgcn_preds_len:,}")
        with cov3:
            xgb_preds_len = len(artifacts['xgb_preds']) if artifacts and artifacts.get('xgb_preds') is not None else 0
            st.metric("XGBoost Predictions", f"{xgb_preds_len:,}")

        st.markdown("---")
        st.subheader("Original vs Stacking Comparison")

        comparison_data = {
            'Method': ['Original (Weighted Avg)', 'Stacking (Gradient Boosting)'],
            'RMSE': [0.1452, ensemble_meta.get('stacking_rmse', 0.0996)],
            'MAE': [0.1170, 0.0669],
            'Within ±10%': ['48.9%', '80.2%'],
            'Within ±5%': ['24.2%', '54.5%']
        }
        st.dataframe(pd.DataFrame(comparison_data), hide_index=True, use_container_width=True)
        st.caption(
            "Stacking substantially improves RMSE and the fraction of movies predicted within 5–10 percentage points of the true audience score."
        )

    # TAB 3: Data Pipeline
    with tab3:
        st.subheader("Data Pipeline")

        st.markdown("#### Data Sources")
        st.write(
            """
            - **TMDB (≈66k films)**  
              Genres, runtime, cast, crew, popularity, budget, release info, posters  
            - **Rotten Tomatoes (≈6.8k films)**  
              Audience and critic scores, critic review snippets  
            - **YouTube API (≈42k trailers)**  
              Trailer metadata, engagement metrics (views, likes, comments), recency  
            """
        )

        st.markdown("#### Cleaning and Integration")
        st.write(
            """
            - Join on TMDB IDs and (title, year) where necessary  
            - De-duplicate trailers with heuristics and URL patterns  
            - Normalize numeric features (e.g., log-transform budget and popularity)  
            - Compute sentiment scores for critic quotes and trailer comments  
            - Derive gender representation signals (e.g., female cast share, alignment with overall cast)  
            - Store the final integrated dataset in MongoDB with a structured movie document schema  
            """
        )

        st.markdown("#### High-Level Schema (Conceptual)")
        st.graphviz_chart(
            """
            digraph {
                rankdir=LR;
                TMDB -> Merge;
                RottenTomatoes -> Merge;
                YouTube -> Merge;
                Merge -> MongoDB;
                MongoDB -> "Feature Engineering";
                "Feature Engineering" -> GNN;
                "Feature Engineering" -> XGBoost;
                "Feature Engineering" -> KGCN;
                GNN -> Ensemble;
                KGCN -> Ensemble;
                XGBoost -> Ensemble;
                Ensemble -> "Audience Score Prediction";
            }
            """
        )


def visual_graph_explorer_page(db):
    st.title("Visual Graph Explorer")
    st.markdown("Explore a local similarity neighborhood around a selected film.")

    titles = get_all_movie_titles(db)
    if not titles:
        st.info("No titles available for graph visualization.")
        return

    choice = st.selectbox("Select a film:", titles)
    movie = search_movie(db, choice)
    if not movie:
        st.error("Selected movie not found in database.")
        return

    genres = movie["production"].get("genres", [])
    if not genres:
        st.info("Selected movie has no genre information; cannot construct similarity graph.")
        return

    # Get neighbors from same primary genre
    primary_genre = genres[0]
    neighbors = get_movies_by_genre(db, primary_genre, limit=20)

    # Build a simple graph: center movie + neighbors
    G = nx.Graph()
    center_id = movie["tmdb_id"]
    G.add_node(center_id, label=choice, group="center")

    for m in neighbors:
        mid = m["tmdb_id"]
        if mid == center_id:
            continue
        G.add_node(mid, label=m["title"], group="neighbor")
        G.add_edge(center_id, mid)

    if G.number_of_nodes() <= 1:
        st.info("Not enough neighbors in this genre for a graph.")
        return

    # Layout
    pos = nx.spring_layout(G, seed=42, k=0.6)

    # Build Plotly scatter for nodes
    node_x = []
    node_y = []
    node_text = []
    node_color = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]["label"])
        if node == center_id:
            node_color.append("crimson")
        else:
            node_color.append("steelblue")

    # Edges
    edge_x = []
    edge_y = []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1),
        hoverinfo="none",
        mode="lines"
    )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=node_text,
        textposition="top center",
        hoverinfo="text",
        marker=dict(
            size=[16 if t == choice else 10 for t in node_text]
        )
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        title=f"Local Similarity Graph for “{choice}” (genre: {primary_genre})",
        showlegend=False,
        margin=dict(l=20, r=20, t=60, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=600
    )

    st.plotly_chart(fig, use_container_width=True)


def acknowledgements_page():
    st.title("Team and Acknowledgements")

    st.subheader("Team 15 — Cinemaniacs")
    st.write(
        """
        - Angelina Cottone  
        - Nidhi Deshmukh  
        - Dylan Sidhu  
        - Matthew Ward  
        - Clara Wei  
        """
    )

    st.subheader("Data Sources")
    st.write(
        """
        - TMDB API  
        - Rotten Tomatoes (web-scraped)  
        - YouTube Data API  
        """
    )

    st.subheader("Acknowledgements")
    st.write(
        """
        We thank the STA 160 instructional team for guidance and support throughout the project, 
        as well as collaborative tools used for development, organization, and research.
        """
    )


# =============================================================================
# MAIN APP
# =============================================================================

def main():
    db = get_database_connection()
    if db is None:
        st.error("Unable to connect to database. Please check your MongoDB credentials.")
        st.stop()

    artifacts, error = load_ensemble_artifacts()
    if error:
        st.error(f"Ensemble artifacts failed to load: {error}")
        st.stop()
    else:
        st.sidebar.success("Artifacts loaded")

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        [
            "Home",
            "Movie Search",
            "Compare Movies",
            "Analytics Dashboard",
            "Modeling",
            "Visual Graph Explorer",
            "Acknowledgements"
        ]
    )

    if page == "Home":
        home_page(db)
    elif page == "Movie Search":
        movie_search_page(db, artifacts)
    elif page == "Compare Movies":
        compare_movies_page(db, artifacts)
    elif page == "Analytics Dashboard":
        analytics_page(db)
    elif page == "Modeling":
        modeling_page(artifacts)
    elif page == "Visual Graph Explorer":
        visual_graph_explorer_page(db)
    elif page == "Acknowledgements":
        acknowledgements_page()

    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; font-size: 0.9rem; color: #9ca3af;'>
            <p>Filmlytics | STA 160 Project | Team 15</p>
            <p>Ensemble Audience Score Prediction Platform</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

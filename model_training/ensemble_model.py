"""
Ensemble Model for Cinemaniacs Movie Success Prediction

This script loads pre-computed predictions from three models:
- GNN (Graph Neural Network)
- KGCN (Knowledge Graph Convolutional Network)
- XGBoost (Gradient Boosting)

And combines them using optimized weights to produce final ensemble predictions.
"""

import os
import pandas as pd
import numpy as np
from pymongo import MongoClient
from pymongo.server_api import ServerApi
import certifi
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
import pickle

# =============================================================================
# CONFIGURATION
# =============================================================================

ARTIFACT_DIR = os.path.join(os.path.dirname(__file__), 'model_artifacts')

# MongoDB connection
MONGODB_URI = "mongodb+srv://cinemaniacs:filmlytics@filmlytics.1emhcue.mongodb.net/?appName=filmlytics"

# Default weights (will be optimized)
DEFAULT_WEIGHTS = {
    'gnn': 0.33,
    'kgcn': 0.34,
    'xgb': 0.33
}

# Prediction file mappings
PREDICTION_FILES = {
    'gnn': ('gnn_preds_all_movies.csv', 'gnn_pred_audience_score'),
    'kgcn': ('kgcn_preds_all_movies.csv', 'pred_audience_score'),
    'xgb': ('xgb_preds_all_movies.csv', 'predicted_audience_score'),
}


# =============================================================================
# LOAD PREDICTIONS
# =============================================================================

def load_predictions():
    """Load all prediction CSVs and return as dict of Series indexed by tmdb_id."""
    predictions = {}
    
    for model_name, (filename, pred_col) in PREDICTION_FILES.items():
        filepath = os.path.join(ARTIFACT_DIR, filename)
        
        if not os.path.exists(filepath):
            print(f"⚠️  Warning: {filename} not found. {model_name.upper()} predictions will be missing.")
            predictions[model_name] = None
            continue
        
        try:
            df = pd.read_csv(filepath)
            df['tmdb_id'] = pd.to_numeric(df['tmdb_id'], errors='coerce').fillna(0).astype(int)
            predictions[model_name] = df.set_index('tmdb_id')[pred_col]
            print(f"✅ Loaded {model_name.upper()}: {len(predictions[model_name]):,} predictions")
        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            predictions[model_name] = None
    
    return predictions


def get_ensemble_prediction(tmdb_id, predictions, weights=DEFAULT_WEIGHTS):
    """
    Get ensemble prediction for a single movie.
    
    Args:
        tmdb_id: The TMDB ID of the movie
        predictions: Dict of prediction Series from load_predictions()
        weights: Dict of model weights
        
    Returns:
        tuple: (ensemble_pred, breakdown_dict)
    """
    breakdown = {}
    
    for model_name, preds in predictions.items():
        if preds is None:
            breakdown[model_name] = np.nan
        else:
            pred = preds.get(tmdb_id, np.nan)
            # Handle Series result (duplicate IDs)
            if isinstance(pred, pd.Series):
                pred = pred.iloc[0] if len(pred) > 0 else np.nan
            breakdown[model_name] = float(pred) if not pd.isna(pred) else np.nan
    
    # Calculate weighted average of valid predictions
    valid_preds = {k: v for k, v in breakdown.items() if not np.isnan(v)}
    
    if not valid_preds:
        return np.nan, breakdown
    
    # Renormalize weights for available models
    total_weight = sum(weights[k] for k in valid_preds.keys())
    ensemble_pred = sum(valid_preds[k] * (weights[k] / total_weight) for k in valid_preds.keys())
    
    return float(np.clip(ensemble_pred, 0, 1)), breakdown


def generate_all_ensemble_predictions(predictions, weights=DEFAULT_WEIGHTS):
    """
    Generate ensemble predictions for all movies.
    
    Returns:
        DataFrame with tmdb_id, ensemble_pred, gnn_pred, kgcn_pred, xgb_pred
    """
    # Get all unique tmdb_ids across all models
    all_ids = set()
    for preds in predictions.values():
        if preds is not None:
            all_ids.update(preds.index.tolist())
    
    print(f"\nGenerating ensemble predictions for {len(all_ids):,} movies...")
    
    results = []
    for tmdb_id in all_ids:
        ensemble_pred, breakdown = get_ensemble_prediction(tmdb_id, predictions, weights)
        results.append({
            'tmdb_id': tmdb_id,
            'ensemble_pred': ensemble_pred,
            'gnn_pred': breakdown.get('gnn', np.nan),
            'kgcn_pred': breakdown.get('kgcn', np.nan),
            'xgb_pred': breakdown.get('xgb', np.nan),
        })
    
    df = pd.DataFrame(results)
    return df


# =============================================================================
# WEIGHT OPTIMIZATION
# =============================================================================

def load_actual_scores():
    """Load actual audience scores from MongoDB."""
    print("\nConnecting to MongoDB...")
    client = MongoClient(
        MONGODB_URI,
        server_api=ServerApi('1'),
        tlsCAFile=certifi.where()
    )
    db = client['cinemaniacs']

    # Get movies with audience scores
    movies = list(db.movies.find(
        {"rotten_tomatoes.audience_score": {"$exists": True, "$ne": None}},
        {"tmdb_id": 1, "rotten_tomatoes.audience_score": 1}
    ))

    print(f"Found {len(movies):,} movies with audience scores")

    # Parse audience scores
    scores = {}
    for m in movies:
        tmdb_id = m.get('tmdb_id')
        score_str = m.get('rotten_tomatoes', {}).get('audience_score')
        if tmdb_id and score_str:
            # Parse "87%" -> 0.87
            if isinstance(score_str, str) and score_str.endswith('%'):
                try:
                    scores[tmdb_id] = float(score_str[:-1]) / 100.0
                except:
                    pass
            elif isinstance(score_str, (int, float)):
                scores[tmdb_id] = float(score_str) / 100.0 if score_str > 1 else float(score_str)

    return scores


def optimize_weights(predictions, actual_scores):
    """
    Find optimal ensemble weights using grid search.
    Minimizes RMSE on movies with known audience scores.
    """
    print("\n" + "=" * 60)
    print("OPTIMIZING ENSEMBLE WEIGHTS")
    print("=" * 60)

    # Helper to safely get a scalar prediction
    def safe_get_pred(preds, tmdb_id):
        if preds is None:
            return None
        val = preds.get(tmdb_id)
        if val is None:
            return None
        # Handle duplicate tmdb_ids returning Series
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) > 0 else None
        if pd.isna(val):
            return None
        return float(val)

    # Build aligned arrays for movies with all predictions and actual scores
    valid_ids = []
    gnn_vals, kgcn_vals, xgb_vals, actual_vals = [], [], [], []

    for tmdb_id, actual in actual_scores.items():
        gnn = safe_get_pred(predictions['gnn'], tmdb_id)
        kgcn = safe_get_pred(predictions['kgcn'], tmdb_id)
        xgb = safe_get_pred(predictions['xgb'], tmdb_id)

        # Only include if all 3 models have predictions
        if gnn is not None and kgcn is not None and xgb is not None:
            valid_ids.append(tmdb_id)
            gnn_vals.append(gnn)
            kgcn_vals.append(kgcn)
            xgb_vals.append(xgb)
            actual_vals.append(actual)

    print(f"Movies with all 3 predictions + actual score: {len(valid_ids):,}")

    if len(valid_ids) < 100:
        print("⚠️  Not enough data for optimization. Using default weights.")
        return DEFAULT_WEIGHTS

    gnn_arr = np.array(gnn_vals)
    kgcn_arr = np.array(kgcn_vals)
    xgb_arr = np.array(xgb_vals)
    actual_arr = np.array(actual_vals)

    # Grid search with minimum weight constraint
    MIN_WEIGHT = 0.05  # Each model gets at least 5%
    best_rmse = float('inf')
    best_weights = None

    print("\nRunning grid search (min weight per model: 5%)...")
    for w_gnn in np.arange(MIN_WEIGHT, 1.0 - 2*MIN_WEIGHT + 0.01, 0.05):
        for w_kgcn in np.arange(MIN_WEIGHT, 1.0 - w_gnn - MIN_WEIGHT + 0.01, 0.05):
            w_xgb = 1.0 - w_gnn - w_kgcn
            if w_xgb >= MIN_WEIGHT:
                # Calculate weighted ensemble
                ensemble = w_gnn * gnn_arr + w_kgcn * kgcn_arr + w_xgb * xgb_arr
                rmse = np.sqrt(np.mean((ensemble - actual_arr) ** 2))

                if rmse < best_rmse:
                    best_rmse = rmse
                    best_weights = {'gnn': w_gnn, 'kgcn': w_kgcn, 'xgb': w_xgb}

    # Also calculate individual model RMSEs
    rmse_gnn = np.sqrt(np.mean((gnn_arr - actual_arr) ** 2))
    rmse_kgcn = np.sqrt(np.mean((kgcn_arr - actual_arr) ** 2))
    rmse_xgb = np.sqrt(np.mean((xgb_arr - actual_arr) ** 2))

    print("\n" + "-" * 40)
    print("INDIVIDUAL MODEL PERFORMANCE:")
    print("-" * 40)
    print(f"  GNN   RMSE: {rmse_gnn:.4f}")
    print(f"  KGCN  RMSE: {rmse_kgcn:.4f}")
    print(f"  XGB   RMSE: {rmse_xgb:.4f}")

    print("\n" + "-" * 40)
    print("OPTIMAL ENSEMBLE WEIGHTS:")
    print("-" * 40)
    print(f"  GNN:  {best_weights['gnn']*100:.1f}%")
    print(f"  KGCN: {best_weights['kgcn']*100:.1f}%")
    print(f"  XGB:  {best_weights['xgb']*100:.1f}%")
    print(f"\n  Ensemble RMSE: {best_rmse:.4f}")
    print(f"  Improvement over best single: {min(rmse_gnn, rmse_kgcn, rmse_xgb) - best_rmse:.4f}")

    return best_weights


def train_stacking_model(predictions, actual_scores):
    """
    Train a stacking meta-learner that learns how to combine base model predictions.
    Uses cross-validation to prevent overfitting.
    """
    print("\n" + "=" * 60)
    print("TRAINING STACKING META-LEARNER")
    print("=" * 60)

    # Helper to safely get a scalar prediction
    def safe_get_pred(preds, tmdb_id):
        if preds is None:
            return None
        val = preds.get(tmdb_id)
        if val is None:
            return None
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) > 0 else None
        if pd.isna(val):
            return None
        return float(val)

    # Build feature matrix (base model predictions) and target
    X_list, y_list, ids_list = [], [], []

    for tmdb_id, actual in actual_scores.items():
        gnn = safe_get_pred(predictions['gnn'], tmdb_id)
        kgcn = safe_get_pred(predictions['kgcn'], tmdb_id)
        xgb = safe_get_pred(predictions['xgb'], tmdb_id)

        if gnn is not None and kgcn is not None and xgb is not None:
            X_list.append([gnn, kgcn, xgb])
            y_list.append(actual)
            ids_list.append(tmdb_id)

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Training samples: {len(X):,}")

    # Try multiple meta-learners
    models = {
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Ridge (α=10.0)': Ridge(alpha=10.0),
        'Linear Regression': LinearRegression(),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=50, max_depth=3, learning_rate=0.1, random_state=42
        )
    }

    print("\nCross-validation results (5-fold):")
    print("-" * 50)

    best_model = None
    best_score = float('inf')
    best_name = None

    kfold = KFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        # Negative MSE (sklearn convention), convert to RMSE
        scores = cross_val_score(model, X, y, cv=kfold, scoring='neg_mean_squared_error')
        rmse = np.sqrt(-scores.mean())
        std = np.sqrt(-scores).std()

        print(f"  {name:25s} RMSE: {rmse:.4f} (±{std:.4f})")

        if rmse < best_score:
            best_score = rmse
            best_model = model
            best_name = name

    print("-" * 50)
    print(f"  Best: {best_name} with RMSE {best_score:.4f}")

    # Train final model on all data
    print(f"\nTraining final {best_name} model on all data...")
    best_model.fit(X, y)

    # Show learned coefficients for linear models
    if hasattr(best_model, 'coef_'):
        coefs = best_model.coef_
        intercept = best_model.intercept_
        print("\n" + "-" * 40)
        print("LEARNED STACKING COEFFICIENTS:")
        print("-" * 40)
        print(f"  GNN coefficient:  {coefs[0]:+.4f}")
        print(f"  KGCN coefficient: {coefs[1]:+.4f}")
        print(f"  XGB coefficient:  {coefs[2]:+.4f}")
        print(f"  Intercept:        {intercept:+.4f}")
        print(f"\n  Formula: pred = {coefs[0]:.3f}×GNN + {coefs[1]:.3f}×KGCN + {coefs[2]:.3f}×XGB + {intercept:.3f}")

    return best_model, best_name, best_score


def generate_stacking_predictions(predictions, meta_model):
    """Generate predictions using the stacking meta-learner."""

    # Helper to safely get a scalar prediction
    def safe_get_pred(preds, tmdb_id):
        if preds is None:
            return None
        val = preds.get(tmdb_id)
        if val is None:
            return None
        if isinstance(val, pd.Series):
            val = val.iloc[0] if len(val) > 0 else None
        if pd.isna(val):
            return None
        return float(val)

    # Get all unique tmdb_ids
    all_ids = set()
    for preds in predictions.values():
        if preds is not None:
            all_ids.update(preds.index.tolist())

    print(f"\nGenerating stacking predictions for {len(all_ids):,} movies...")

    results = []
    for tmdb_id in all_ids:
        gnn = safe_get_pred(predictions['gnn'], tmdb_id)
        kgcn = safe_get_pred(predictions['kgcn'], tmdb_id)
        xgb = safe_get_pred(predictions['xgb'], tmdb_id)

        # Need all 3 predictions for stacking
        if gnn is not None and kgcn is not None and xgb is not None:
            X = np.array([[gnn, kgcn, xgb]])
            ensemble_pred = float(np.clip(meta_model.predict(X)[0], 0, 1))
        else:
            # Fallback: average of available predictions
            valid = [p for p in [gnn, kgcn, xgb] if p is not None]
            ensemble_pred = np.mean(valid) if valid else np.nan

        results.append({
            'tmdb_id': tmdb_id,
            'ensemble_pred': ensemble_pred,
            'gnn_pred': gnn if gnn is not None else np.nan,
            'kgcn_pred': kgcn if kgcn is not None else np.nan,
            'xgb_pred': xgb if xgb is not None else np.nan,
        })

    return pd.DataFrame(results)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import json

    print("=" * 60)
    print("CINEMANIACS ENSEMBLE MODEL")
    print("=" * 60)

    # Load predictions
    print("\nLoading model predictions...")
    predictions = load_predictions()

    # Load actual scores
    actual_scores = load_actual_scores()

    # =========================================================================
    # METHOD 1: Grid Search (Simple Weighted Average)
    # =========================================================================
    optimal_weights = optimize_weights(predictions, actual_scores)

    # =========================================================================
    # METHOD 2: Stacking (Learned Meta-Model)
    # =========================================================================
    meta_model, model_name, stacking_rmse = train_stacking_model(predictions, actual_scores)

    # Save the stacking model
    meta_model_path = os.path.join(ARTIFACT_DIR, 'stacking_meta_model.pkl')
    with open(meta_model_path, 'wb') as f:
        pickle.dump(meta_model, f)
    print(f"\nSaved stacking model to: {meta_model_path}")

    # =========================================================================
    # Generate predictions using STACKING (better method)
    # =========================================================================
    print("\n" + "=" * 60)
    print("GENERATING STACKING ENSEMBLE PREDICTIONS")
    print("=" * 60)
    ensemble_df = generate_stacking_predictions(predictions, meta_model)

    # Save to CSV
    output_path = os.path.join(ARTIFACT_DIR, 'ensemble_preds_all_movies.csv')
    ensemble_df.to_csv(output_path, index=False)

    # Save metadata
    weights_path = os.path.join(ARTIFACT_DIR, 'ensemble_weights.json')
    meta_info = {
        'method': 'stacking',
        'meta_model': model_name,
        'stacking_rmse': float(stacking_rmse),
        'grid_search_weights': optimal_weights
    }
    with open(weights_path, 'w') as f:
        json.dump(meta_info, f, indent=2)
    print(f"Saved ensemble metadata to: {weights_path}")

    # Summary stats
    print("\n" + "=" * 60)
    print("ENSEMBLE PREDICTIONS SUMMARY")
    print("=" * 60)
    print(f"Total movies: {len(ensemble_df):,}")
    print(f"Valid ensemble predictions: {ensemble_df['ensemble_pred'].notna().sum():,}")
    print(f"Prediction range: [{ensemble_df['ensemble_pred'].min():.4f}, {ensemble_df['ensemble_pred'].max():.4f}]")
    print(f"Mean prediction: {ensemble_df['ensemble_pred'].mean():.4f}")
    print(f"\nSaved to: {output_path}")
    print("=" * 60)

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json
from typing import Dict, Tuple
from datetime import datetime
import joblib 
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    roc_curve, precision_recall_curve, mean_absolute_error
)
import xgboost as xgb
import lightgbm as lgb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NetflixEngagementPredictor:
    """
    Predicts Netflix engagement decline risk based on content quality metrics
    """
    
    def __init__(self, data_dir: str = "./data", model_dir: str = "./models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.models = {}
        self.scaler = StandardScaler()
        self.feature_importance = {}
        
    def load_data(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load weekly features dataset and most recent top 10 movies"""
        logger.info("Loading weekly features dataset...")
        
        df = pd.read_csv(self.data_dir / "weekly_features.csv")
        df['week'] = pd.to_datetime(df['week'])
        
        logger.info(f"✓ Loaded {len(df)} weeks of data")
        logger.info(f"  Date range: {df['week'].min()} to {df['week'].max()}")
        logger.info(f"  Target distribution: {df['engagement_decline'].value_counts().to_dict()}")
        
        logger.info("Loading most recent top 10 movies...")
        top10_df = pd.read_csv(self.data_dir / "netflix_top10_raw.csv")
        top10_df['week'] = pd.to_datetime(top10_df['week'])
        
        most_recent_week = top10_df['week'].max()
        recent_top10 = top10_df[top10_df['week'] == most_recent_week].copy()
        
        recent_top10 = recent_top10.sort_values('ranking')
        
        logger.info(f"✓ Loaded top 10 movies for week: {most_recent_week.date()}")
        logger.info(f"  Movies: {', '.join(recent_top10['title'].tolist()[:3])}...")
        
        return df, recent_top10
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, list]:
        """Prepare features for modeling"""
        logger.info("Preparing features...")
        
        feature_cols = [
            'hours_viewed_sum',
            'hours_viewed_mean',
            'hours_viewed_std',
            'weeks_in_top10_mean',
            'ranking_nunique',
            
            'avg_rating_mean',
            'rating_std_mean',
            'vader_compound_mean_mean',
            'vader_positive_mean_mean',
            'vader_negative_mean_mean',
            'textblob_polarity_mean_mean',
            'num_reviews_with_text_sum',
            
            'hours_lag_1',
            'hours_lag_2',
            'hours_lag_3',
            'hours_lag_4',
            'sentiment_lag_1',
            'sentiment_lag_2',
            'sentiment_lag_3',
            'sentiment_lag_4',
            
            'hours_ma_4week',
            'sentiment_ma_4week',
            
            'month',
            'week_of_year'
        ]
        
        available_features = [col for col in feature_cols if col in df.columns]
        logger.info(f"Using {len(available_features)} features")
        
        df_clean = df[df['engagement_decline'].notna()].copy()
        
        X = df_clean[available_features].fillna(method='ffill').fillna(0)
        y = df_clean['engagement_decline']
        
        logger.info(f"✓ Prepared {len(X)} samples with {len(available_features)} features")
        logger.info(f"  Class balance: {y.value_counts(normalize=True).to_dict()}")
        
        return X, y, available_features
    
    def train_baseline_models(self, X: pd.DataFrame, y: pd.Series, 
                             feature_names: list) -> Dict:
        """Train and evaluate baseline models"""
        logger.info("=" * 70)
        logger.info("TRAINING BASELINE MODELS")
        logger.info("=" * 70)
        
        tscv = TimeSeriesSplit(n_splits=5)
        
        results = {}
        
        naive_pred = np.zeros(len(y))
        naive_acc = (naive_pred == y).mean()
        results['naive'] = {
            'accuracy': naive_acc,
            'description': 'Always predicts no engagement decline'
        }
        logger.info(f"\n1. Naive Baseline Accuracy: {naive_acc:.3f}")
        
        logger.info("\n2. Training Logistic Regression...")
        lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        lr_scores = cross_val_score(lr, X, y, cv=tscv, scoring='roc_auc')
        lr.fit(X, y)
        
        results['logistic_regression'] = {
            'model': lr,
            'cv_auc_mean': lr_scores.mean(),
            'cv_auc_std': lr_scores.std(),
            'cv_scores': lr_scores.tolist() 
        }
        
        self.models['logistic_regression'] = lr
        
        logger.info(f"   AUC-ROC: {lr_scores.mean():.3f} (+/- {lr_scores.std():.3f})")
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': np.abs(lr.coef_[0])
        }).sort_values('coefficient', ascending=False)
        
        logger.info("\n   Top 5 Features:")
        for idx, row in feature_importance.head(5).iterrows():
            logger.info(f"     {row['feature']}: {row['coefficient']:.3f}")
        
        logger.info("\n3. Training Random Forest...")
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        )
        rf_scores = cross_val_score(rf, X, y, cv=tscv, scoring='roc_auc')
        rf.fit(X, y)
        
        results['random_forest'] = {
            'model': rf,
            'cv_auc_mean': rf_scores.mean(),
            'cv_auc_std': rf_scores.std(),
            'cv_scores': rf_scores.tolist()
        }
        
        self.models['random_forest'] = rf
        logger.info(f"   AUC-ROC: {rf_scores.mean():.3f} (+/- {rf_scores.std():.3f})")
        
        logger.info("\n4. Training XGBoost...")
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
            eval_metric='auc'
        )
        xgb_scores = cross_val_score(xgb_model, X, y, cv=tscv, scoring='roc_auc')
        xgb_model.fit(X, y)
        
        results['xgboost'] = {
            'model': xgb_model,
            'cv_auc_mean': xgb_scores.mean(),
            'cv_auc_std': xgb_scores.std(),
            'cv_scores': xgb_scores.tolist()
        }
        
        self.models['xgboost'] = xgb_model
        logger.info(f"   AUC-ROC: {xgb_scores.mean():.3f} (+/- {xgb_scores.std():.3f})")
        
        self.feature_importance['xgboost'] = pd.DataFrame({
            'feature': feature_names,
            'importance': xgb_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        logger.info("\n   Top 5 Features:")
        for idx, row in self.feature_importance['xgboost'].head(5).iterrows():
            logger.info(f"     {row['feature']}: {row['importance']:.3f}")
        
        logger.info("\n5. Training LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            class_weight='balanced',
            verbose=-1
        )
        lgb_scores = cross_val_score(lgb_model, X, y, cv=tscv, scoring='roc_auc')
        lgb_model.fit(X, y)
        
        results['lightgbm'] = {
            'model': lgb_model,
            'cv_auc_mean': lgb_scores.mean(),
            'cv_auc_std': lgb_scores.std(),
            'cv_scores': lgb_scores.tolist()
        }
        
        self.models['lightgbm'] = lgb_model
        logger.info(f"   AUC-ROC: {lgb_scores.mean():.3f} (+/- {lgb_scores.std():.3f})")
        
        return results
    
    def evaluate_on_test_set(self, X: pd.DataFrame, y: pd.Series, 
                            test_size: int = 10) -> Dict:
        """Evaluate models on held-out test set (last N weeks)"""
        logger.info("\n" + "=" * 70)
        logger.info(f"EVALUATING ON TEST SET (Last {test_size} weeks)")
        logger.info("=" * 70)
        
        X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
        y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
        
        logger.info(f"Train set: {len(X_train)} weeks")
        logger.info(f"Test set: {len(X_test)} weeks")
        logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
        
        results = {}
        
        for model_name, model in self.models.items():
            logger.info(f"\n{model_name.upper()}:")
            model_clone = clone(model)
            model_clone.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            
            auc = roc_auc_score(y_test, y_pred_proba)
            
            logger.info(f"  AUC-ROC: {auc:.3f}")
            logger.info("\n  Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['No Decline', 'Decline']))
            
            results[model_name] = {
                'auc': auc,
                # predictions and true_values are not needed for saving to S3 for API
                # but kept here for local testing.
            }
        
        return results
    
    def predict_next_week(self, current_week_features: Dict) -> Dict:
        """Predict engagement decline risk for next week"""
        
        features_df = pd.DataFrame([current_week_features])
        
        predictions = {}
        
        for model_name, model in self.models.items():
            pred_proba = model.predict_proba(features_df)[0, 1]
            pred_class = model.predict(features_df)[0]
            
            predictions[model_name] = {
                'decline_probability': float(pred_proba),
                'prediction': 'DECLINE' if pred_class == 1 else 'NO DECLINE',
                'risk_level': self._risk_level(pred_proba)
            }
        
        avg_proba = np.mean([p['decline_probability'] for p in predictions.values()])
        
        result = {
            'individual_models': predictions,
            'ensemble': {
                'decline_probability': float(avg_proba),
                'prediction': 'DECLINE' if avg_proba > 0.5 else 'NO DECLINE',
                'risk_level': self._risk_level(avg_proba)
            }
        }
        
        return result
    
    def _risk_level(self, probability: float) -> str:
        """Convert probability to risk level"""
        if probability >= 0.7:
            return 'HIGH'
        elif probability >= 0.4:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def generate_insights(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Generate business insights from model"""
        logger.info("\n" + "=" * 70)
        logger.info("GENERATING BUSINESS INSIGHTS")
        logger.info("=" * 70)
        
        model = self.models['xgboost']
        
        top_features = self.feature_importance['xgboost'].head(10)
        
        logger.info("\nTop 10 Most Important Features:")
        for idx, row in top_features.iterrows():
            logger.info(f"  {row['feature']}: {row['importance']:.3f}")
        
        engagement_cols = ['hours_viewed_sum', 'hours_pct_change', 'engagement_decline']
        sentiment_cols = ['avg_rating_mean', 'vader_compound_mean_mean', 
                         'textblob_polarity_mean_mean']
        
        correlation_data = X.copy()
        correlation_data['engagement_decline'] = y
        
        logger.info("\nCorrelation with Engagement Decline:")
        for col in sentiment_cols:
            if col in correlation_data.columns:
                corr = correlation_data[col].corr(correlation_data['engagement_decline'])
                logger.info(f"  {col}: {corr:.3f}")
        
        insights = {
            'top_features': top_features.to_dict('records'),
            'key_findings': self._generate_key_findings(X, y, top_features)
        }
        
        return insights
    
    def _generate_key_findings(self, X: pd.DataFrame, y: pd.Series, 
                               top_features: pd.DataFrame) -> list:
        """Generate human-readable key findings"""
        findings = []
        
        sentiment_features = top_features[
            top_features['feature'].str.contains('rating|vader|textblob', case=False)
        ]
        
        if len(sentiment_features) > 0:
            findings.append(
                f"Content quality metrics rank among top {len(sentiment_features)} "
                f"predictors of engagement decline"
            )
        
        lag_features = top_features[top_features['feature'].str.contains('lag', case=False)]
        if len(lag_features) > 0:
            findings.append(
                f"Historical engagement patterns (lagged features) are strong predictors"
            )
        
        if 'vader_compound_mean_mean' in X.columns:
            corr = X['vader_compound_mean_mean'].corr(y)
            if abs(corr) > 0.3:
                direction = "Negative" if corr > 0 else "Positive"
                findings.append(
                    f"{direction} sentiment shows {abs(corr):.1%} correlation "
                    f"with engagement decline"
                )
        
        return findings
    
    def save_models(self):
        """Save trained models"""
        logger.info("\nSaving models...")
        
        for model_name, model in self.models.items():
            model_path = self.model_dir / f"{model_name}_model.pkl"
            
            joblib.dump(model, model_path)
            logger.info(f"✓ Saved {model_name} to {model_path}")
        
        importance_path = self.model_dir / "feature_importance.json"
        serializable_importance = {
            k: v.to_dict('records') for k, v in self.feature_importance.items()
        }
        
        with open(importance_path, 'w') as f:
            json.dump(serializable_importance, f, indent=2)
            
        logger.info(f"✓ Saved feature importance to {importance_path}")


def main():
    """Main execution"""
    
    predictor = NetflixEngagementPredictor(
        data_dir="./data",
        model_dir="./models"
    )
    
    df, top_10 = predictor.load_data()
    
    X, y, feature_names = predictor.prepare_features(df)
    test_size = min(25, len(X) // 5)

    X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
    y_train, y_test = y.iloc[:-test_size], y.iloc[-test_size:]
    training_results = predictor.train_baseline_models(X, y, feature_names)
    
    test_results = predictor.evaluate_on_test_set(X, y, test_size=test_size)
    
    insights = predictor.generate_insights(X, y)
    
    predictor.save_models()
    
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING AND SAVING API PAYLOAD")
    logger.info("=" * 70)
    
    last_week_features = X.iloc[-1].to_dict()
    prediction = predictor.predict_next_week(last_week_features)
    
    serializable_training_results = {}
    for model_name, res in training_results.items():
        if model_name != 'naive':
            serializable_training_results[model_name] = {
                'cv_auc_mean': float(res.get('cv_auc_mean', 'N/A')),
                'cv_auc_std': float(res.get('cv_auc_std', 'N/A')),
            }
    
    serializable_feature_importance = {
        k: v.to_dict('records') for k, v in predictor.feature_importance.items()
    }
    
    latest_features_context = {
        'date': X.index[-1].strftime('%Y-%m-%d') if isinstance(X.index, pd.DatetimeIndex) else 'N/A',
        'hours_viewed_sum': float(last_week_features.get('hours_viewed_sum', 0)),
        'sentiment_lag_2': float(last_week_features.get('sentiment_lag_2', 0))
    }

    top_10_serializable = top_10.copy()
    top_10_serializable['week'] = top_10_serializable['week'].dt.strftime('%Y-%m-%d')


    api_payload = {
        'status': 'success',
        'prediction_date': datetime.now().strftime('%Y-%m-%d'),
        'ensemble_prediction': prediction['ensemble'],
        'top_predictive_features': serializable_feature_importance,
        'latest_data_context': latest_features_context,
        'model_performance_summary': serializable_training_results,
        'recent_top_10_movies': top_10_serializable.to_dict('records')
    }

    output_path = predictor.model_dir / "latest_prediction.json"
    with open(output_path, 'w') as f:
        json.dump(api_payload, f, indent=2)
    logger.info(f"✓ Saved final API payload to: {output_path}")

    print("\n" + "=" * 70)
    print("MODEL TRAINING COMPLETE")
    print("=" * 70)
    print("\nModel Performance Summary (CV AUC-ROC):")
    for model_name, results in training_results.items():
        if 'cv_auc_mean' in results:
            print(f"  {model_name}: {results['cv_auc_mean']:.3f} "
                  f"(+/- {results['cv_auc_std']:.3f})")
    
    print("\nKey Business Insights:")
    for i, finding in enumerate(insights['key_findings'], 1):
        print(f"  {i}. {finding}")
    
    print(f"\n✓ All artifacts saved to: {predictor.model_dir.absolute()}")


if __name__ == "__main__":
    main()
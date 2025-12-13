# Hosted Link
https://epchilders1.github.io/CS451-Final/

# Netflix Engagement Decline Predictor
A machine learning system that predicts weekly Netflix viewership declines by analyzing content quality signals from user reviews and historical engagement patterns.

# Project Overview
This project combines web scraping, sentiment analysis, and time-series classification to forecast whether Netflix's weekly viewership will decline by more than 10% in the upcoming week. By integrating Netflix Top 10 viewership data with Rotten Tomatoes user sentiment, the system provides early warning signals for content performance trends.

# Key Features

- Automated Data Collection: Weekly scraping of Netflix Top 10 movies and Rotten Tomatoes reviews
- Dual Sentiment Analysis: VADER and TextBlob for comprehensive sentiment scoring
- Time-Series Modeling: Four ML models with temporal feature engineering
- Serverless Deployment: AWS Lambda + API Gateway for scalable predictions
- Interactive Dashboard: React frontend for visualizing predictions and insights

# System Architecture

The system implements a fully automated, event-driven architecture spanning data ingestion, feature engineering, model training, and deployment. GitHub Actions orchestrates the weekly pipeline, triggering web scraping tasks that collect Netflix Top 10 viewership metrics via BeautifulSoup and Rotten Tomatoes user reviews through Selenium WebDriver. Raw data undergoes a multi-stage transformation: movie-level Netflix and sentiment data are joined on title keys via left merge, then aggregated into weekly time-series features through pandas groupby operations computing engagement statistics (sum, mean, std of hours viewed) and sentiment averages (mean VADER compound, TextBlob polarity). Temporal feature engineering adds 1-4 week lags and 4-week moving averages to capture viewing momentum and content quality trends.

Four classification models (Logistic Regression, Random Forest, XGBoost, LightGBM) are trained using TimeSeriesSplit cross-validation to predict binary engagement decline events (>10% weekly viewership decrease), with scikit-learn's pipeline ensuring consistent preprocessing. Trained models, feature importance rankings, and weekly predictions are serialized to JSON format and uploaded to AWS S3 for persistent storage. An AWS Lambda function serves as the serverless API backend, triggered via API Gateway requests, which retrieves the latest prediction artifacts from S3 and returns formatted JSON responses. The React frontend consumes this REST API to render interactive visualizations of decline probabilities, model performance metrics, top predictive features, and current Top 10 movie listings, providing stakeholders with real-time content performance insights. The entire pipeline executes autonomously every Sunday, ensuring predictions reflect the most recent week's data.

# Local Setup Instructions
Prerequisites

- Python 3.8 or higher
- Git
- pip (Python package manager)
- Virtual environment tool (venv)

Step 1: Clone the Repository
```
  git clone <repository-url>
  cd <repository-name>
```
Step 2: Navigate to Backend Directory
```
  cd backend
```
Step 3: Create and Activate Virtual Environment

On macOS/Linux:
```
  python3 -m venv venv
  source venv/bin/activate
```
On Windows:
```
  python -m venv venv
  venv\Scripts\activate
```
Step 4: Install Dependencies
```
  pip install -r requirements.txt
```
# Execution
The following command will trigger the web scraping and data collection pipeline:
```
python3 main.py
```

The following command will run the evaluation procedure, evaluating all included models by the AUC-ROC classification method.
```
python3 train_models.py
```


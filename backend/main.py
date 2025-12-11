import requests
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import time
import json
import re
from typing import List, Dict, Optional
from collections import Counter
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import TimeoutException, NoSuchElementException
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

import nltk
from nltk.corpus import stopwords

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def format_date_for_storage(date_string: str) -> Optional[str]:
    """Convert date string to YYYY-MM-DD format"""
    if not date_string:
        return None

    input_format = "%b %d, %Y"
    output_format = "%Y-%m-%d"
    try:
        cleaned_date_string = date_string.strip()
        date_object = datetime.strptime(cleaned_date_string, input_format)
        formatted_date = date_object.strftime(output_format)
        return formatted_date
    except ValueError as e:
        logger.debug(f"Error parsing date '{date_string}': {e}")
        return None


class NetflixTopTenCollector:
    def __init__(self, cache_dir: str = "./cache/netflix"):
        self.top10_url = "https://www.netflix.com/tudum/top10"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = self._create_session()

    def _create_session(self) -> requests.Session:
        """Create session with retry logic"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session

    def generate_sundays(self, start_date: str = "2021-07-04", 
                        end_date: Optional[str] = None) -> List[str]:
        """Generate list of Sunday dates"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d") if end_date else datetime.today()

        # Ensure start is a Sunday
        days_until_sunday = (6 - start.weekday()) % 7
        start += timedelta(days=days_until_sunday)

        sundays = []
        current = start
        while current <= end:
            sundays.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=7)

        return sundays

    def scrape_week(self, week_date: str, use_cache: bool = True) -> List[Dict]:
        """Scrape Netflix Top 10 for a specific week"""
        cache_file = self.cache_dir / f"netflix_{week_date}.json"
        
        if use_cache and cache_file.exists():
            with open(cache_file, 'r') as f:
                logger.info(f"Loading {week_date} from cache")
                return json.load(f)

        url = f"{self.top10_url}?week={week_date}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            logger.info(f"Scraping {url} → {response.status_code}")

            soup = BeautifulSoup(response.text, "html.parser")
            rows = soup.select("table tbody tr")

            week_rows = []
            for row in rows:
                try:
                    title_btn = row.find("button")
                    title = title_btn.text.strip() if title_btn else None
                    
                    if not title:
                        continue

                    rank_el = row.find("span", class_="rank")
                    rank = rank_el.text.strip() if rank_el else None

                    tds = row.find_all("td")
                    
                    weeks_in_top10 = self._extract_text(
                        row.find("td", attrs={"data-uia": "top10-table-row-weeks"}),
                        tds[1] if len(tds) > 1 else None
                    )
                    
                    hours_viewed = self._extract_text(
                        row.find("td", attrs={"data-uia": "top10-table-row-hours"}),
                        tds[4] if len(tds) > 4 else None
                    )

                    week_rows.append({
                        "week": week_date,
                        "title": title,
                        "ranking": rank,
                        "weeks_in_top10": weeks_in_top10,
                        "hours_viewed": hours_viewed
                    })

                except Exception as e:
                    logger.error(f"Error parsing row: {e}")
                    continue

            # Cache results
            if week_rows:
                with open(cache_file, 'w') as f:
                    json.dump(week_rows, f, indent=2)

            time.sleep(0.5)
            return week_rows

        except Exception as e:
            logger.error(f"Error scraping {week_date}: {e}")
            return []

    def _extract_text(self, *elements) -> Optional[str]:
        """Extract text from first non-empty element"""
        for el in elements:
            if el is not None:
                text = el.text.strip()
                if text:
                    return text
        return None

    def collect_all_data(self, start_date: str = "2021-07-04", 
                        end_date: Optional[str] = None) -> pd.DataFrame:
        """Collect all Netflix Top 10 data"""
        weeks = self.generate_sundays(start_date, end_date)
        all_data = []

        logger.info(f"Collecting {len(weeks)} weeks of data")
        
        for i, week in enumerate(weeks, 1):
            logger.info(f"Week {i}/{len(weeks)}: {week}")
            data = self.scrape_week(week)
            all_data.extend(data)

        df = pd.DataFrame(all_data)
        
        if not df.empty:
            df = self._clean_data(df)
            logger.info(f"✓ Collected {len(df)} records")
        
        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate Netflix data"""
        df = df.copy()
        df['week'] = pd.to_datetime(df['week'])
        df['ranking'] = df['ranking'].astype(str).str.lstrip('0').astype(int)
        df['weeks_in_top10'] = pd.to_numeric(df['weeks_in_top10'], errors='coerce')
        
        df['hours_viewed'] = (df['hours_viewed']
                             .astype(str)
                             .str.replace(',', '', regex=False)
                             .replace('None', np.nan))
        df['hours_viewed'] = pd.to_numeric(df['hours_viewed'], errors='coerce')
        
        df = df.sort_values(['week', 'ranking']).reset_index(drop=True)
        return df


class RottenTomatoesSentimentCollector:

    def __init__(self, cache_dir: str = "./cache/rotten_tomatoes"):
        self.base_url = "https://rottentomatoes.com"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = self._create_session()
        self.vader = SentimentIntensityAnalyzer()
        self.driver = None

    def _create_session(self) -> requests.Session:
        """Create session with retry logic"""
        session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        return session

    def _init_driver(self):
        """Initialize Selenium WebDriver with options"""
        if self.driver is None:
            chrome_options = Options()
            chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36')
            
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Selenium WebDriver initialized")

    def _close_driver(self):
        """Close Selenium WebDriver"""
        if self.driver:
            self.driver.quit()
            self.driver = None
            logger.info("Selenium WebDriver closed")

    def search_movie(self, title: str) -> Optional[str]:
        """Search for movie on Rotten Tomatoes"""
        search_url = f"{self.base_url}/search?search={title.replace(' ', '+')}/"
        
        try:
            response = self.session.get(search_url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')

            results_list = soup.find("ul", attrs={"slot": "list"})
            
            if results_list:
                top_result_li = soup.find("search-page-media-row")
            else:
                top_result_li = None

            if top_result_li:
                link = top_result_li.find("a")
                if link and 'href' in link.attrs:
                    movie_link = link['href']
                    logger.debug(f"✓ Found '{title}' on Rotten Tomatoes")
                    return movie_link
            
            logger.warning(f"⚠️ Could not find '{title}' on Rotten Tomatoes")
            return None
                
        except Exception as e:
            logger.error(f"❌ Error searching for '{title}': {e}")
            return None

    def scrape_reviews(self, movie_url: str, debug: bool = False) -> List[Dict]:
        """Scrape reviews using Selenium for dynamic content"""
        movie_slug = movie_url.split('/')[-1] or movie_url.split('/')[-2]
        base_review_url = f"{movie_url}/reviews?type=user"
        cache_file = self.cache_dir / f"reviews_{movie_slug}.json"
        logger.info(f"Scraping: {base_review_url}")
        
        # Check cache
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                logger.info(f"Loading reviews for {movie_slug} from cache")
                return json.load(f)

        all_reviews = []
        
        try:
            self._init_driver()
            
            self.driver.get(base_review_url)
            
            wait = WebDriverWait(self.driver, 15)
            try:
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "review-card")))
                logger.info(f"Review cards loaded for {movie_slug}")
            except TimeoutException:
                logger.warning(f"No review cards found for {movie_slug} after timeout")
                return []
            
            time.sleep(0.1)
            
            if debug:
                screenshot_path = self.cache_dir / f"screenshot_{movie_slug}.png"
                html_path = self.cache_dir / f"page_source_{movie_slug}.html"
                self.driver.save_screenshot(str(screenshot_path))
                with open(html_path, 'w', encoding='utf-8') as f:
                    f.write(self.driver.page_source)
                logger.info(f"Debug files saved: {screenshot_path} and {html_path}")
            
            reviews = self.driver.find_elements(By.TAG_NAME, 'review-card')
            logger.info(f"Found {len(reviews)} review cards")
            
            if reviews and len(reviews) > 0:
                try:
                    first_review_html = reviews[0].get_attribute('outerHTML')
                    logger.debug(f"First review HTML sample (first 500 chars): {first_review_html[:500]}")
                except Exception as e:
                    logger.debug(f"Could not get HTML sample: {e}")
            
            for idx, review in enumerate(reviews):
                review_text = ""
                date = None
                rating = None
                name = None
                
                try:
                    try:
                        rating_span = review.find_element(By.CSS_SELECTOR, 'span[slot="rating"]')
                        rating_text = rating_span.text.strip()
                        match = re.search(r'([\d.]+)/\d+', rating_text)
                        if match:
                            rating = float(match.group(1))
                    except (NoSuchElementException, ValueError):
                        rating = None

                    try:
                        drawer = review.find_element(By.CSS_SELECTOR, 'drawer-more[slot="review"]')
                        
                        try:
                            shadow_root = self.driver.execute_script('return arguments[0].shadowRoot', drawer)
                            if shadow_root:
                                review_text = self.driver.execute_script('''
                                    const drawer = arguments[0];
                                    const shadowRoot = drawer.shadowRoot;
                                    if (shadowRoot) {
                                        const content = shadowRoot.querySelector('span[slot="content"]') || 
                                                       shadowRoot.querySelector('.review-text') ||
                                                       shadowRoot.querySelector('p');
                                        return content ? content.textContent.trim() : '';
                                    }
                                    return '';
                                ''', drawer)
                                
                                if idx < 3:
                                    logger.debug(f"Review {idx}: Extracted from shadow DOM, length: {len(review_text)}")
                        except Exception as e:
                            if idx < 3:
                                logger.debug(f"Review {idx}: No shadow root or error accessing it: {e}")
                        
                        if not review_text:
                            try:
                                text_el = drawer.find_element(By.CSS_SELECTOR, 'span[slot="content"]')
                                review_text = text_el.text.strip()
                                
                                if not review_text:
                                    review_text = self.driver.execute_script('''
                                        const span = arguments[0].querySelector('span[slot="content"]');
                                        return span ? span.textContent.trim() : '';
                                    ''', drawer)
                                
                                if idx < 3:
                                    logger.debug(f"Review {idx}: Extracted normally, length: {len(review_text)}")
                                    
                            except NoSuchElementException:
                                if idx < 3:
                                    logger.debug(f"Review {idx}: Drawer found but no content span")
                            
                    except NoSuchElementException:
                        if idx < 3:
                            logger.debug(f"Review {idx}: No drawer-more element found")
                        review_text = ""
                    
                    try:
                        date_el = review.find_element(By.CSS_SELECTOR, 'span[slot="timestamp"]')
                        date_text = date_el.text.strip()
                        date_obj = datetime.strptime(date_text, "%m/%d/%Y")
                        date = date_obj.strftime("%Y-%m-%d")
                    except (NoSuchElementException, ValueError):
                        date = None
                    
                    try:
                        user_el = review.find_element(By.CSS_SELECTOR, 'rt-link[slot="name"]')
                        name = user_el.text.strip()
                    except NoSuchElementException:
                        name = "Anonymous"

                    if review_text:
                        all_reviews.append({
                            'user': name,
                            'rating': rating,
                            'text': review_text,
                            'date': date
                        })
                        # print("Name: ", name, "Rating:", rating, "Rating Text:", review_text, "Date:", date)
                
                except Exception as e:
                    logger.debug(f"Error parsing individual review: {e}")
                    continue
            
            logger.info(f"Scraped {movie_slug}: {len(all_reviews)} reviews")
        
        except Exception as e:
            logger.error(f"Error scraping {movie_slug}: {e}")
        
        if all_reviews:
            with open(cache_file, 'w') as f:
                json.dump(all_reviews, f, indent=2)
        
        return all_reviews


    def analyze_sentiment(self, reviews: List[Dict]) -> Dict:
        """Analyze sentiment from reviews"""
        if not reviews:
            return self._empty_sentiment()

        ratings = []
        for r in reviews:
            if r['rating'] is not None:
                try:
                    ratings.append(float(r['rating']))
                except (ValueError, TypeError):
                    continue
        
        texts = [r['text'] for r in reviews if r['text']]
        
        rating_metrics = {
            'avg_rating': float(np.mean(ratings)) if ratings else None,
            'median_rating': float(np.median(ratings)) if ratings else None,
            'rating_std': float(np.std(ratings)) if ratings else None,
            'num_ratings': len(ratings),
            'rating_distribution': json.dumps(dict(Counter(ratings))) if ratings else '{}'
        }
        
        if texts:
            vader_scores = [self.vader.polarity_scores(text) for text in texts]
            textblob_scores = [TextBlob(text).sentiment.polarity for text in texts]
            
            text_metrics = {
                'vader_compound_mean': float(np.mean([s['compound'] for s in vader_scores])),
                'vader_compound_std': float(np.std([s['compound'] for s in vader_scores])),
                'vader_positive_mean': float(np.mean([s['pos'] for s in vader_scores])),
                'vader_negative_mean': float(np.mean([s['neg'] for s in vader_scores])),
                'vader_neutral_mean': float(np.mean([s['neu'] for s in vader_scores])),
                'textblob_polarity_mean': float(np.mean(textblob_scores)),
                'textblob_polarity_std': float(np.std(textblob_scores)),
                'num_reviews_with_text': len(texts)
            }
        else:
            text_metrics = {
                'vader_compound_mean': None,
                'vader_compound_std': None,
                'vader_positive_mean': None,
                'vader_negative_mean': None,
                'vader_neutral_mean': None,
                'textblob_polarity_mean': None,
                'textblob_polarity_std': None,
                'num_reviews_with_text': 0
            }
        
        return {**rating_metrics, **text_metrics}

    def _empty_sentiment(self) -> Dict:
        """Return empty sentiment metrics"""
        return {
            'avg_rating': None,
            'median_rating': None,
            'rating_std': None,
            'num_ratings': 0,
            'rating_distribution': '{}',
            'vader_compound_mean': None,
            'vader_compound_std': None,
            'vader_positive_mean': None,
            'vader_negative_mean': None,
            'vader_neutral_mean': None,
            'textblob_polarity_mean': None,
            'textblob_polarity_std': None,
            'num_reviews_with_text': 0
        }

    def __del__(self):
        """Cleanup: close driver when object is destroyed"""
        self._close_driver()

class NetflixEngagementPipeline:
    """Complete pipeline for Netflix engagement prediction"""
    
    def __init__(self, output_dir: str = "./data"):
        self.netflix_collector = NetflixTopTenCollector()
        self.rotten_tomatoes_collector = RottenTomatoesSentimentCollector()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def collect_netflix_data(self, start_date: str = "2021-07-04", 
                            end_date: Optional[str] = None) -> pd.DataFrame:
        """Collect Netflix Top 10 data"""
        logger.info("=== Collecting Netflix Top 10 Data ===")
        df = self.netflix_collector.collect_all_data(start_date, end_date)
        
        # Save raw data
        output_path = self.output_dir / "netflix_top10_raw.csv"
        df.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to {output_path}")
        
        return df

    def collect_sentiment_data(self, netflix_df: pd.DataFrame) -> pd.DataFrame:
        """Collect sentiment data for all unique titles"""
        logger.info("=== Collecting Rotten Tomatoes Sentiment Data ===")
        
        unique_movies = netflix_df['title'].unique()
        logger.info(f"Found {len(unique_movies)} unique movies")
        
        sentiment_data = []
        
        for i, title in enumerate(unique_movies, 1):
            logger.info(f"Processing {i}/{len(unique_movies)}: {title}")
            
            url = self.rotten_tomatoes_collector.search_movie(title)
            if url:
                movie_slug = url.split('/')[-1] or url.split('/')[-2]
                if movie_slug:
                    reviews = self.rotten_tomatoes_collector.scrape_reviews(url)
                    
                    sentiment = self.rotten_tomatoes_collector.analyze_sentiment(reviews)
                    sentiment['title'] = title
                    sentiment['rotten_tomatoes_slug'] = movie_slug
                    sentiment_data.append(sentiment)
                else:
                    sentiment = self.rotten_tomatoes_collector._empty_sentiment()
                    sentiment['title'] = title
                    sentiment['rotten_tomatoes_slug'] = None
                    sentiment_data.append(sentiment)
                
            time.sleep(1)
        
        df_sentiment = pd.DataFrame(sentiment_data)
        
        output_path = self.output_dir / "rotten_tomatoes_sentiment.csv"
        df_sentiment.to_csv(output_path, index=False)
        logger.info(f"✓ Saved to {output_path}")
        
        return df_sentiment

    def create_unified_dataset(self, netflix_df: pd.DataFrame, 
                              sentiment_df: pd.DataFrame) -> pd.DataFrame:
        """Merge Netflix and sentiment data"""
        logger.info("=== Creating Unified Dataset ===")
        
        merged = netflix_df.merge(
            sentiment_df,
            on='title',
            how='left'
        )
        
        output_path = self.output_dir / "netflix_rotten_tomatoes_unified.csv"
        merged.to_csv(output_path, index=False)
        logger.info(f"✓ Saved unified dataset to {output_path}")
        logger.info(f"  Shape: {merged.shape}")
        logger.info(f"  Coverage: {merged['rotten_tomatoes_slug'].notna().sum()}/{len(merged)} "
                   f"({merged['rotten_tomatoes_slug'].notna().mean():.1%}) titles have RT data")
        
        return merged

    def create_weekly_features(self, unified_df: pd.DataFrame) -> pd.DataFrame:
        """Create weekly aggregated features for modeling"""
        logger.info("=== Creating Weekly Feature Set ===")
        
        weekly_features = unified_df.groupby('week').agg({
            'hours_viewed': ['sum', 'mean', 'std', 'min', 'max'],
            'weeks_in_top10': 'mean',
            'ranking': 'nunique',
            
            'avg_rating': 'mean',
            'rating_std': 'mean',
            'num_ratings': 'sum',
            'vader_compound_mean': 'mean',
            'vader_positive_mean': 'mean',
            'vader_negative_mean': 'mean',
            'textblob_polarity_mean': 'mean',
            'num_reviews_with_text': 'sum'
        }).reset_index()
        
        weekly_features.columns = ['_'.join(col).strip('_') 
                                   for col in weekly_features.columns.values]
        weekly_features.rename(columns={'week_': 'week'}, inplace=True)
        
        weekly_features['total_hours_next_week'] = weekly_features['hours_viewed_sum'].shift(-1)
        weekly_features['hours_pct_change'] = (
            (weekly_features['total_hours_next_week'] - weekly_features['hours_viewed_sum']) 
            / weekly_features['hours_viewed_sum']
        )
        
        # Create binary churn target (10% decline threshold)
        weekly_features['engagement_decline'] = (
            weekly_features['hours_pct_change'] < -0.10
        ).astype(int)
        
        weekly_features['week'] = pd.to_datetime(weekly_features['week'])
        weekly_features['year'] = weekly_features['week'].dt.year
        weekly_features['month'] = weekly_features['week'].dt.month
        weekly_features['week_of_year'] = weekly_features['week'].dt.isocalendar().week
        
        for lag in [1, 2, 3, 4]:
            weekly_features[f'hours_lag_{lag}'] = (
                weekly_features['hours_viewed_sum'].shift(lag)
            )
            sentiment_filled = weekly_features['vader_compound_mean_mean'].fillna(0)
            weekly_features[f'sentiment_lag_{lag}'] = sentiment_filled.shift(lag)
        
        weekly_features['hours_ma_4week'] = (
            weekly_features['hours_viewed_sum'].rolling(4, min_periods=1).mean()
        )
        weekly_features['sentiment_ma_4week'] = (
            weekly_features['vader_compound_mean_mean']
            .fillna(0)
            .rolling(4, min_periods=1).mean()
        )
        
        output_path = self.output_dir / "weekly_features.csv"
        weekly_features.to_csv(output_path, index=False)
        logger.info(f"✓ Saved weekly features to {output_path}")
        logger.info(f"  Shape: {weekly_features.shape}")
        logger.info(f"  Target distribution (engagement_decline): "
                   f"{weekly_features['engagement_decline'].value_counts().to_dict()}")
        
        missing_sentiment = weekly_features['vader_compound_mean_mean'].isna().sum()
        logger.info(f"  Missing sentiment data: {missing_sentiment}/{len(weekly_features)} weeks")
        
        return weekly_features

    def run_full_pipeline(self, start_date: str = "2021-07-04", 
                         end_date: Optional[str] = None):
        """Run complete data collection and feature engineering pipeline"""
        logger.info("=" * 70)
        logger.info("STARTING NETFLIX ENGAGEMENT PREDICTION PIPELINE")
        logger.info("=" * 70)
        
        # Step 1: Collect Netflix data
        netflix_df = self.collect_netflix_data(start_date, end_date)
        
        # Step 2: Collect sentiment data
        sentiment_df = self.collect_sentiment_data(netflix_df)
        
        # Step 3: Create unified dataset
        unified_df = self.create_unified_dataset(netflix_df, sentiment_df)
        
        # Step 4: Create weekly features for modeling
        weekly_features = self.create_weekly_features(unified_df)
        
        logger.info("=" * 70)
        logger.info("✓ PIPELINE COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Output files saved to: {self.output_dir.absolute()}")
        logger.info("  1. netflix_top10_raw.csv - Raw Netflix Top 10 data")
        logger.info("  2. rotten_tomatoes_sentiment.csv - Movie sentiment scores")
        logger.info("  3. netflix_rotten_tomatoes_unified.csv - Merged dataset")
        logger.info("  4. weekly_features.csv - Ready for modeling")
        
        return {
            'netflix': netflix_df,
            'sentiment': sentiment_df,
            'unified': unified_df,
            'features': weekly_features
        }


def main():
    """Main execution"""
    pipeline = NetflixEngagementPipeline(output_dir="./data")
    
    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # results = pipeline.run_full_pipeline(
    #     start_date="2024-01-01",
    #     end_date="2024-11-30"
    # )
    results = pipeline.run_full_pipeline(
        start_date="2021-07-04",
        end_date=current_date
    )
    
    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)
    print(f"\nNetflix Top 10 Records: {len(results['netflix'])}")
    print(f"Unique Movies: {results['netflix']['title'].nunique()}")
    print(f"Weeks Covered: {results['netflix']['week'].nunique()}")
    print(f"\nMovies with Rotten Tomatoes Data: "
          f"{results['sentiment']['rotten_tomatoes_slug'].notna().sum()}")
    print(f"Total Reviews Collected: "
          f"{results['sentiment']['num_reviews_with_text'].sum():.0f}")
    print(f"\nWeekly Feature Records: {len(results['features'])}")
    print(f"Engagement Decline Events: {results['features']['engagement_decline'].sum()}")
    print(f"Average weekly hours: "
          f"{results['features']['hours_viewed_sum'].mean():,.0f}")


if __name__ == "__main__":
    main()
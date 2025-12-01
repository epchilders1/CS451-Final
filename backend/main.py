import requests
from bs4 import BeautifulSoup
from datetime import datetime
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from flask import Flask, render_template, jsonify
import logging

kaggle_dataset = "ashishkumarak/netflix-reviews-playstore-daily-updated"
top10_url="https://www.netflix.com/tudum/top10"

date = datetime.now().strftime("%Y-%m-%d")


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class NetflixDataCollector:
    
    def __init__(self):
        
        self.kaggle_dataset = kaggle_dataset
        self.top10_url = top10_url
        self.api = None
        self._init_kaggle_api()

    def _init_kaggle_api(self):
        try:
            self.api = KaggleApi()
            self.api.authenticate()
            logger.info("✓ Kaggle API authenticated successfully")
        except Exception as e:
            logger.error(f"✗ Failed to authenticate Kaggle API: {e}")
            logger.error("Make sure kaggle.json is in ~/.kaggle/ directory")
            raise
    def scrape_top10(self):
        try:
            response = requests.get(top10_url)
            html_content = response.text

            print("Received response:", response.status_code)

            soup = BeautifulSoup(html_content, 'html.parser')
            title_list = soup.find_all('td', class_="title", attrs={'data-uia': 'top10-table-row-title'})
            for title in title_list:
                movie_title = title.find("button")
                print(movie_title.text.strip())
            print(date)

        except Exception as e:
            logger.error(f"Error during web scraping: {e}")
            return []

    def weekly_update(self):
        self.scrape_top10()
    



def main():
    print("Hello, World!")
    pipeline = NetflixDataCollector()
    pipeline.weekly_update()


if __name__ == "__main__":
    main()
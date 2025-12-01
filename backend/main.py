
import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.title.string
    else:
        return None

def main():
    print("Hello, World!")


if __name__ == "__main__":
    main()
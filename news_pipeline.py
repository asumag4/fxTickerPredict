import requests
from bs4 import BeautifulSoup

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from icecream import ic

from datetime import date

import re
import pandas as pd
import numpy as np

import time

# Import the date obj

ticker_fxstreet_url_representation = {
    # key => yahoo ticker 
    # value => fxstreet ticker 
    "EURUSD=X" : "EURUSD",
    "JPY=X" : "USDJPY",
    "GBPUSD=X" : "GBPUSD"
}

def scrape_links(page, ticker):
    """
    page => list of page numbers you want to scrape, from 0 to 59 
    ticker => yahoo finance ticker symbol for forex pair

    scrape_links -> returns a list of tuples for (<article title>,<article link>)

    The output of this function is meant to be used for article scraping with 
    """

    ticker = ticker_fxstreet_url_representation[ticker]

    # Initialize WebDriver
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")  # Run in headless mode (remove if debugging)
    driver = webdriver.Chrome(options=options)

    try:
        # Open FXStreet News page
        driver.get(f'https://www.fxstreet.com/news/latest/asset?q=&hPP=17&idx=FxsIndexPro&p={page}&dFR[Category][0]=News&dFR[Tags][0]={ticker}')

        # Wait for the page to fully load
        WebDriverWait(driver, 10).until(
            EC.presence_of_element_located((By.TAG_NAME, 'body'))
        )

        # Handle pop-up if it appears
        try:
            close_button = WebDriverWait(driver, 5).until(
                EC.element_to_be_clickable((By.XPATH, "//*[contains(text(), 'CONTINUE TO SITE')]"))
            )
            close_button.click()
            print("Pop-up closed.")
        except:
            print("No pop-up found.")

        # Wait for news section to load
        WebDriverWait(driver, 10).until(
            EC.presence_of_all_elements_located((By.CLASS_NAME, 'fxs_headline_tiny'))
        )

        # Scroll down to load more articles (if needed)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        # Extract news headlines and URLs
        headlines = driver.find_elements(By.CLASS_NAME, 'fxs_headline_tiny')

        news_data = []
        for headline in headlines:
            try:
                link = headline.find_element(By.TAG_NAME, "a")
                title = link.text.strip()
                url = link.get_attribute("href")
                news_data.append((title, url))
            except:
                print("Skipping an element due to missing link.")

        print("Articles were scraped successfully")

    finally:
        driver.quit()  # Close the browser
    
    return news_data

def scrape_article(link):
    # get html data from a website
    r = requests.get(link)
    soup = BeautifulSoup(r.text, 'html.parser')
    from dateutil import parser

    # Extract the date
    date = soup.find("time").text
    date # We'll store it the way it is, the raw form of the date object
    
    # Extract the article only 
    articles = soup.find_all('article', class_=("fxs_article") )
    articles

    paragraph_tags = r"<p>(.*?)</p>"
    anchor_tags = r"<a>(.*?)</a>"

    paragraphs_extracted = []
    for article in articles: 
        paragraphs = re.findall(paragraph_tags, str(article), re.DOTALL)
        for paragraph in paragraphs:
            if paragraphs is not None: 
                paragraphs_extracted.append(paragraph)
                
    paragraphs_extracted

    # Extract irrelevant data, including disclaimer data and anchor data. 

    anchor_tags = r"<a.*?</a>"
    HTML_tags = r"<.*?>"

    for idx, paragraph in enumerate(paragraphs_extracted):

        # Remove anchors
        anchor = re.search(anchor_tags, paragraph)
        if anchor:
            paragraph = paragraph.strip(anchor.string)
            paragraphs_extracted[idx] = paragraph
        
        # Remove newline characters
        newlines = re.findall(r"\n", paragraph)
        for newline in newlines:
            paragraph = paragraph.strip(newline)
            paragraphs_extracted[idx] = paragraph

        # Remove any leftover HTML tags
        tags = re.findall(HTML_tags, paragraph)
        for tag in tags:
            print("Tag found")
            paragraph = paragraph.strip(tag)
            paragraphs_extracted[idx] = paragraph
    
    # Amalgamate the extracted text
    HTML_content = "" # init empty holder
    for i in paragraphs_extracted:
        HTML_content += i
    
    return date, HTML_content


# We need a function to combine both `scrape_links()` && `scrape_article()`
def scrape_ticker_articles(page_numbers, ticker):
    """
    page_numbers => list of page numbers you want to scrape
    ticker => string representation of ticker symbol
    """

    df = pd.DataFrame(columns=["title","link","date","text"])
    for page in page_numbers:
        links = scrape_links(page, ticker)
        for title, link in links:
            date, text = scrape_article(link)

            # Store it all in a dict to be appended into the dataframe; use referencing in the dict to save compute space
            to_append = {
                "title" : title,
                "link" : link,
                "date" : date,
                "text" : text,
            }

            # Append into `df`
            df.loc[len(df)] = to_append

            # Implicit wait time to overcome firewall 
            time.sleep(1) # Wait 1 second before scraping the next article

        time.sleep(1.5) # Wait 1.5 seconds before accessing the next article links page
    
    return df



"""
Make this python file retrieve data on a daily basis and append it to a RDB 
to store scraped data. 

The article-data that should be stored on the RDB should be;
title, date, text, embeddings of text
"""

"""
# Curating the main database with as much historical data as accessible 

pages = np.arange(0,59) # From 0 to 58

database = pd.DataFrame(columns=["title","link","date","text"])
for page in pages:
    page_lst = [page]
    ticker = "GBPUSD=X"
    df = scrape_ticker_articles(page_lst,ticker)
    database = pd.concat([database, df])

database.to_parquet(f"{ticker}_fxstreet_database")

"""

""" Main """
# 1. Set the ticker and number of pages 

# Create a ticker for all currency pairs
EURUSD = "EURUSD=X" # EUR/USD
USDJPY = "JPY=X"
GBPUSD = "GBPUSD=X"

def daily_update_table(ticker):
    incoming_data = pd.DataFrame(columns=["title","link","date","text"])
    for page in [0]: # Get the first page
        page_lst = [page]
        df = scrape_ticker_articles(page_lst,ticker)
        incoming_data = pd.concat([incoming_data, df])
    return incoming_data

# 3. Push the data to database
# Open up the database 
def write_to_raw_database(incoming_data, ticker):
    print("Intializing... news scraper")

    # Define file folder path + file name
    file_path = r'data/raw_scrapped_data'
    file_name = f"/{ticker}_fxstreet_database" # Will throw an "invalid escape sequence" error and thats fine, it's not meant to be one

    # Save to a different path just in case
    file_path_dummy = r'data/test_pipeline_data'
    file_name_dummy = f"/{ticker}_fxstreet_database"

    database = pd.read_parquet(file_path + file_name)
    database = pd.concat([incoming_data,database])

    # We also don't want to save duplicates due to modelling steps + storage-space reasons
    database = database.drop_duplicates() 

    # Write out to main database
    database.to_parquet(file_path_dummy + file_name_dummy)

incoming_data = daily_update_table(EURUSD)
write_to_raw_database(incoming_data, EURUSD)

# After getting all historical data, you want to keep the database updated on a daily basis
# 2. Set to only get the first page, maybe every hour of when the market is open? 

import requests
import json
import re
import pandas as pd
from bs4 import BeautifulSoup
import time
import logging
import sys, os

this_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(this_path)
os.chdir(this_path)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='scrape_debug.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

def scrape_data(city_id, city=None):
    logging.info(f"Scraping data for city: {city} (ID: {city_id})")
    url = "https://b2c-catalog-gateway.c24.tech/listing/v1/buy-used-car"

    headers = {
        "accept": "application/json, text/plain, */*",
        "accept-encoding": "gzip, deflate, br, zstd",
        "accept-language": "en-GB,en;q=0.9,en-US;q=0.8",
        "content-type": "application/json",
        "origin": "https://www.cars24.com",
        "referer": "https://www.cars24.com/",
        "sec-ch-ua": "\"Not)A;Brand\";v=\"99\", \"Microsoft Edge\";v=\"127\", \"Chromium\";v=\"127\"",
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": "\"Linux\"",
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "cross-site",
        "source": "WebApp",
        "user-agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0",
        "x-user-city-id": str(city_id),
        "x_experiment_id": "513e95f9-b8f3-4e60-9361-f8c8401c687c"
    }

    body = {
        "searchFilter": [],
        "cityId": str(city_id),
        "sort": "bestmatch",
        "size": 10000
    }

    with requests.Session() as session:
        session.headers.update(headers)
        try:
            logging.debug(f"Sending POST request to {url}")
            response = session.post(url, data=json.dumps(body), timeout=10)

            if response.status_code == 200:
                logging.info(f"Successfully retrieved data for {city}")
                car_data = response.json()
                logging.debug(f"Retrieved {len(car_data.get('content', []))} cars for {city}")
            else:
                logging.error(f"Failed with status code: {response.status_code}")
                logging.error(f"Response Text: {response.text}")
                return None

        except requests.exceptions.RequestException as e:
            logging.error(f"An error occurred while scraping data for {city}: {e}")
            return None

    return car_data['content']

def get_html_content(url, max_retries=10, delay=1):
    logging.debug(f"Fetching HTML content from: {url}")
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36 Edg/127.0.0.0"
    }
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)

            if response.status_code == 200:
                logging.debug(f"Successfully retrieved HTML content from {url}")
                return response.text
            else:
                logging.warning(f"Attempt {attempt + 1} failed: status code {response.status_code} for {url}")
        except requests.exceptions.RequestException as e:
            logging.warning(f"Attempt {attempt + 1} failed with error: {e} for {url}")
        
        if attempt < max_retries - 1:
            time.sleep(delay)
    
    logging.error(f"Failed to retrieve page from {url} after {max_retries} attempts")
    return None

def extract_engine_power(html_content):
    logging.debug("Extracting engine and power information")
    engine = ''
    power = ''
    
    soup = BeautifulSoup(html_content, 'html.parser')
    script_tags = soup.find_all('script')
    for script in script_tags:
        if 'Displacementcc' in script.text:
            displacement_match = re.search(r'"Displacementcc".*?"value":"(\d+)"', script.text)
            max_power_match = re.search(r'"MaxPowerbhp".*?"value":"(\d+)"', script.text)
            
            if displacement_match:
                engine = displacement_match.group(1)
            if max_power_match:
                power = max_power_match.group(1)
    
    logging.debug(f"Extracted engine: {engine}, power: {power}")
    return engine, power

def process_car_html(car, city, city_full):
    id = car.get("appointmentId", None)
    name = car.get("carName", None)
    car_lower = "-".join(name.lower().split()) if name else None
    car_city_lower = "-".join(city_full.lower().split()) if city == 'Delhi' else "-".join(city.lower().split())
    year = car.get("year", None)
    link = f"https://www.cars24.com/buy-used-{car_lower}-{year}-cars-{car_city_lower}-{id}" if car_lower and car_city_lower else None

    if link:
        html_content = get_html_content(link)
        if html_content:
            engine, power = extract_engine_power(html_content)
        else:
            logging.warning(f"Failed to retrieve HTML content for {link}")
            engine, power = None, None
    else:
        logging.warning(f"No link generated for car: {name}")
        engine, power = None, None

    return engine, power, link

def extract_data(car_data, city):
    logging.info(f"Extracting data for {city}")
    if city == 'Delhi':
        city_full = 'New Delhi'
    else:
        city_full = city

    extracted_data = []
    failed_urls = []
    
    for car in car_data:
        engine, power, link = process_car_html(car, city, city_full)
        car_info = {
            'Name': car.get("carName", None),
            'Automaker': car.get("make", None),
            'Location': city,
            'Year': car.get("year", None),
            'Kilometers_Driven': car.get("odometer", None),
            'Fuel_Type': car.get("fuelType", None),
            'Transmission': car.get("transmissionType", None),
            'Owner_Type': car.get("ownership", None),
            'Mileage(kmpl)': 0,
            'Engine (CC)': engine,
            'Power (bhp)': power,
            'Seats': car.get("seats", None),
            'Price': car.get("listingPrice", None)
        }
        extracted_data.append(car_info)
        logging.info(f"Appended car data: {car_info}")
        
        if engine is None and power is None and link:
            failed_urls.append(link)

    logging.info(f"Extracted data for {len(extracted_data)} cars in {city}")
    logging.info(f"Failed URLs: {failed_urls}")
    return extracted_data, failed_urls

def retry_failed_urls(failed_urls):
    logging.info(f"Retrying {len(failed_urls)} failed URLs")
    retried_data = []
    
    for url in failed_urls:
        html_content = get_html_content(url, max_retries=3, delay=2)
        if html_content:
            engine, power = extract_engine_power(html_content)
            car_info = {
                'Engine (CC)': engine,
                'Power (bhp)': power
            }
            retried_data.append(car_info)
            logging.info(f"Successfully retried URL: {url}")
        else:
            logging.warning(f"Failed to retry URL: {url}")
    
    return retried_data

def final_touch(extracted_data):
    logging.info("Performing final data cleanup and conversion")
    
    car_data = extracted_data[0] if extracted_data and isinstance(extracted_data[0], list) else []
    
    if not car_data:
        logging.warning("No car data found in extracted_data")
        return pd.DataFrame()

    df = pd.DataFrame(car_data)
    initial_count = len(df)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    logging.debug(f"Removed {initial_count - len(df)} rows with NaN values or duplicates")

    for column in ['Year', 'Kilometers_Driven', 'Mileage(kmpl)', 'Seats', 'Engine (CC)', 'Power (bhp)', 'Price']:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    
    df.dropna(inplace=True)
    logging.debug(f"Final dataframe shape: {df.shape}")

    return df

def save_data(df, city):
    filename = f"data/raw/crawled/cars24_{city}.csv"
    logging.info(f"Saving data to {filename}")
    df.to_csv(filename, index=False)
    logging.info(f"Data successfully saved to {filename}")

cities = ['Delhi', 'Bangalore', 'Mumbai', 'Hyderabad', 'Ahmedabad', 'Chennai', 'Pune', 'Jaipur', 'Kolkata', 'Kochi', 'Coimbatore']  
city_ids = [2, 4709, 2378, 3686, 1692, 5732, 2423, 2130, 777, 6356, 6105]

for city, city_id in zip(cities, city_ids):
    logging.info(f"Starting data collection for {city}")
    car_data = scrape_data(city_id, city)
    if car_data:
        extracted_data, failed_urls = extract_data(car_data, city)
        logging.debug(f"Sample of extracted_data (first 2 items): {json.dumps(extracted_data[:2], indent=2)}")
        if extracted_data and isinstance(extracted_data[0], dict) and extracted_data[0]:
            logging.debug(f"Data types of first car item: {json.dumps({k: str(type(v)) for k, v in extracted_data[0].items()}, indent=2)}")
        else:
            logging.warning("extracted_data structure is not as expected")        
        final_data = final_touch([extracted_data])
        save_data(final_data, city)
        logging.info(f"Data collection and saving completed for {city}")
    else:
        logging.error(f"Failed to collect data for {city}")

logging.info("All data collection and saving processes completed!")
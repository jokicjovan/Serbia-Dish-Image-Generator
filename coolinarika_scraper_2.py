import argparse
import os
import shutil
import threading
import time
import uuid
import requests
import json
import re
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

chrome_options = Options()
chrome_options.add_argument("--headless")
dish_counter=0
dishes = {
    "dishes": []
}
dishes_lock = threading.Lock()


class Dish:
    def __init__(self, name=None, ingredients=None, preparation=None, image_path=None):
        self.name = name
        self.ingredients = ingredients
        self.preparation = preparation
        self.image_path = image_path

    def __str__(self):
        return f"{self.name}\n{self.ingredients}\n{self.preparation}\n{self.image_path}"

    def __repr__(self):
        return f"{self.name}\n{self.ingredients}\n{self.preparation}\n{self.image_path}"

    @property
    def name(self):
        return self.__name

    @name.setter
    def name(self, name):
        self.__name = name

    @property
    def ingredients(self):
        return self.__ingredients

    @ingredients.setter
    def ingredients(self, ingredients):
        self.__ingredients = ingredients

    @property
    def preparation(self):
        return self.__preparation

    @preparation.setter
    def preparation(self, preparation):
        self.__preparation = preparation

    @property
    def image_path(self):
        return self.__image_path

    @image_path.setter
    def image_path(self, image_path):
        self.__image_path = image_path

    def make_json(self):
        return {
            "name": self.name,
            "ingredients": self.ingredients,
            "preparation": self.preparation,
            "image_path": self.image_path
        }


def fetch_recipe_data(url):
    try:
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, 'html.parser')
        script_tag = soup.find('script', type='application/ld+json')

        # Extract the JSON content
        json_data = json.loads(script_tag.string)

        name = re.sub(r'\xa0', '', json_data['name'].replace("\"", "").replace("'", "").replace('\xad', ''))
        image_url = json_data['image']
        ingredients = [re.sub(r'\xa0', '', ingredient.replace("\"", "").replace("'", "").replace('\xad', '')) for ingredient in json_data['recipeIngredient']]
        preparation_steps = [re.sub(r'\xa0', '', step['text'].replace("\"", "").replace("'", "").replace('\xad', '')) for step in json_data['recipeInstructions']]

        # Download the image
        image_name = str(uuid.uuid4()) + ".jpg"
        image_path = os.path.join('data_coolinarika/images/', image_name)
        download_and_save_image(image_url, image_path)
        dish= Dish(name=name, ingredients=ingredients, preparation=preparation_steps, image_path=image_path)
        with dishes_lock:
            dishes["dishes"].append(dish.make_json())
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return None


def download_and_save_image(url, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    response = requests.get(url)
    with open(path, 'wb') as f:
        f.write(response.content)

from concurrent.futures import ThreadPoolExecutor, as_completed

def scrape_recipes_parallel(recipe_urls):
    with ThreadPoolExecutor() as executor:
        executor.map(fetch_recipe_data, recipe_urls)




def scrape_listing_page(base_url, listing_url):
    # Init new driver
    driver = webdriver.Chrome(options=chrome_options)

    try:
        # Go to the listing page
        driver.get(listing_url)

        # Scroll down until we reach the bottom of the page
        last_height = driver.execute_script("return document.body.scrollHeight")

        # Track the initial number of recipe cards
        initial_card_count = len(driver.find_elements(By.CSS_SELECTOR, 'a.cardInner'))

        while True:
            time.sleep(2)
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

            # Wait for new recipe cards to load
            try:
                WebDriverWait(driver, 10).until(
                    lambda d: len(d.find_elements(By.CSS_SELECTOR, 'a.cardInner')) > initial_card_count
                )
            except TimeoutException:
                # Bottom is reached
                break

            # Track the new number of recipe cards
            new_card_count = len(driver.find_elements(By.CSS_SELECTOR, 'a.cardInner'))


            # Update the count for the next iteration
            initial_card_count = new_card_count

            # Get new height to check if we've reached the bottom
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:  # Stop if we have reached the bottom
                break
            last_height = new_height

        # Get the page source
        soup = BeautifulSoup(driver.page_source, 'html.parser')

        # Extract the recipe links
        recipe_links = soup.find_all('a', {'class': 'cardInner'})
        recipe_urls = [link['href'] for link in recipe_links]
        full_urls = [base_url + url for url in recipe_urls]

        with open("data_coolinarika/links.txt", "w") as file:
            # Join each string with a newline or write each line individually
            file.write("\n".join(full_urls))
        return full_urls
    finally:
        driver.quit()


def save_to_json(filename='data_coolinarika/data.json'):
    print(dishes)
    with open(filename, 'w', encoding="utf-8") as json_file:
        json_file.write(str(dishes).replace("'", "\""))


def clear_data_and_images():
    # Clear data.json if it exists
    data_json_path = 'data_coolinarika/data.json'
    if os.path.exists(data_json_path):
        os.remove(data_json_path)
        print(f"Deleted {data_json_path}")

    # Clear the entire images folder if it exists
    images_folder_path = 'data_coolinarika/images'
    if os.path.exists(images_folder_path) and os.path.isdir(images_folder_path):
        shutil.rmtree(images_folder_path)
        print(f"Deleted {images_folder_path}")

def read_links_from_file(file_path):
    try:
        with open(file_path, 'r') as file:
            links = [line.strip() for line in file if line.strip()]
        return links
    except Exception as e:
        print(f"Error reading file: {e}")
        return []


def main(base_url):
    # Step 0 (optional): Clear old data
    clear_data_and_images()

    # Step 1: Scrape the listing page to get individual recipe URLs and save it to file
    #listing_url = base_url + '/recepti/by-coolinarika'
    #recipe_urls = scrape_listing_page(base_url, listing_url)

    # Step 2: Read links from file
    file_path = 'data_coolinarika/links.txt'
    recipe_urls = read_links_from_file(file_path)

    # Step 3: For each recipe URL, fetch the full recipe data
    dishes = scrape_recipes_parallel(recipe_urls)

    # Step 4: Save the scraped data to a JSON file
    save_to_json()




if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Scrape recipes from a list of URLs.")

    # Define optional arguments
    parser.add_argument(
        "--base-url",
        type=str,
        nargs="*",
        default="https://www.coolinarika.com",
        help="Base url for scraping recipes.",
    )

    # Parse arguments
    args = parser.parse_args()

    # Pass parsed arguments to the main function
    main(args.base_url)
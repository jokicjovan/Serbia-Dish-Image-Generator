import re
import uuid
from pathlib import Path

import requests
import time
from bs4 import BeautifulSoup
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import threading


BASE_DIR = Path(__file__).resolve().parents[1]

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


dishes = {
    "dishes": []
}
dishes_lock = threading.Lock()

def process_page(pageCounter):
    main_url = "https://www.recepti.com"
    image_folder = "images/"
    url = main_url + '/kuvar/glavna-jela/' + str(pageCounter * 16 + 1)

    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.content, 'html.parser')
        ul = soup.find('ul', class_='re-list')
        li_dishes = ul.find_all('li')
        for li_dish in li_dishes:
            dish = Dish()
            dish.name = li_dish.find('a').text.replace("\"", "").replace("'","")

            img = li_dish.find('img')
            dish.image_path = image_folder + str(uuid.uuid4()) + ".jpg"
            image = Image.open(requests.get(main_url + img['src'], stream=True).raw)
            image.save(BASE_DIR / "data/raw/recepti/" / dish.image_path)

            dish_url = main_url + li_dish.find('a')['href']
            dish_response = requests.get(dish_url)
            dish_soup = BeautifulSoup(dish_response.content, 'html.parser')
            li_ingridients = dish_soup.findAll('li', itemprop='ingredients')
            dish.ingredients = [re.sub(r'\xa0', '', li.text.replace("\n", "").strip().replace("\"", "").replace("'","").replace('\xad','')) for li in li_ingridients]
            li_instructions = dish_soup.findAll('li', itemprop='recipeInstructions')
            dish.preparation = [re.sub(r'\xa0', '', li.text.replace("\n", "").strip().replace("\"", "").replace("'","").replace('\xad','')) for li in li_instructions] 
            with dishes_lock:
                dishes["dishes"].append(dish.make_json())
    else:
        print(f"Failed to retrieve the page. Status code: {response.status_code}")

start_time = time.time()
pages = 606

with ThreadPoolExecutor() as executor:
    executor.map(process_page, range(pages))

with open(BASE_DIR / "data/raw/recepti/dishes.json", "w", encoding="utf-8") as file:
    file.write(str(dishes).replace("'", "\""))

print(f"Execution time for {16*pages} images: {time.time() - start_time} seconds")

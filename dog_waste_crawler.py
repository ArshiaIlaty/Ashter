import os
import requests
from bs4 import BeautifulSoup
from googlesearch import search
from google_images_download import google_images_download
import time

def download_images(query, num_images, save_folder):
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    search_query = f"{query} images"
    urls = []

    for url in search(search_query, num_results=num_images):
        urls.append(url)
        print(url)  # Print the URL to verify

        if len(urls) >= num_images:
            break
        
        time.sleep(1)  # Add a 1-second delay between requests

    print(f"Total URLs collected: {len(urls)}")
    
    response = google_images_download.googleimagesdownload()
    arguments = {"keywords": query, "limit": num_images, "output_directory": save_folder}
    paths = response.download(arguments)
    print(paths)

    for i, url in enumerate(urls):
        print(f"Downloading image {i+1}/{num_images}: {url}")
        try:
            img_data = requests.get(url).content
            img_name = f"{query}_{i+1}.jpg"
            img_path = os.path.join(save_folder, img_name)

            with open(img_path, 'wb') as f:
                f.write(img_data)
        except Exception as e:
            print(f"Error downloading image {i+1}: {e}")

    print(f"Total images downloaded: {len(urls)}")

if __name__ == "__main__":
    query = "dog waste"
    num_images = 5
    save_folder = "dog_waste_images"
    download_images(query, num_images, save_folder)
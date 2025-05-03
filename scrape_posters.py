import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re

# Where the posters will be saved
os.makedirs("posters", exist_ok=True)

# Target page
base_url = "https://www.moviepostersgallery.com"
gallery_url = f"{base_url}/meryl-streep/"

# Fetch and parse the page
response = requests.get(gallery_url)
soup = BeautifulSoup(response.text, "html.parser")

# Function to normalize file names
def normalize_title(title):
    title = title.lower().strip()
    title = re.sub(r"[^\w\s]", "", title)  # remove punctuation
    return title.replace(" ", "_") + ".jpg"

# Find all images
images = soup.find_all("img")
downloaded = []

for img in images:
    src = img.get("src")
    alt = img.get("alt")

    if not src or not alt:
        continue
    if "posters" not in src:
        continue

    # Normalize and build filename
    filename = normalize_title(alt)
    poster_url = urljoin(base_url, src)

    try:
        img_data = requests.get(poster_url).content
        with open(os.path.join("posters", filename), "wb") as f:
            f.write(img_data)
        downloaded.append(filename)
        print(f"✅ Downloaded: {filename}")
    except Exception as e:
        print(f"❌ Failed: {poster_url} — {e}")

print(f"\nDone. Downloaded {len(downloaded)} posters.")

import requests
import os

# URL of the dataset raw file in GitHub
url="https://raw.githubusercontent.com/kimiaa-hrr/AI-CPS/refs/heads/main/data/Dataset/sports_performance_data.csv"


# Download file
r=requests.get(url)
with open("data/Dataset/scraped_dataset.csv","wb") as f:
    f.write(r.content)

print("Dataset Downloaded")
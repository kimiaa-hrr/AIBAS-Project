import requests

raw_url = "https://raw.githubusercontent.com/kimiaa-hrr/AIBAS-Project/refs/heads/main/data/dataset/README.md"
readme_text = requests.get(raw_url).text
print(readme_text)

try:
    with open("../data/dataset/scrapedDataset.csv", "w") as f:
        f.write(readme_text)
        print("Saved as CSV file!")
except:
    print("A problem has occured while saving the CSV file!")





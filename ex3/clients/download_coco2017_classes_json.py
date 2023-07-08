# A script for extracting the category names from the COCO 2017 dataset annotations and storing them in a JSON file.
# Adapted from https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/ (with help of ChatGPT)
import os
import urllib.request
import zipfile
import json
import shutil

REMOVE_ANNOTATIONS_FOLDER = (
    True  # remove the annotations folder after extracting the JSON file
)

categories_json_path = os.path.join("data", "coco2017_categories.json")
if os.path.exists(categories_json_path):
    print(
        f"COCO 2017 dataset categories JSON file already exists in '{categories_json_path}'."
    )
    exit()


def write_json_to_file(data, path):
    with open(path, "w") as file:
        json.dump(data, file)


print("Downloading COCO 2017 train/val annotation file...")
url_2017 = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
filename_2017 = "annotations_trainval2017.zip"
urllib.request.urlretrieve(url_2017, filename_2017)

print("Extracting annotation zip archive...")
with zipfile.ZipFile(filename_2017, "r") as zip_ref:
    zip_ref.extractall("data")

# Remove the zip file
os.remove(filename_2017)


json_file_2017 = os.path.join("data", "annotations", "instances_val2017.json")

with open(json_file_2017, "r") as COCO:
    js = json.loads(COCO.read())
    categories_data = js["categories"]

write_json_to_file(categories_data, categories_json_path)
print("JSON file for 2017 version created and saved.")

if REMOVE_ANNOTATIONS_FOLDER:
    shutil.rmtree(os.path.join("data", "annotations"))
    print("Removed files used to create file.")

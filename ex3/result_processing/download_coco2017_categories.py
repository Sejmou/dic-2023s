import os
import urllib.request
import zipfile
import json
import shutil

REMOVE_ANNOTATIONS_FOLDER = (
    True  # remove the annotations folder after extracting the JSON file
)

script_directory = os.path.dirname(os.path.abspath(__file__))
categories_json_path = os.path.join(script_directory, "coco2017_categories.json")
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
filename_2017 = os.path.join(script_directory, "annotations_trainval2017.zip")
urllib.request.urlretrieve(url_2017, filename_2017)

print("Extracting annotation zip archive...")
with zipfile.ZipFile(filename_2017, "r") as zip_ref:
    zip_ref.extractall(script_directory)

# Remove the zip file
os.remove(filename_2017)

json_file_2017 = os.path.join(script_directory, "annotations", "instances_val2017.json")

with open(json_file_2017, "r") as coco_file:
    coco_data = json.load(coco_file)
    categories_data = coco_data["categories"]

write_json_to_file(categories_data, categories_json_path)
print("JSON file for the 2017 version created and saved.")

if REMOVE_ANNOTATIONS_FOLDER:
    shutil.rmtree(os.path.join(script_directory, "annotations"))
    print("Removed files used to create the JSON file.")

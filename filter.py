import json
from tqdm import tqdm


def write_json(new_data, filename='result.json'):
    with open(filename, 'r+') as file:
        # First we load existing data into a dict.
        file_data = json.load(file)
        # Join new_data with file_data inside emp_details
        file_data["results"].append(new_data)
        # Sets file's current position at offset.
        file.seek(0)
        # convert back to json.
        json.dump(file_data, file, indent=4)


filename = "results3.json"
with open(filename, 'r') as file:
    # First we load existing data into a dict.
    file_data = json.load(file)
    # Join new_data with file_data inside emp_details
    for result in tqdm(file_data['results']):
        if not (result['aggr'] == 'exp' or result["int_owa"] == "exp"):
            write_json(result, "results4.json")

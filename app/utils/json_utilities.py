import os
import json

def read_json_files(directory):
    """
    Read JSON files from a directory and return their content as a list of dictionaries.

    Args:
        directory (str): Path to the directory containing JSON files.

    Returns:
        list: A list of dictionaries, each representing the content of a JSON file.
    """
    json_files = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            with open(os.path.join(directory, filename), 'r') as f:
                json_data = json.load(f)
                json_files.append({'filename': filename, 'content': json_data})
    return json_files


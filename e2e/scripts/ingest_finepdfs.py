import os
import pandas as pd
import json
from urllib.parse import urlparse
from datasets import load_dataset

# 1. Load the dataset from finepdfs from hugging face
# 2. Iterate through streaming finepdfs and follow the passages JSON schema
# 3. Merge the finePdfs and the frames json to get one json for ingestion for the single shot retrival.

def get_base_filename(url):
    return os.path.basename(urlparse(url).path)

def finepdfs_to_json(iteration,frames_passage_json,output_json):
    dataset = load_dataset("HuggingFaceFW/finepdfs",split="train",streaming=True)

    track = 0
    streaming_data = {
        "base_dir" : "/work/processed_html",
        "passages" : []
}

    print("Iterating through streaming dataset")
    for track, item in enumerate(dataset):
        if track >= iteration:
            break
        streaming_data["passages"].append({
            "index": track,
            "base_filename": get_base_filename(item['url']),
            "original_url": item['url'],
            "passage": item['text']
        })
        track+=1

    print("Dumping fine pdfs into a json")
    with open("finepdf_passages.json", "w", encoding="utf-8",errors="strict") as f:
        json.dump(streaming_data, f, indent=2, ensure_ascii=False)

    print("Loading existing JSONS to merge")
    with open("finepdf_passages.json", "r", encoding="utf-8") as json1, open(frames_passage_json, "r", encoding="utf-8") as json2:
        finepdf_json = json.load(json1)
        frames_json = json.load(json2)

    start = len(finepdf_json['passages'])

    for entry in frames_json['passages']:
        finepdf_json['passages'].append({
            "index" : start,
            "base_filename" : entry['base_filename'],
            "original_url" : entry['original_url'],
            "passage" : entry['passage']
        })
        start+=1

    with open(output_json, "w", encoding="utf-8") as out:
        json.dump(finepdf_json, out, indent=2, ensure_ascii=False)

    print(f"Merged JSON including FRAMES and FinePDF DATA is saved to: {output_json}...")

#set paths
frames_passage_json = "/work/passages/doc_html_len256_overlap32_word.json"
output_json = "/work/finepdfs_frames_output.json"

finepdfs_to_json(100,frames_passage_json,output_json)

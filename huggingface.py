import glob
import os
import json
from pathlib import Path

from datasets import Dataset


# one dataset with the city metadata only
sites = []
paths = [
    p for p in glob.glob("../scraped/*.json")
    if len(os.path.splitext(os.path.basename(p))[0]) == 5
]
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    site_id = Path(path).stem
    sites.append({
        "site_id": site_id,
        "site": data["site"],
        "subgrouping": data["subgrouping"],
        "sentences": list(data["sentences"].keys()),  # avoid nested
    })
print(len(sites))
site_dataset = Dataset.from_list(sites)
site_dataset.push_to_hub('kalbin/yubao_sites')


# one dataset with each video's metadata separately
video_records = []
for site in sites:
    for sentence_id, sentence_data in site["sentences"].items():
        video_example = {
            **sentence_data,
            "utterance_id": sentence_id,
            "site": site["site"],
            "subgrouping": site["subgrouping"]
        }
        video_records.append(video_example)
video_dataset = Dataset.from_list(video_records)
video_dataset.push_to_hub('kalbin/yubao_videos')

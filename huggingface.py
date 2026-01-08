import glob
import os
import json
from pathlib import Path

from datasets import Dataset
from retrieval import text_normalize_chinese


# assumes you've downloaded city metadata and stored in a file like: scraped/02G38.json
# one dataset with the city metadata only
sites = []
# one dataset with each video's metadata separately
video_records = []
paths = [
    p for p in glob.glob("../scraped/*.json")
    if len(os.path.splitext(os.path.basename(p))[0]) == 5
]
for path in paths:
    with open(path, "r", encoding="utf-8") as f:
        site_data = json.load(f)

    site_id = Path(path).stem
    sites.append({
        "site_id": site_id,
        "site": site_data["site"],
        "subgrouping": site_data["subgrouping"],
        "sentences": list(site_data["sentences"].keys()),  # avoid nested
    })

    for sentence_id, sentence_data in site_data["sentences"].items():
        video_example = {
            **sentence_data,
            "utterance_id": sentence_id,
            "site": site_data["site"],
            "subgrouping": site_data["subgrouping"]
        }
        video_example["transcript"] = text_normalize_chinese(video_example["transcript"])
        video_example["translation"] = text_normalize_chinese(video_example["translation"])
        # to preprocess the IPA, see https://github.com/b05102139/phonotactic-complexity-across-dialects/blob/main/data/min/preprocess_min.py#L14
        video_records.append(video_example)
print(len(sites), "sites")
print(len(video_records), "utterances")

site_dataset = Dataset.from_list(sites)
site_dataset.push_to_hub('kalbin/yubao_sites')

video_dataset = Dataset.from_list(video_records)
video_dataset.push_to_hub('kalbin/yubao_videos')

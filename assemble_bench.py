import argparse
from collections import defaultdict
import json
import logging
from os import PathLike
from pathlib import Path
import re
from typing import Dict, Set, Optional, Tuple

from datasets import load_dataset, Dataset
from lhotse import (
    validate_recordings_and_supervisions,
    CutSet
)
from lhotse.audio import Recording, RecordingSet
from lhotse.qa import fix_manifests
from lhotse.manipulation import combine
from lhotse.supervision import SupervisionSegment, SupervisionSet
from tqdm.notebook import tqdm
import yaml


def get_cities(corpus_dir):
    # some sites do not have all 50 sentences
    FULLY_SCRAPED_THRESHOLD = 45
    cities = {}
    for filepath in Path(corpus_dir).iterdir():
        site_id = filepath.stem
        if filepath.suffix == '.json' and len(site_id) == 5:
            with open(filepath, 'r') as f:
                site_data = json.load(f)
            # TODO: make sure sentences is not empty

            # has audio and was not partially scraped
            if (Path(corpus_dir) / site_data['site']).exists() and len(list((Path(corpus_dir) / site_data['site']).iterdir())) > FULLY_SCRAPED_THRESHOLD:
                cities[site_id] = site_data['site']
    return cities

def text_normalize_chinese(text):
    # remove anything between / and 。, which indicates a translation
    # ex: 这辆汽车要开到广州去。/这辆汽车要开去广州。这辆汽车要开到广州去。/这辆汽车要开去广州。
    # -> 这辆汽车要开到广州去。这辆汽车要开到广州去。

    # only allow Han characters
    # remove punctuation (Latin and Chinese)
    # and the weird placeholders (e.g. #56)
    # https://stackoverflow.com/questions/1366068/whats-the-complete-range-for-chinese-characters-in-unicode
    pattern = '(/[^。]*。|[^\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002a6df\U0002a700-\U0002b73f\U0002b740-\U0002b81f\U0002b820-\U0002ceaf\U0002ceb0-\U0002ebef\U00030000-\U0003134f\U00031350-\U000323af])'
    text = text.lower()
    return re.sub(pattern, '', text)


def _parse_utterance(
    corpus_dir,
    utt_id,
    text,
    site_name,
    subgroup
) -> Optional[Tuple[Recording, SupervisionSegment]]:
    """
    Generate audio metadata for a single utterance
    """
    audio_path = (Path(corpus_dir) / site_name / utt_id).with_suffix('.wav')

    if not audio_path.is_file():
        print(f"No such file: {audio_path}")
        logging.info(f"No such file: {audio_path}")
        return None

    recording = Recording.from_file(path=audio_path, recording_id=utt_id)
    # ensures the audio file doesn't have any issues
    recording.load_audio()

    segment = SupervisionSegment(
        id=utt_id,
        recording_id=utt_id,
        start=0.0,
        duration=recording.duration,
        channel=0,
        language='Chinese', # can be overridden
        speaker='aaaaa',
        text=text,  # transcript and translation are added later
        custom={
            'city': site_name,
            'subgrouping': subgroup
        }
    )

    return recording, segment


def index_by_city(dataset):
    city_index = defaultdict(list)
    for ex in dataset:
        city = ex["utt_id"][:5]
        city_index[city].append(ex)
    return city_index


def _prepare_city(
    corpus_dir,
    site_name,
    utterances
) -> Tuple[CutSet, CutSet]:
    """
    Generate audio metadata for all ~50 utterances in a site
    """
    recordings, supervisions = [], []

    for ex in utterances:
        sentence_id = ex["utterance_id"]
        subgrouping = ex["subgrouping"]
        dialect_transcript = ex["transcript"]
        mandarin_translation = ex["translation"]

        recording, supervision = _parse_utterance(corpus_dir, sentence_id, '', site_name, subgrouping)
        recordings.append(recording)

        transcript, translation = text_normalize_chinese(dialect_transcript), text_normalize_chinese(mandarin_translation)
        supervision.text = transcript
        supervision.translation = translation
        supervisions.append(supervision)

    recording_set = RecordingSet.from_recordings(recordings)
    supervision_set = SupervisionSet.from_segments(supervisions)
    recording_set, supervision_set = fix_manifests(recording_set, supervision_set)
    validate_recordings_and_supervisions(recording_set, supervision_set)

    return CutSet.from_manifests(recording_set, supervision_set)


def prepare_yubao(
    corpus_dir: PathLike,
    manual_list: Dict[str, Set[str]],
    utterance_metadata: Dataset,
    output_dir: Optional[PathLike] = None,
):
    """
    Generate CutSet (audio metadata) that holds both ASR transcript and translation
        at the corpus level (all of yubao), subgroup level, and site level
    """
    corpus_dir = Path(corpus_dir)
    assert corpus_dir.is_dir(), f"No such directory: {corpus_dir}"
    assert output_dir is not None, (
        "This recipe requires to specify the output "
        "manifest directory (output_dir cannot be None)."
    )
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cuts = []
    groups = {}
    city2utterances = index_by_city(utterance_metadata)
    for subgroup, subgroup_set in tqdm(sorted(manual_list.items())):
        for site_name in tqdm(sorted(subgroup_set)):
            utterances = city2utterances[site_name]
            cutset = _prepare_city(corpus_dir, site_name, utterances)

            if subgroup not in groups:
                groups[subgroup] = [cutset]
            else:
                groups[subgroup].append(cutset)
        
            # city-level cutset
            cutset.to_file(
                f"{output_dir}/yubao-{site_name}.jsonl.gz"
            )
            cuts.append(cutset)

        # generate one cutset per subgroup
        subgroup_cutset = combine(groups[subgroup])
        subgroup_cutset.to_file(
            f"{output_dir}/{subgroup}-yubao.jsonl.gz"
        )

    final_cutset = combine(cuts)
    if output_dir is not None:
        final_cutset.to_file(
            f"{output_dir}/yubao.jsonl.gz"
        )

    return final_cutset


def get_subgroup(subgrouping, city):
    if not subgrouping:
        return None
    
    subgroup = ''
    mapping = {
        '闽': 'min',
        '粤': 'yue',
        '官': 'zh',
        '吴': 'wuu',
        '客': 'hak',
        '粤': 'yue',
        '赣': 'gan',
        '湘': 'hsn'
    }

    # add more to this set as we expand the corpus
    manual_label = {
        '潮汕': '闽',
        '上海': '吴',
        '杭州': '吴',
        '南昌': '赣',
        '长沙': '湘',
        '文昌': '闽',
        '沈阳': '官',
        '太原': '晋', # exclude Jin for now
        '秀山': '官',
        '张家港': '吴',
        '兰州': '官'
    }
    for keyword, sub in manual_label.items():
        if keyword in subgrouping or keyword in city:
            subgroup = sub
            break

    if not subgroup and len(subgrouping) < 20:
        # when the subgroup was already provided and determined while scraping
        subgroup = subgrouping[0]
        # TODO: other edge cases

    if subgroup not in mapping:        
        subgroup = ''
        for keyword in {'官', '吴', '闽', '客', '粤', '赣', '湘'}:
            if keyword in subgrouping:
                subgroup = keyword
                break
    # TODO: catch when it says 属于

    if subgroup not in mapping:
        logging.info(f'{subgrouping}: subgroup could not be determined - best we got was {subgroup}')

        return None
    return mapping[subgroup]


def get_sentence_id(cut, corpus):
    """
    specific to YuBao
    """
    if corpus == "yubao":
        # ex: 10G71mb01yf0001-0 -> 10G71mb01yf0001 -> 1
        cut_id = cut.id
        video_id = cut_id.split('-')[0]
        sentence_id = int(video_id[-4:])
    elif corpus == "fleurs":
        supervision_id = cut.supervisions[0].id
        # https://github.com/lhotse-speech/lhotse/blob/8c56a3e66a856bb6fbdf856e9c4eb378565055f5/lhotse/recipes/fleurs.py#L379
        # promptid_counter_filename - there are many instances of a promptid, which counter seeks to record
        prompt_id, counter, filename = supervision_id.split('_')
        # we only want 1 utterance per prompt id
        # thus we can return the counter
        return prompt_id + '-' + counter
    else:
        raise Exception(corpus + ' is not a supported corpus')
    return sentence_id


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assemble the retrieval dataset.")
    parser.add_argument("--audio_dir", help="The directory where the audio files are located.")
    parser.add_argument("--hf_cache_dir",  help="HuggingFace cache directory.")
    args = parser.parse_args()
    BASE_DIR = args.audio_dir

    logging.basicConfig(
        level=logging.INFO,  # Set the minimum log level you want to see
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler()]  # Ensures output goes to stdout (the notebook)
    )

    # load metadata: list of sites for each subgroup and mapping from name to id
    with open('retrieval_sites.yaml', 'r') as f:
        manual_site_list = yaml.safe_load(f)
    assert len(manual_site_list) == 8
    # all subgroups have the same number of cities
    assert len(set([len(manual_site_list[subgroup]) for subgroup in manual_site_list if subgroup != 'zh_standard' ])) == 1
    # to update: siteid_to_name = get_cities(BASE_DIR)
    # with open(f'siteid_to_name.json', 'r', encoding='utf-8') as f:
    #     siteid_to_name = json.load(f)
    # sitename_to_id = { site_name: site_id for site_id, site_name in siteid_to_name.items() }
    utterance_metadata = load_dataset("kalbin/yubao_videos", cache_dir=args.hf_cache_dir)

    # generate cutsets and save to cutsets/
    cuts = prepare_yubao(args.audio_dir, manual_site_list, utterance_metadata, 'cutsets')
    logging.info(cuts.describe())

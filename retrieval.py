import logging
import yaml
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

import hydra
from hydra.core.hydra_config import HydraConfig
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from lhotse import CutSet, Fbank, FbankConfig
from lhotse.dataset import (
    SimpleCutSampler,
    K2SpeechRecognitionDataset,
    OnTheFlyFeatures
)
from lhotse.features.kaldi.extractors import Mfcc, MfccConfig


def load_model(model_path):
    raise NotImplementedError('Implement method to load your speech encoder/model here')


def get_encoder_feats(model, model_name, feats, num_frames):
    raise NotImplementedError('Implement method to load your speech encoder embeddings here')


def slice_encoder_feats(encoder_feats, num_frames):
    # no slicing needed for Zipformer or TeleSpeech b/c we didn't pad the input
    # return encoder_feats

    # but if we did pad the input to the encoder
    # then we can slice the encoder features to remove embeddings corresponding to padding
    # encoder_feats: (batch size, T, C)
    # divide by 4 since Zipformer downsamples from 100 Hz to 25 Hz (Zipformer Figure 1)
    raise NotImplementedError('Implement method to slice encoder features if you implemented padding')


"""
SeqSim - BERTScore (Zhang et al. 2020) adapted for speech embeddings (Ma et al. 2025)

Measures the similarity between 2 speech utterances, computing frame-level similarity

src: T1 x D
trg: T2 x D
T1/T2 is the duration of the source/target in encoder frames, D is the embedding dimension

where src is treated as a hypothesis and trg, the reference, for the purposes of simulating precision and recall
"""
def seq_sim(src, trg):
    # T1 x T2 - cosine similarity between all pairs of frames 
    frame_similarity_matrix = F.cosine_similarity(src, trg, dim=1)
    # for each elem in src, take the max over the trg dim then sum and normalize by # elems (src dim)
    precision = torch.sum(torch.max(frame_similarity_matrix, dim=1)) / src.shape[0]
    # for each elem in trg, take the max over the src dim then sum and normalize by # elems (trg dim)
    recall = torch.sum(torch.max(frame_similarity_matrix, dim=0)) / trg.shape[0]

    bert_score = 2 * precision * recall / (precision + recall)
    return bert_score


"""
Given embeddings for paired source-target utterances,
for each utterance, find the utterance in this corpus with the highest similarity (SeqSim) to the utterance

src_emb: List[Tensor[T1, D]] of length N
trg_emb: List[Tensor[T2, D]] of length N
(N is the number of aligned utterances/~50, T1/T2 is the duration of the source/target in encoder frames, D is the embedding dimension)
"""
def retrieve_most_similar_utterance(src_utt_emb, trg_utt_emb):
    assert len(src_utt_emb) == len(trg_utt_emb)
    N_aligned_utt = len(src_utt_emb)
    utt_similarity_matrix = torch.zeros((N_aligned_utt, N_aligned_utt))

    # for all pairs of utterances (not aligned), measure the sequence similarity
    for src_idx in range(N_aligned_utt):
        for trg_idx in range(N_aligned_utt):
            utt_similarity_matrix[src_idx][trg_idx] = seq_sim(src_utt_emb[src_idx], trg_utt_emb[trg_idx])

    # for each source utterance, find the target utterance with the highest similarity to the current utterance
    # could technically be extended to support more items being retrieved. for now, we retrieve the top
    most_similar_utterance = torch.argsort(utt_similarity_matrix, dim=1)[:, -1:]
    return most_similar_utterance


"""
Cross-lingual speech-to-speech retrieval performance for one pair of language varieties (languages or dialects)

Assumes variety_A_loader, variety_B_loader are semantically aligned
i.e. for index i, variety_A_loader[i] and variety_B_loader[i] have the same meaning
"""
@torch.no_grad
def retrieval(model, variety_A_loader, variety_B_loader, device, cfg):
    embed_A, embed_B = [], []
    for utt_A, utt_B in tqdm(zip(variety_A_loader, variety_B_loader)):
        num_frames_A, num_frames_B = utt_A["supervisions"]["num_frames"].to(device), utt_B["supervisions"]["num_frames"].to(device)
        feats_A, feats_B = utt_A["inputs"].to(device), utt_B["inputs"].to(device)

        utt_embed_A, utt_embed_B = \
            get_encoder_feats(model, cfg.model.name, feats_A, num_frames_A), \
            get_encoder_feats(model, cfg.model.name, feats_B, num_frames_B)

        # "only keep the embedding vectors that correspond to meaningful content in the original audio and remove the ones associated with the padded part."
        utt_embed_A, utt_embed_B = slice_encoder_feats(utt_embed_A, num_frames_A), slice_encoder_feats(utt_embed_B, num_frames_B)
        # remove the batch dim: (1, T, D) -> (T, D)
        embed_A.append(utt_embed_A.squeeze(dim=0))
        embed_B.append(utt_embed_B.squeeze(dim=0))

    most_similar_utterance = retrieve_most_similar_utterance(embed_A, embed_B)
    N_aligned_utt = len(embed_A)
    retrieval_success = 0
    # how many elements are on the diagonal of the source-target matrix?
    for row_idx in range(N_aligned_utt):
        # if the utterance in the target language with the same meaning as the source utterance 
        #   has the the highest similarity to the source utterance,
        #   then we retrieved the correct utterance
        # note: this assumes for each index, source file A is semantically aligned/paired with source file B (i.e. they have the same meaning)
        # thus the right answer is to predict the target utterance with the same index as the source index
        if row_idx in most_similar_utterance[row_idx]:
            retrieval_success += 1
    # R @ 1
    recall_rate = 100 * retrieval_success / N_aligned_utt
    return recall_rate


def save_retrieval_results(
    cfg,
    test_set_name: str,
    results: Dict[str, List[Dict[str, float]]],
):
    res_dir = Path(cfg.exp_dir) / "retrieval"

    if cfg.inference.iter > 0:
        suffix = f"iter-{cfg.inference.iter}-avg-{cfg.inference.avg}-retrieval"
    else:
        suffix = f"epoch-{cfg.inference.epoch}-avg-{cfg.inference.avg}-retrieval"

    s = "\nFor {}, recall is:\n".format(test_set_name)

    for group, group_results in results.items():
        path = (
            res_dir / f"recall-{test_set_name}-{suffix}.txt"
        )

        with open(path, "w") as f:
            print("Recall", file=f)
            r = 0
            for city, recall in group_results.items():
                print("{}\t{}".format(city, recall), file=f)
                r += recall

            mean_recall = r / len(group_results)
            print("{}\t{}".format(group, mean_recall), file=f)

            s += f"{group}: {mean_recall}\n"

    logging.info(s)


def get_test_cuts(test_data_config_file):
    testset_to_cuts = {}
    with open(test_data_config_file, 'r') as file:
        test_data_config = yaml.load(file, Loader=yaml.FullLoader)

    for test_set in test_data_config:
        logging.info(f"Getting {test_set['name']} cuts")
        cutset = CutSet.from_file(test_set['manifest'])
        # no need to specify the language of the utterance since we're only dealing with the encoder
        testset_name = test_set['name']

        testset_to_cuts[testset_name] = cutset
    return testset_to_cuts


def get_mfcc_extractor():
    # https://github.com/Tele-AI/TeleSpeech-ASR/blob/master/mfcc_hires.conf
    config = MfccConfig(
        sampling_rate=16000,
        num_ceps=40,
        use_energy=False,
        num_filters=40, # num_mel_bins deprecated
        low_freq=40.0,
        high_freq=-200.0
    )
    return Mfcc(config)


def get_fbank_extractor():
    return Fbank(FbankConfig(num_mel_bins=80))


def cutset_to_loader(cutset, cfg):
    # NOTE: override if needed, as some models use different features (e.g. TeleSpeech uses MFCCs)
    if cfg.model.feats == "fbank":
        feature_extractor = get_fbank_extractor()
    elif cfg.model.feats == "mfcc":
        feature_extractor = get_mfcc_extractor()
    features = OnTheFlyFeatures(feature_extractor)

    testset = K2SpeechRecognitionDataset(
        input_strategy=features,
        return_cuts=True,
    )
    # sample each utterance one at a time b/c we need them in order
    sampler = SimpleCutSampler(
        cutset,
        max_cuts=1,
        shuffle=False,
    )
    return DataLoader(
        testset,
        batch_size=None,
        sampler=sampler,
        num_workers=cfg.data.num_workers,
    )


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


def align_yubao(source_cutset, target_cutset, corpus):
    """
    Generates paired utterances for retrieval, aligning based on sentence ID

    Assumes both source_cutset and target_cutset consist of multiple cities,
    where each city has ~ 50 sentences

    e.g. one dataset consists of multiple Wu varieties (~50 sent. each), another dataset consists of multiple Min varieties (~50 sent. each)

    Aligns using the sentence ID

    Returns:
    - Dict[str, Tuple[CutSet, CutSet]], a mapping from city to a pair of aligned source and target utterances
    """
    source_city_id_cut, target_city_id_cut = defaultdict(dict), defaultdict(dict)
    for cut in source_cutset:
        sentence_id = get_sentence_id(cut, corpus)
        city = cut.supervisions[0].custom['city']
        source_city_id_cut[city][sentence_id] = cut
    for cut in target_cutset:
        sentence_id = get_sentence_id(cut, corpus)
        city = cut.supervisions[0].custom['city']
        target_city_id_cut[city][sentence_id] = cut

    aligned_pairs = {}
    for src_city, src_ids in source_city_id_cut.items():
        for trg_city, trg_ids in target_city_id_cut.items():
            aligned_src, aligned_trg = [], []
            for trg_cut_id in trg_ids:
                # Only include an utterance if it appears in both the source and the target
                # Important b/c some sites in YuBao won't have 50 sentences
                if trg_cut_id in src_ids:
                    aligned_src.append(src_ids[trg_cut_id])
                    aligned_trg.append(trg_ids[trg_cut_id])
            assert len(aligned_src) == len(aligned_trg)
            aligned_pairs[(src_city, trg_city)] = (CutSet.from_cuts(aligned_src), CutSet.from_cuts(aligned_trg))
    return aligned_pairs


def align_fleurs(source_cutset, target_cutset, corpus):
    """
    Generates paired utterances for retrieval, aligning based on sentence ID

    Assumes both datasets contain semantically aligned corpora, but utterances with the same index
        are not aligned

    This function pairs semanitcally aligned cuts across both corpora using the sentence ID

    Returns:
    - Dict[str, Tuple[CutSet, CutSet]], a mapping from city to a pair of aligned source and target utterances
    """
    source_id_to_cut = {}
    for cut in source_cutset:
        sentence_id = get_sentence_id(cut, corpus)
        if sentence_id.split('-')[1] != '1':
            # only keep the first instance of each prompt to avoid duplicates
            continue

        source_id_to_cut[sentence_id] = cut

    aligned_source, aligned_target = [], []
    target_city_to_cut = defaultdict(list)
    for target_cut in target_cutset:
        sentence_id = get_sentence_id(target_cut, corpus)

        # Only include an utterance if it appears in both the source and the target
        if sentence_id in source_id_to_cut:
            aligned_source.append(source_id_to_cut[sentence_id])
            aligned_target.append(target_cut)

    assert len(aligned_source) == len(aligned_target)
    return (CutSet.from_cuts(aligned_source), CutSet.from_cuts(aligned_target))


@hydra.main(version_base=None, config_path="configs", config_name="asr_inference")
@torch.no_grad()
def main(cfg: DictConfig):
    logging.info(f"Hydra log directory: {HydraConfig.get().run.dir}")
    logging.info("\n" + OmegaConf.to_yaml(cfg))

    assert cfg.retrieval.source.split('_')[0] == cfg.retrieval.target.split('_')[0], "source and target must come from the same corpus"
    assert cfg.retrieval.source != cfg.retrieval.target, "source and target cannot be the same"
    corpus = cfg.retrieval.source.split('_')[0]

    # get the lhotse cutsets (speech metadata)
    # do not load the data just yet though, since we need to align source and target for retrieval
    testset_to_cuts = get_test_cuts(cfg.data.test_data_config)

    # initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg.inference.pretrained_model)
    model.eval()
    model.to(device)

    # result dir
    Path(f"{cfg.exp_dir}/retrieval/{corpus}").mkdir(parents=True, exist_ok=True)
    results = defaultdict(dict)

    # the retrieval src/target lang is a single language (zh, en, fr, de, jp) except when dialect is specified
    # in this case, there are multiple cities within a dialect subgroup
    if "yubao" in cfg.retrieval.source or "yubao" in cfg.retrieval.target:
        logging.info(f"Retrieval between the subgroups, src={cfg.retrieval.source} and trg={cfg.retrieval.target}")

        # Mandarin is a single site
        aligned_pairs = align_yubao(
            testset_to_cuts[cfg.retrieval.source],
            testset_to_cuts[cfg.retrieval.target],
            corpus
        )
        results = {}
        pair_results = defaultdict(float)
        for (src_city, trg_city) in aligned_pairs.keys():
            logging.info(f"City pair: src={src_city} and trg={trg_city}")

            aligned_source_cuts, aligned_target_cuts = aligned_pairs[(src_city, trg_city)]
            # could use CutPairsSampler in the future
            aligned_source_loader, aligned_target_loader = cutset_to_loader(aligned_source_cuts, cfg), \
                cutset_to_loader(aligned_target_cuts, cfg)

            recall_rate = retrieval(model, aligned_source_loader, aligned_target_loader, device, cfg)
            city_pair = f'{src_city}-{trg_city}'
            pair_results[city_pair] = recall_rate

        subgroup_pair = f'{cfg.retrieval.source}_{cfg.retrieval.target}'
        results[subgroup_pair] = pair_results
        logging.info(f"Recall for {subgroup_pair}: {sum(pair_results.values()) / len(pair_results.values())}")

        save_retrieval_results(cfg, subgroup_pair, results)
    else:
        # both the source and target are single languages
        logging.info(f"Retrieval where src={cfg.retrieval.source}, trg={cfg.retrieval.target}")

        aligned_source_cuts, aligned_target_cuts = align_fleurs(
            testset_to_cuts[cfg.retrieval.source],
            testset_to_cuts[cfg.retrieval.target],
            corpus
        )
        assert len(aligned_source_cuts) == len(aligned_target_cuts)
        aligned_source_loader, aligned_target_loader = cutset_to_loader(aligned_source_cuts, cfg), \
            cutset_to_loader(aligned_target_cuts, cfg)

        recall_rate = retrieval(model, aligned_source_loader, aligned_target_loader, device, cfg)
        pair = f'{cfg.retrieval.source}-{cfg.retrieval.target}'
        results[pair] = {
            pair: recall_rate
        }
        logging.info(f"Recall: {recall_rate}")
        save_retrieval_results(cfg, f'{cfg.retrieval.source}_{cfg.retrieval.target}', results)


if __name__ == "__main__":
    main()

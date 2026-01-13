# 中國語言保護 YuBao: A New Chinese Dialect Speech Benchmark

The Chinese Language Resources Protection Project Collection and Display Platform (which we nickname YuBao / 語保 for short) has speech, dialect transcripts, phonetic (IPA) transcriptions, and Mandarin translations for 1,000 characters, 1,200 words, and 50 sentences, all of which are parallel (semantically aligned), across 1,300+ sites in China.

We create a speech-to-speech retrieval benchmark to evaluate progress in speech technologies for Chinese dialects.
Due to the non-standardization of Chinese dialect writing, Chinese dialect ASR may not be as useful as Chinese dialect to Mandarin speech-to-text translation.
We thus measure the degree of cross-dialect semantic alignment of speech language models that support Chinese dialects.

Our benchmark consists of 50 parallel spoken sentences across 78 sites, spanning the seven major subgroups: Mandarin (including dialectal Mandarin), Yue, Min (Southern Min/Minnan for now), Hakka, Xiang, Wu, and Gan.
See [our paper](https://arxiv.org/abs/2601.07274) for more details.


Please cite [our paper](https://arxiv.org/abs/2601.07274), the original YuBao dataset, and Ma et al. (2025) who originally proposed cross-lingual speech retrieval to measure crosslingual alignment of speech embeddings:
```
@article{chang2026yubao,
    author  = {Kalvin Chang and Yiwen Shao and Jiahong Li and Dong Yu},
    title = {Towards Comprehensive Semantic Speech Embeddings for Chinese Dialects},
    journal = {arXiv preprint arXiv:2601.07274},
    year = {2026}
}

@misc{yubao,
  author       = {{Centre for the Protection of Language Resources of China}},
  title        = {The Chinese Language Resources Protection Project Collection and Display Platform},
  url          = "https://zhongguoyuyan.cn",
  month        = "",
  year         = 2023,
  annote       = ""
}

@inproceedings{ma-etal-2025-cross,
    title = "Cross-Lingual Transfer Learning for Speech Translation",
    author = "Ma, Rao  and
      Qian, Mengjie  and
      Fathullah, Yassir  and
      Tang, Siyuan  and
      Gales, Mark  and
      Knill, Kate",
    editor = "Chiruzzo, Luis  and
      Ritter, Alan  and
      Wang, Lu",
    booktitle = "Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 2: Short Papers)",
    month = apr,
    year = "2025",
    address = "Albuquerque, New Mexico",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.naacl-short.4/",
    doi = "10.18653/v1/2025.naacl-short.4",
    pages = "33--43",
    ISBN = "979-8-89176-190-2",
}
```

Note: The audio and video copyrights belong to the Centre for the Protection of Language Resources of China.


## Data
We release the metadata for all Chinese dialect sites at:
* Site data (site name, subgrouping): https://huggingface.co/datasets/kalbin/yubao_sites
* Video metadata (dialect transcript, Mandarin translation, IPA, English translation): https://huggingface.co/datasets/kalbin/yubao_videos
    * For an example of how to preprocess the IPA, see https://github.com/b05102139/phonotactic-complexity-across-dialects/blob/main/data/min/preprocess_min.py#L14

## Retrieval benchmark
To create the retrieval dataset, you must first download the metadata and videos, extract the audio, and then assemble the retrieval dataset.

* ```pip install -r requirements.txt```
* ```bash
    export HF_HUB_CACHE="/path/to/your/hf_cache
    hf download kalbin/yubao_sites --repo-type dataset
  ```

### Downloading the videos
* Create account on zhongguoyuyan.cn (you will need a Chinese phone number)
* Downloading the videos for a site (SITE_ID):
    * Only the video, not the audio, is currently available
    * Using Chrome, log into the website and manually go to a site (https://zhongguoyuyan.cn/point/SITE_ID)
    * Inspect Element to open the JavaScript code console
    * Paste the code in ```download_video.js``` into the browser's JavaScript code console
        * The script will click the buttons on the page to download the videos

Warning: Scraping videos takes a while because of the delay between downloading successive videos, which is why we only downloaded a subset of the videos

### Extracting audio
* Extract audio from video: ```./mp4_to_wav.sh``` (change the directory where you downloaded the videos)
    * Make sure ffmpeg is installed (https://github.com/espnet/espnet/blob/master/tools/installers/install_ffmpeg.sh)

### Assembling the retrieval dataset
Next, we create lhotse CutSets that will be loaded during retrieval

* ```mkdir scraped```
    * Then move the audio to ```scraped```
    * ```scraped``` should look like this, where audio files are located in scraped/SITE_NAME/UTT_ID.wav:
        ```
        scraped
        ├── 上海_上海_上海
        │   ├── 04G95mb01yf0001.mp4
        │   ├── 04G95mb01yf0001.wav
        │   ├── 04G95mb01yf0002.mp4
        │   ├── 04G95mb01yf0002.wav
        │   ├── 04G95mb01yf0003.mp4
        │   ├── 04G95mb01yf0003.wav
        │   ├── 04G95mb01yf0004.mp4
        │   ├── 04G95mb01yf0004.wav
        │   ├── 04G95mb01yf0005.mp4
        │   ├── 04G95mb01yf0005.wav
        │   ├── 04G95mb01yf0006.mp4
        │   ├── 04G95mb01yf0006.wav
        │   ├── 04G95mb01yf0007.mp4
        │   ├── 04G95mb01yf0007.wav
        │   ├── 04G95mb01yf0008.mp4
        ...
        ```
        (the video/mp4 files will not be used)
* ```mkdir cutsets```
    * This is where the lhotse CutSets will be stored
* ```python assemble_bench.py --audio_dir scraped --hf_cache_dir /path/to/your/hf_cache```


### Running speech-to-speech retrieval
* ```retrieval.sh```
    * replace with your model path
    * add code to load your model embeddingss here (see NotImplementedError)


## Site metadata
In case you want the latest version of YuBao's site metadata:
* Install dependencies (```pip install ddddocr pandas retrying```)
* ```python crawl.py EMAIL PASSWORD SAVE_DIRECTORY```
    * This will save metadata from each site
    * We adapted https://github.com/lernanto/sincomp's crawler
* Uploading to HuggingFace: ```python huggingface.py``` (replace the directory where you downloaded the metadata)
* Beware of duplicate site names - one possible solution is to manually rename some of the cities (as we have suggested in parentheses)
    * 江西_上饶市_广丰区: 18356, 18E30 (江西_上饶市_广丰区_min)
    * 浙江_温州_苍南: 08F60, 08K12 (浙江_温州_苍南_min)
    * 海南_三亚市_崖州区: 23J20, 23D67 (海南_三亚市_崖州区_min), 23C56
    * 福建_宁德_霞浦: 02545, 02A15 (福建_宁德_霞浦_min)
    * 福建_南平_浦城: 02A12 (福建_南平_浦城_min), 02G42
    * 海南_（无）_东方市: 23C84 (海南_（无）_东方市_min), 23J18

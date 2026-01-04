# 中國語言保護 Yubao

The Chinese Language Resources Protection Project Collection and Display Platform (which we nickname Yubao / 語保 for short) has speech, dialect transcripts, phonetic (IPA) transcriptions, and Mandarin translations for 1,000 characters, 1,200 words, and 50 sentences, all of which are parallel (semantically aligned), across 1,300+ sites in China.

We create a speech-to-speech retrieval benchmark to evaluate progress in speech technologies for Chinese dialects.
Due to the non-standardization of Chinese dialect writing, Chinese dialect ASR may not be as useful as Chinese dialect to Mandarin speech-to-text translation.
We thus measure the degree of cross-dialect semantic alignment of speech language models that support Chinese dialects.

Our benchmark consists of 50 parallel spoken sentences across 78 sites, spanning the seven major subgroups: Mandarin (including dialectal Mandarin), Yue, Min (Southern Min/Minnan for now), Hakka, Xiang, Wu, and Gan.



Please cite the original Yubao dataset:
```
@misc{yubao,
  author       = {{Centre for the Protection of Language Resources of China}},
  title        = {The Chinese Language Resources Protection Project Collection and Display Platform},
  url          = "https://zhongguoyuyan.cn",
  month        = "",
  year         = 2023,
  annote       = ""
}
```

Note: The audio or video copyrights belong to the Centre for the Protection of Language Resources of China.


## Data
We release the metadata for all Chinese dialects sites at:
* Site data (site name, subgrouping): https://huggingface.co/datasets/kalbin/yubao_sites
* Video metadata (dialect transcript, Mandarin translation, IPA, English translation): https://huggingface.co/datasets/kalbin/yubao_videos

## Retrieval benchmark
To create the retrieval dataset, you must first download the metadata and videos, extract the audio, and then assemble the retrieval dataset.

* ```pip install -r requirements.txt```
* ```bash
    export HF_HUB_CACHE="/path/to/your/hf_cache
    hf download kalbin/yubao_sites
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

Warning: Scraping videos takes a while because of the delay between downloading successive videos, which is why we only downloaded a subset of the videos

# 中國語言保護

Source: zhongguoyuyan.cn. All rights belong to the dataset maintainers.

* Site data: https://huggingface.co/datasets/kalbin/yubao_sites
* Video metadata: https://huggingface.co/datasets/kalbin/yubao_videos


## Instructions
* Create account on zhongguoyuyan.cn (you will need a Chinese phone number)

### Site data
In case you want the latest version of yubao's site dataset:
* Install requirements (```pip install ddddocr pandas retrying```)
* ```python crawl.py EMAIL PASSWORD SAVE_DIRECTORY```

### Video data
To get the original videos for a site (SITE_ID):
* Using Chrome, log into the website and manually go to a site (https://zhongguoyuyan.cn/point/SITE_ID)
* Inspect Element to open the JavaScript code console
* Paste the code in ```download_video.js``` into the browser's JavaScript code console
    * The script will click the buttons on the page to download the videos
* Extract audio from video: ```./mp4_to_wav.sh``` (change the directory where you downloaded the videos)
    * Make sure ffmpeg is installed
* Uploading to HuggingFace: ```python huggingface.py''' (replace the directory where you downloaded the videos)

Warning: Scraping videos takes a while because of the delay between downloading successive videos, which is why we only downloaded a subset of the videos

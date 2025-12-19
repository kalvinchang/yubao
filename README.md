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
To get the original videos:
* Using Chrome, log into the website and manually go to a site
* Inspect Element to open the JavaScript code console
* Paste the code in ```download_video.js``` into the browser's JavaScript code console
    * The script will click the buttons on the page to download the videos
* Extract audio from video: ```./mp4_to_wav.sh``` (change the directory where you downloaded the videos)

for file in scraped/*/*.mp4
do
    if [ ! -e "${file%.mp4}.wav" ]; then
        # exclude video stream
        # downsample from 44.1 kHz to 16 kHz for ASR
        ffmpeg -i "$file" -vn -ar 16000 -acodec pcm_s16le "${file%.mp4}.wav"
    fi
done

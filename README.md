Videoannotation
--

Applying pretrained DL models to annotate videos. 

Requirements
--
pytorch 1.4.0

torchvision 0.5.0 

Usage
--
    python run_generate_srt.py video.mkv

This will generate a srt file with the same name as the video file. 

Currently runs on the first 10 minutes of the video, annotating one frame every three seconds, and generates a subtitle that lasts 2 seconds.
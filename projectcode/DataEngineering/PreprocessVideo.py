## this code compress the video, convert to mp4 and delete the original video to extract frames ##

from projectcode.InputFileProcessor.PreprocessVidoes_CompressCutAndExtractFrames import VideoPreprocessor
import os
os.listdir()

vp = VideoPreprocessor(ffmpeg_path="ffmpeg")
mp4_path, n_frames = vp.process(
    input_path="/app/mediaFiles/videos/inputvideos/PouringWater.MOV",
    output_mp4="/app/mediaFiles/videos/inputvideos/PouringWater.mp4",
    frames_dir="/app/mediaFiles/videos/AnnotateVideos/RawVideos/WaterPour",
    # convert/encode options
    start_time=0, duration=20,      # trim first 20s
    scale_width=640, fps=15,        # resize + set fps
    video_codec="libx264", preset="fast", crf=30,
    keep_audio=True, faststart=True, overwrite=True,
    delete_original=True,           # delete original after successful convert
    # frame extraction options
    every_nth=1,                    # save every frame
    start_index=0,
    image_ext="jpg", jpeg_quality=95
)
print(f"MP4 saved to: {mp4_path}")
print(f"Frames saved: {n_frames}")


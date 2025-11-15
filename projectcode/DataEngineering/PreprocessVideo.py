from projectcode.InputFileProcessor.PreprocessVidoes_CompressCutAndExtractFrames import VideoPreprocessor
from pathlib import Path

# ================== PATHS ==================
SRC = "/app/mediaFiles/videos/InputVideos/ProteinShakeIV/proteinshake_IV.MP4"
OUT_MP4 = "/app/mediaFiles/videos/InputVideos/ProteinShakeIV/CompressedVideos/ProteinShakeInf.MP4"
#OUT_MP4 = "/app/mediaFiles/videos/InputVideos/ScoopVolume/FilledScoop/CompressedVideo/FilledScoop.MP4"
FRAMES_DIR = "/app/mediaFiles/videos/InputVideos/ProteinShakeIV/ExtractFrames/ProteinShakeInf/"

# OUT_MP4 = "/app/mediaFiles/videos/InputVideos/VideosToAnnotate/VideoInfFile/FishAndPotatoes/SlicingSweetPotatoes/SlicingSweetPotatoes.MP4"
# FRAMES_DIR = "/app/mediaFiles/videos/InputVideos/VideosToAnnotate/VideoInfFile/FishAndPotatoes/SlicingSweetPotatoes/"


# make sure output folders exist
Path(OUT_MP4).parent.mkdir(parents=True, exist_ok=True)
Path(FRAMES_DIR).mkdir(parents=True, exist_ok=True)

# init helper
vp = VideoPreprocessor(ffmpeg_path="ffmpeg")

# ========== STEP 1: COMPRESS VIDEO ==========
mp4_path = vp.convert_to_mp4(
    input_path=SRC,
    output_path=OUT_MP4,
    start_time=None,        # start at 0s
    duration=None,         # keep 20s segment
    scale_width=640,     # resize
    fps=15,              # set fps
    video_codec="libx264",
    preset="fast",
    crf=30,
    audio_codec="aac",
    audio_bitrate="96k",
    faststart=True,
    keep_audio=True,
    overwrite=True,
)
print("Compressed MP4:", mp4_path)

# optional: delete original after compression succeeds
# try:
#     Path(SRC).unlink()
#     print("Original deleted:", SRC)
# except Exception as e:
#     print("Delete skipped:", e)

# ========== STEP 2: EXTRACT FRAMES ==========
n_frames = vp.extract_frames(
    video_path=mp4_path,
    output_dir=FRAMES_DIR,
    image_ext="jpg",
    jpeg_quality=95,
    every_nth=1,         # save every frame
    start_index=0,
)
print(f"Frames saved in {FRAMES_DIR}: {n_frames}")

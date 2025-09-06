import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2


class VideoPreprocessor:
    """
    Convert video → MP4 (optional trim/resize), optionally delete the original, then extract frames.

    Example:
        vp = VideoPreprocessor(ffmpeg_path="ffmpeg")
        mp4_path, n_frames = vp.process(
            input_path="/app/mediaFiles/videos/inputvideos/pouringMilkandsugar.MOV",
            output_mp4="/app/mediaFiles/videos/labeledvideos/pouringMilkandsugar_trimmed.mp4",
            frames_dir="/app/mediaFiles/videos/AnnotateVideos/CoffeePour",
            # Conversion knobs
            start_time=0, duration=20, scale_width=640, fps=15,
            video_codec="libx264", preset="fast", crf=30,
            audio_codec="aac", audio_bitrate="96k", faststart=True,
            keep_audio=True, overwrite=True, delete_original=True,
            # Extraction knobs
            image_ext="jpg", jpeg_quality=95, every_nth=1, start_index=0
        )
    """

    def __init__(self, ffmpeg_path: str = "ffmpeg"):
        self.ffmpeg_path = ffmpeg_path

    # -------------------------- Public API --------------------------

    def process(
        self,
        input_path: str,
        output_mp4: str,
        frames_dir: str,
        *,
        # Convert/encode options
        start_time: Optional[float] = None,
        duration: Optional[float] = None,
        scale_width: Optional[int] = 640,
        fps: Optional[int] = 15,
        video_codec: str = "libx264",
        preset: str = "fast",
        crf: int = 30,
        audio_codec: str = "aac",
        audio_bitrate: str = "96k",
        faststart: bool = True,
        keep_audio: bool = True,
        overwrite: bool = True,
        delete_original: bool = True,
        # Frame extraction options
        image_ext: str = "jpg",
        jpeg_quality: int = 95,
        every_nth: int = 1,
        start_index: int = 0,
    ) -> Tuple[str, int]:
        """
        Returns:
            (output_mp4_path, num_frames_extracted)
        """
        input_path = str(Path(input_path))
        output_mp4 = str(Path(output_mp4))
        frames_dir = str(Path(frames_dir))

        self._ensure_exists(input_path)
        Path(frames_dir).mkdir(parents=True, exist_ok=True)
        Path(output_mp4).parent.mkdir(parents=True, exist_ok=True)

        # 1) Convert → MP4 (trim/resize if requested)
        mp4_path = self.convert_to_mp4(
            input_path=input_path,
            output_path=output_mp4,
            start_time=start_time,
            duration=duration,
            scale_width=scale_width,
            fps=fps,
            video_codec=video_codec,
            preset=preset,
            crf=crf,
            audio_codec=audio_codec,
            audio_bitrate=audio_bitrate,
            faststart=faststart,
            keep_audio=keep_audio,
            overwrite=overwrite,
        )

        # 2) Delete original only after successful conversion
        if delete_original:
            self._safe_delete_original(original_path=input_path, converted_path=mp4_path)

        # 3) Extract frames from the converted MP4
        n_frames = self.extract_frames(
            video_path=mp4_path,
            output_dir=frames_dir,
            image_ext=image_ext,
            jpeg_quality=jpeg_quality,
            every_nth=every_nth,
            start_index=start_index,
        )

        return mp4_path, n_frames

    # ----------------------- Conversion (FFmpeg) -----------------------

    def convert_to_mp4(
        self,
        *,
        input_path: str,
        output_path: str,
        start_time: Optional[float],
        duration: Optional[float],
        scale_width: Optional[int],
        fps: Optional[int],
        video_codec: str,
        preset: str,
        crf: int,
        audio_codec: str,
        audio_bitrate: str,
        faststart: bool,
        keep_audio: bool,
        overwrite: bool,
    ) -> str:
        """
        Build and run an FFmpeg command to convert (and optionally trim/resize) a video to MP4.
        """
        cmd = [self.ffmpeg_path]

        # Fast seek BEFORE -i is faster (keyframe-accurate seek comes after -i, but slower)
        if start_time is not None:
            cmd += ["-ss", str(start_time)]
        if duration is not None:
            cmd += ["-t", str(duration)]

        cmd += ["-i", input_path]

        vf_parts = []
        if scale_width:
            # Keep aspect ratio with height = -2 (even number).
            vf_parts.append(f"scale={scale_width}:-2")
        if fps:
            vf_parts.append(f"fps={fps}")

        if vf_parts:
            cmd += ["-vf", ",".join(vf_parts)]

        cmd += ["-c:v", video_codec, "-preset", preset, "-crf", str(crf)]

        if keep_audio:
            cmd += ["-c:a", audio_codec, "-b:a", audio_bitrate]
        else:
            cmd += ["-an"]

        if faststart:
            cmd += ["-movflags", "+faststart"]

        if overwrite:
            cmd += ["-y"]  # overwrite without asking

        cmd += [output_path]

        # Remove existing output if not overwriting and exists
        if not overwrite and Path(output_path).exists():
            raise FileExistsError(f"Output file exists and overwrite=False: {output_path}")

        self._run(cmd, "FFmpeg conversion failed")
        self._ensure_exists(output_path)
        return output_path

    # ---------------------- Frame Extraction (cv2) ----------------------

    def extract_frames(
        self,
        *,
        video_path: str,
        output_dir: str,
        image_ext: str = "jpg",
        jpeg_quality: int = 95,
        every_nth: int = 1,
        start_index: int = 0,
        filename_pattern: str = "frame_{:05d}.{}",
    ) -> int:
        """
        Extract frames using OpenCV. Saves every `every_nth` frame.

        Returns:
            int: number of frames saved.
        """
        self._ensure_exists(video_path)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video for reading: {video_path}")

        idx_in = 0
        idx_out = start_index
        saved = 0

        # Configure JPEG/PNG params
        if image_ext.lower() in ("jpg", "jpeg"):
            encode_ext = ".jpg"
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]
        elif image_ext.lower() == "png":
            encode_ext = ".png"
            encode_params = [int(cv2.IMWRITE_PNG_COMPRESSION), 3]  # moderate compression
        else:
            raise ValueError("image_ext must be 'jpg' or 'png'")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if idx_in % max(1, every_nth) == 0:
                filename = filename_pattern.format(idx_out, image_ext.lower())
                out_path = str(Path(output_dir) / filename)

                # Use cv2.imencode for quality control, then write bytes
                success, buf = cv2.imencode(encode_ext, frame, encode_params)
                if not success:
                    raise RuntimeError(f"Failed to encode frame {idx_in}")

                with open(out_path, "wb") as f:
                    f.write(buf.tobytes())

                idx_out += 1
                saved += 1

            idx_in += 1

        cap.release()
        return saved

    # ------------------------------ Utils ------------------------------

    @staticmethod
    def _run(cmd, err_msg: str):
        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{err_msg}\nCommand: {' '.join(map(str, cmd))}\nStderr:\n{e.stderr.decode(errors='ignore')}") from e

    @staticmethod
    def _ensure_exists(path: str):
        if not Path(path).exists():
            raise FileNotFoundError(f"Path not found: {path}")

    @staticmethod
    def _safe_delete_original(original_path: str, converted_path: str):
        # Only delete if both paths exist and are different files
        try:
            o = Path(original_path).resolve()
            c = Path(converted_path).resolve()
            if o.exists() and o != c:
                o.unlink()
        except Exception as ex:
            # Non-fatal—warn in logs if you have a logger. Here we silently continue.
            pass

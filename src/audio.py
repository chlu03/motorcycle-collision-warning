"""Add warning sounds to the result video and export it with H.264 + AAC."""
from moviepy.editor import AudioFileClip, CompositeAudioClip, VideoFileClip


def select_warning_frames(frames, fps, min_gap_s):
    """Keep high-risk frames that are at least `min_gap_s` seconds apart (one sound each)."""
    selected = []
    for frame in sorted(set(frames)):
        if not selected or (frame - selected[-1]) / fps >= min_gap_s:
            selected.append(frame)
    return selected


def export_video(video_path, output_path, sound_path=None, warning_times=()):
    """Re-encode `video_path` to `output_path`, inserting `sound_path` at each warning time.

    A warning close to the end of the video is shortened instead of being skipped.
    """
    video = VideoFileClip(str(video_path))
    sound = AudioFileClip(str(sound_path)) if sound_path and warning_times else None

    clips = []
    if sound is not None:
        for t in warning_times:
            remaining = video.duration - t
            if remaining <= 0:
                continue
            clip = sound if sound.duration <= remaining else sound.subclip(0, remaining)
            clips.append(clip.set_start(t))

    if clips:
        video = video.set_audio(CompositeAudioClip(clips).set_duration(video.duration))
        video.write_videofile(str(output_path), codec='libx264', audio_codec='aac', logger=None)
    else:
        video.write_videofile(str(output_path), codec='libx264', audio=False, logger=None)

    video.close()
    if sound is not None:
        sound.close()

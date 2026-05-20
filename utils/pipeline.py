from collections import Counter

from utils.frame_extractor import extract_frames
from utils.captioning import generate_captions
from utils.object_detection import detect_objects
from utils.motion_analysis import analyze_motion
from utils.video_renderer import create_captioned_video



def process_video(video_path):

    frames = extract_frames(video_path)

    captions = generate_captions(frames)

    objects = detect_objects(frames)

    motion = analyze_motion(
        video_path,
        objects
    )

    common_caption = Counter(
        captions
    ).most_common(1)[0][0]

    object_text = ", ".join(objects)

    final_caption = (
        f"The video likely shows {common_caption}. "
        f"Detected objects include {object_text}. "
        f"The motion pattern suggests {motion}."
    )

    output_path = "outputs/final_output.mp4"

    create_captioned_video(
        video_path,
        output_path,
        final_caption
    )

    return output_path, final_caption
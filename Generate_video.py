import os
from pickle import dump as p_dump

# Profiling
from typing import Tuple

from Video_Framework import VideoFramework

from support_scripts.utils import MastersModel
from support_scripts.utils import ModelSettingsManager
from support_scripts.sampling import sample_video_from_model


def generate_and_save_video(
    model_frame: VideoFramework, video_frame_range: Tuple[int, int], out_dir: str
):
    # video_frame_ranges = [(0, 599), (599, 1699), (1699, 2899)]

    os.makedirs(out_dir, exist_ok=True)
    original_video_array, output_video_array = sample_video_from_model(
        model_frame, video_frame_range, out_dir
    )
    with open(os.path.join(out_dir, "original_video_array.pickle"), "wb") as f:
        p_dump(original_video_array, f)
    with open(os.path.join(out_dir, "output_video_array.pickle"), "wb") as f:
        p_dump(output_video_array, f)

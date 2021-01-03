import os
from pickle import dump as p_dump

# Profiling
from Video_Framework import VideoFramework

from support_scripts.utils import MastersModel
from support_scripts.utils import ModelSettingsManager
from support_scripts.sampling import sample_video_from_model

if __name__ == '__main__':
    manager: ModelSettingsManager = ModelSettingsManager()

    # Create model framework
    # model_frame: VideoFramework = VideoFramework.from_model_settings_manager(manager)

    if not os.path.exists(manager.args["model_save_dir"]):
        raise EnvironmentError("Model save dir does not exist.")

    # Load model
    # model_frame.load_model(manager.args["load_saved_model"])
    model_frame = VideoFramework.load_model_with_embedded_settings(manager)

    # video_frame_ranges = [(0, 599), (599, 1699), (1699, 2899)]
    video_frame_range = (0, 2)

    out_dir: str = os.path.join(manager.args["image_save_dir"], "video_output/")

    os.makedirs(out_dir, exist_ok=True)
    original_video_array, output_video_array = sample_video_from_model(
        model_frame, video_frame_range, out_dir
    )
    with open(os.path.join(out_dir, "original_video_array.pickle"), "wb") as f:
        p_dump(original_video_array, f)
    with open(os.path.join(out_dir, "output_video_array.pickle"), "wb") as f:
        p_dump(output_video_array, f)
    pass

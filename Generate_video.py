import os
from pickle import dump as p_dump

# Profiling
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


from support_scripts.sampling import sample_video_from_model
from support_scripts.utils import MastersModel
from support_scripts.utils import ModelSettingsManager


manager: ModelSettingsManager = ModelSettingsManager()

model_frame: MastersModel
if manager.args["model"] == "CRN":
    from CRN import CRNFramework

    model_frame = CRNFramework.from_model_settings_manager(manager)
elif manager.args["model"] == "GAN":
    from GAN import GANFramework

    model_frame: GANFramework = GANFramework.from_model_settings_manager(manager)
    print(model_frame.generator)
else:
    raise SystemExit

if not os.path.exists(manager.args["model_save_dir"]):
    raise EnvironmentError("Model save dir does not exist.")

# Load model
model_frame.load_model(manager.args["load_saved_model"])

video_frame_ranges = [(0, 599), (599, 1699), (1699, 2899)]

out_dir: str = "./test_output2"
os.makedirs(out_dir, exist_ok=True)
original_video_array, output_video_array = sample_video_from_model(
    model_frame, video_frame_ranges[0], out_dir, output_image_number=0, batch_size=3
)
print(original_video_array.max(), original_video_array.min())
with open(os.path.join(out_dir, "original_video_array.pickle"), "wb") as f:
    p_dump(original_video_array, f)
with open(os.path.join(out_dir, "output_video_array.pickle"), "wb") as f:
    p_dump(output_video_array, f)
pass

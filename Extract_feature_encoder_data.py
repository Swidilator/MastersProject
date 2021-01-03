import os
import torch


from support_scripts.utils import ModelSettingsManager, MastersModel
from support_scripts.components import FeatureEncoder
from Video_Framework import VideoFramework

if __name__ == "__main__":
    # Initialise settings manager to read args and set up environment
    manager: ModelSettingsManager = ModelSettingsManager()

    model_frame = VideoFramework.load_model_with_embedded_settings(manager)

    # Generate folder for run
    if not os.path.exists(manager.args["base_model_save_dir"]):
        os.makedirs(manager.args["base_model_save_dir"])

    if not os.path.exists(manager.args["model_save_dir"]):
        os.makedirs(manager.args["model_save_dir"])

    # # Load model
    # if manager.args["load_saved_model"]:
    #     model_frame.load_model(manager.args["load_saved_model"])

    encoder: FeatureEncoder = model_frame.feature_encoder

    out_tensor, out_dataframe = encoder.extract_features(
        model_frame.data_loader_train, False, manager.args["model_save_dir"]
    )
    torch.save(
        out_tensor, os.path.join(manager.args["model_save_dir"], "clustered_means.pt")
    )

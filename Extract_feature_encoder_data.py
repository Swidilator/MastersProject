import os
import torch

from support_scripts.utils import ModelSettingsManager
from support_scripts.components import FeatureEncoder

if __name__ == "__main__":
    # Initialise settings manager to read args and set up environment
    manager: ModelSettingsManager = ModelSettingsManager()

    if manager.args["model"] == "CRN":
        from CRN import CRNFramework

        model_frame: CRNFramework = CRNFramework.from_model_settings_manager(manager)

    elif manager.args["model"] == "GAN":
        from GAN import GANFramework

        model_frame: GANFramework = GANFramework.from_model_settings_manager(manager)

    elif manager.args["model"] == "CRNVideo":
        from CRN import CRNVideoFramework

        model_frame: CRNVideoFramework = CRNVideoFramework.from_model_settings_manager(
            manager
        )

    else:
        raise SystemExit

    # Generate folder for run
    if not os.path.exists(manager.args["base_model_save_dir"]):
        os.makedirs(manager.args["base_model_save_dir"])

    if not os.path.exists(manager.args["model_save_dir"]):
        os.makedirs(manager.args["model_save_dir"])

    # Load model
    if manager.args["load_saved_model"]:
        model_frame.load_model(manager.args["load_saved_model"])

    encoder: FeatureEncoder = model_frame.feature_encoder

    out_tensor, out_dataframe = encoder.extract_features(
        model_frame.data_loader_train, True, manager.args["model_save_dir"]
    )
    torch.save(
        out_tensor, os.path.join(manager.args["model_save_dir"], "clustered_means.pt")
    )

import os
from typing import Optional
from PIL.Image import Image
from tqdm import tqdm
import random

from support_scripts.utils.ModelSettingsManager import ModelSettingsManager
from support_scripts.sampling.SamplingHelpers import (
    sample_from_model,
    create_image_directories,
)
from support_scripts.utils.MastersModel import MastersModel
from CRN import CRNFramework
from GAN import GANFramework

if __name__ == "__main__":

    manager: ModelSettingsManager = ModelSettingsManager()

    model_frame: Optional[MastersModel] = None

    if manager.model == "CRN":
        model_frame: CRNFramework = CRNFramework.from_model_settings_manager(manager)
        pass
    elif manager.model == "GAN":
        model_frame: GANFramework = GANFramework.from_model_settings_manager(manager)
    else:
        raise SystemExit

    # indices: tuple = random.sample(
    #     range(0, len(model_frame.__data_set_val__)), manager.args["sample"]
    # )
    indices: tuple = (32, 112, 130, 132, 175, 285, 377, 389, 447, 467)
    num_samples: int = len(indices) if (
        len(indices) and manager.args["sample_mode"] == "fixed"
    ) < manager.args["sample"] else manager.args["sample"]

    prefix: str = manager.args["model_save_prefix"]
    suffix: str = ".pt"

    model_list = os.listdir(manager.args["model_save_dir"])
    model_numbers = sorted(
        [
            int(x[len(prefix) : -len(suffix)])
            for x in model_list
            if prefix in x and suffix in x
        ]
    )
    print("Model numbers:", model_numbers)

    model_image_dir: str = create_image_directories(
        manager.args["image_output_dir"], manager.args["model"]
    )

    sample_args: dict = {}
    if manager.args["model"] == "GAN":
        sample_args.update({"use_extracted_features": True})

    for num in model_numbers:
        model_filename: str = os.path.join(prefix + str(num) + suffix)
        model_frame.load_model(manager.args["model_save_dir"], model_filename)

        if manager.args["model"] == "GAN":
            clustered_means, output_dataframe = model_frame.extract_feature_means()
            model_frame.__data_set_val__.set_clustered_means(clustered_means)

        # Sample image from dataset
        output_dicts, sample_list = sample_from_model(
            model=model_frame,
            sample_args=sample_args,
            mode=manager.args["sample_mode"],
            num_images=num_samples,
            indices=indices,
        )

        image: Image
        for i, img_dict in enumerate(tqdm(output_dicts, desc="Saving")):
            filename = os.path.join(
                model_image_dir,
                "figure_{_figure_}_epoch_{epoch}.png".format(
                    epoch=num, _figure_=sample_list[i]
                ),
            )
            img_dict["composite_img"].save(filename)

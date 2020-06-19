import os
from typing import Tuple, Any, Optional
from PIL.Image import Image
from tqdm import tqdm
from pickle import dump as p_dump
from pickle import dumps as p_dumps
from zstd import dumps as z_dumps

import wandb

from CRN.CRN_Framework import CRNFramework
from GAN import GANFramework
from support_scripts.sampling import sample_from_model
from support_scripts.utils import MastersModel
from support_scripts.utils import ModelSettingsManager

from support_scripts.utils.visualisation import eigenvector_visualisation

if __name__ == "__main__":
    # Initialise settings manager to read args and set up environment
    manager: ModelSettingsManager = ModelSettingsManager()

    model_frame: Optional[MastersModel] = None

    assert "train" in manager.args

    if manager.model == "CRN":
        model_frame: CRNFramework = CRNFramework.from_model_settings_manager(manager)
        # for rms in model_frame.crn.rms_list:
        #     # rms.conv_2.register_forward_hook(analyse_activations)
        #     rms.conv_1.register_forward_hook(eigenvector_visualisation)
        #     rms.layer_norm_1.register_forward_hook(eigenvector_visualisation)
        #     rms.conv_2.register_forward_hook(eigenvector_visualisation)
        #     if hasattr(rms, "layer_norm_2"):
        #         rms.layer_norm_2.register_forward_hook(eigenvector_visualisation)
        #     else:
        #         rms.final_conv.register_forward_hook(eigenvector_visualisation)
        # pass
    elif manager.model == "GAN":
        model_frame: GANFramework = GANFramework.from_model_settings_manager(manager)
    else:
        raise SystemExit

    # Generate folder for run
    full_save_path: str = os.path.join(
        manager.args["model_save_dir"], manager.args["run_folder_name"]
    )

    if manager.args["load_saved_model"]:
        model_frame.load_model(full_save_path, manager.args["load_saved_model"])

    if not manager.args["wandb"]:
        os.environ["WANDB_MODE"] = "dryrun"

    # Training
    if manager.args["train"]:
        wandb.init(
            project=manager.model.lower(),
            config={**manager.args, **manager.model_conf},
            name=manager.args["run_name"],
            group=manager.args["run_name"],
        )

        # Have WandB watch the components of the model
        for val in model_frame.wandb_trainable_model:
            wandb.watch(val)

        # Indices list for sampling
        final_chosen: list = [
            9,
            11,
            26,
            34,
            76,
            83,
            95,
            100,
            158,
            166,
            190,
            227,
            281,
            290,
            322,
        ]

        # Select a subset of final_chosen
        indices_list: tuple = final_chosen[
            0 : manager.args["sample"]
            if manager.args["sample"] <= len(final_chosen)
            else len(final_chosen)
        ]

        for current_epoch in range(
            manager.args["starting_epoch"], manager.args["train"] + 1
        ):
            print("Epoch:", current_epoch)

            # Decay learning rate
            if manager.model == "GAN" and manager.model_conf["GAN_DECAY_LEARNING_RATE"]:
                model_frame.adjust_learning_rate(
                    current_epoch,
                    manager.model_conf["GAN_DECAY_STARTING_EPOCH"],
                    manager.args["train"],
                    manager.model_conf["GAN_BASE_LEARNING_RATE"],
                )

            # Perform single epoch of training
            epoch_loss, _ = model_frame.train(current_epoch=current_epoch)

            # Sample images from the model
            if manager.args["sample"]:

                sample_args: dict = {}
                # if manager.args["model"] == "GAN":
                sample_args.update({"use_extracted_features": False})

                output_dicts, sample_list = sample_from_model(
                    model=model_frame,
                    sample_args=sample_args,
                    mode=manager.args["sample_mode"],
                    num_images=manager.args["sample"],
                    indices=indices_list,
                )

                wandb_img_list: list = []

                if not os.path.exists(manager.args["image_output_dir"]):
                    os.makedirs(manager.args["image_output_dir"])

                model_image_dir: str = os.path.join(
                    manager.args["image_output_dir"], manager.args["run_folder_name"]
                )
                if not os.path.exists(model_image_dir):
                    os.makedirs(model_image_dir)

                img_dict: dict
                for image_dict_index, img_dict in enumerate(
                    tqdm(output_dicts, desc="Saving / Uploading Images")
                ):
                    # Create base filename for saving images and pickle files
                    filename_no_extension = os.path.join(
                        model_image_dir,
                        "{model_name}_{run_name}_figure_{_figure_}_epoch_{epoch}".format(
                            model_name=manager.args["model"].replace(" ", "_"),
                            run_name=manager.args["run_name"].replace(" ", "_"),
                            epoch=current_epoch,
                            _figure_=sample_list[image_dict_index],
                        ),
                    )

                    # Save composite image for easy viewing
                    img_dict["composite_img"].save(filename_no_extension + ".png")

                    # Save dict with all images for advanced usage later on
                    with open(filename_no_extension + ".pickle", "wb") as pickle_file:
                        p_dump(z_dumps(p_dumps(img_dict)), pickle_file)

                    # Caption used for images on WandB
                    caption: str = str(
                        "Epoch: {epoch}, figure: {fig}".format(
                            epoch=current_epoch, fig=image_dict_index
                        )
                    )

                    # Append composite images to list of images to be uploaded on WandB
                    wandb_img_list.append(
                        wandb.Image(img_dict["composite_img"], caption=caption)
                    )

                # Log sample images to wandb, do not commit yet
                wandb.log({"Sample Images": wandb_img_list}, commit=False)

            # Commit epoch loss, and sample images if they exist.
            wandb.log(
                {
                    "Epoch Loss": epoch_loss,
                    "Epoch": current_epoch,
                }
            )

            # Delete output of training
            del epoch_loss, _

            # Save model if requested to save every epoch
            if (
                manager.args["save_every_num_epochs"] > 0
                and current_epoch % manager.args["save_every_num_epochs"] == 0
            ):
                print("Saving model")
                if not os.path.exists(manager.args["model_save_dir"]):
                    os.makedirs(manager.args["model_save_dir"])

                if not os.path.exists(full_save_path):
                    os.makedirs(full_save_path)
                model_frame.save_model(full_save_path, current_epoch)

        # If not saving every epoch, save model only once at the end
        if not manager.args["save_every_num_epochs"]:
            print("Saving model")
            if not os.path.exists(manager.args["model_save_dir"]):
                os.makedirs(manager.args["model_save_dir"])
            model_frame.save_model(manager.args["model_save_dir"])

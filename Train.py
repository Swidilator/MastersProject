import os
from pickle import dump as p_dump
from pickle import dumps as p_dumps
from typing import Optional

import wandb
from tqdm import tqdm
from zstd import dumps as z_dumps

from support_scripts.sampling import sample_from_model
from support_scripts.utils import MastersModel
from support_scripts.utils import ModelSettingsManager

# from support_scripts.utils.visualisation import eigenvector_visualisation
from support_scripts.utils import RunTimer

if __name__ == "__main__":
    # Initialise settings manager to read args and set up environment
    manager: ModelSettingsManager = ModelSettingsManager()

    if not manager.args["wandb"]:
        os.environ["WANDB_MODE"] = "dryrun"

    wandb.init(
        project=manager.args["model"].lower(),
        config={**manager.args, **manager.model_conf},
        name=manager.args["run_name"],
        group=manager.args["run_name"],
    )

    model_frame: Optional[MastersModel] = None

    assert "train" in manager.args

    run_timer: RunTimer = RunTimer(manager.args["max_run_hours"])

    if manager.args["model"] == "CRN":
        from CRN import CRNFramework

        model_frame: CRNFramework = CRNFramework.from_model_settings_manager(manager)

    elif manager.args["model"] == "GAN":
        from GAN import GANFramework

        model_frame: GANFramework = GANFramework.from_model_settings_manager(manager)

    elif manager.args["model"] == "CRNVideo":
        from CRN import CRNVideoFramework

        model_frame: CRNVideoFramework = CRNVideoFramework.from_model_settings_manager(manager)

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

    # Profiling
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    run_timer.reset_timer()

    # Have WandB watch the components of the model
    for val in model_frame.wandb_trainable_model:
        wandb.watch(val, log="all", log_freq=50)

    # Indices list for sampling
    # fmt: off
    sample_indices_list: list = [9, 11, 26, 34, 76, 83, 95, 100, 158, 166, 190, 227, 281, 290, 322]
    # fmt: on

    # Select a subset of sample_indices_list
    indices_list: tuple = sample_indices_list[
        0 : manager.args["sample"]
        if manager.args["sample"] <= len(sample_indices_list)
        else len(sample_indices_list)
    ]

    for current_epoch in range(
        manager.args["starting_epoch"], manager.args["train"] + 1
    ):
        print("Epoch:", current_epoch)

        save_this_epoch = (
            manager.args["save_every_num_epochs"] > 0
            and current_epoch % manager.args["save_every_num_epochs"] == 0
        ) or (current_epoch == manager.args["train"])

        # Decay learning rate
        if (
            manager.args["model"] == "GAN"
            and manager.model_conf["GAN_DECAY_LEARNING_RATE"]
        ):
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

            output_dicts, sample_list = sample_from_model(
                model=model_frame,
                sample_args=sample_args,
                mode=manager.args["sample_mode"],
                num_images=manager.args["sample"],
                indices=indices_list,
            )

            wandb_img_list: list = []

            if not os.path.exists(manager.args["base_image_save_dir"]):
                os.makedirs(manager.args["base_image_save_dir"])

            if not os.path.exists(manager.args["image_save_dir"]):
                os.makedirs(manager.args["image_save_dir"])

            img_dict: dict
            for image_dict_index, img_dict in enumerate(
                tqdm(output_dicts, desc="Saving / Uploading Images")
            ):
                # Create base filename for saving images and pickle files
                filename_no_extension = os.path.join(
                    manager.args["image_save_dir"],
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
            {"Epoch Loss": epoch_loss, "Epoch": current_epoch,}
        )

        # Delete output of training
        del epoch_loss, _

        if not run_timer.update_and_predict_interval_security():
            print("Stopping due to run timer.")
            print("Saving model: {}".format(current_epoch))
            model_frame.save_model(current_epoch)
            break

        # Save model
        if save_this_epoch:
            print("Saving model: {}".format(current_epoch))
            model_frame.save_model(current_epoch)

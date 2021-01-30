import os
import sys
from pickle import dump as p_dump
from pickle import dumps as p_dumps
from typing import Optional

import wandb
from tqdm import tqdm
from zstd import dumps as z_dumps

from support_scripts.utils import MastersModel
from Video_Framework import VideoFramework

from support_scripts.sampling import sample_from_model, SampleDataHolder
from support_scripts.utils import ModelSettingsManager

from support_scripts.utils import RunTimer

if __name__ == "__main__":
    # Initialise settings manager to read args and set up environment
    manager: ModelSettingsManager = ModelSettingsManager()

    # Set wandb to run in dryrun mode if requested
    if not manager.args["wandb"]:
        os.environ["WANDB_MODE"] = "dryrun"

    # Initialise wandb
    wandb.init(
        project=manager.args["model"].lower(),
        config={**manager.args, **manager.model_conf},
        name=manager.args["run_name"],
        group=manager.args["run_name"],
    )

    assert "train" in manager.args

    # Initialise run timer
    run_timer: RunTimer = RunTimer(manager.args["max_run_hours"])

    model_frame: VideoFramework = VideoFramework.from_model_settings_manager(manager)

    # Generate folder for run
    os.makedirs(manager.args["model_save_dir"], exist_ok=True)

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
    sample_indices_list: list = [int(x * 3) for x in range(len(model_frame.dataset_val) // 3)]
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
        sys.stdout.flush()
        sys.stderr.flush()

        save_this_epoch = (
            manager.args["save_every_num_epochs"] > 0
            and current_epoch % manager.args["save_every_num_epochs"] == 0
        ) or (current_epoch == manager.args["train"])

        # Decay learning rate
        if manager.model_conf["DECAY_LEARNING_RATE"]:
            model_frame.adjust_learning_rate(
                current_epoch,
                manager.model_conf["DECAY_STARTING_EPOCH"],
                manager.args["train"],
                manager.model_conf["BASE_LEARNING_RATE"],
            )

        # Perform single epoch of training
        epoch_loss, _ = model_frame.train(current_epoch=current_epoch)

        # Sample images from the model
        if manager.args["sample"] and (manager.args["sample_every_epoch"] or save_this_epoch):
            image_data_holders, sample_list = sample_from_model(
                model=model_frame,
                mode=manager.args["sample_mode"],
                num_images=manager.args["sample"],
                indices=indices_list,
            )

            wandb_img_list: list = []
            wandb_video_list: list = []

            # Create directory if it does not already exist.
            os.makedirs(manager.args["image_save_dir"], exist_ok=True)

            image_data_holder: SampleDataHolder
            for image_dict_index, image_data_holder in enumerate(
                tqdm(image_data_holders, desc="Saving / Uploading Images")
            ):
                # Create base filename for saving images and pickle files
                filename_no_extension = os.path.join(
                    manager.args["image_save_dir"],
                    "{model_name}_{run_name}_figure_{figure}_epoch_{epoch}".format(
                        model_name=manager.args["model"].replace(" ", "_"),
                        run_name=manager.args["run_name"].replace(" ", "_"),
                        epoch=current_epoch,
                        figure=sample_list[image_dict_index],
                    ),
                )

                # Save composite image for easy viewing
                image_data_holder.composite_image[0].save(
                    filename_no_extension + ".png"
                )

                # Save dict with all images for advanced usage later on
                with open(filename_no_extension + ".pickle", "wb") as pickle_file:
                    p_dump(z_dumps(p_dumps(image_data_holder)), pickle_file)

                # Caption used for images on WandB
                caption: str = str(
                    "Epoch: {epoch}, figure: {fig}".format(
                        epoch=current_epoch, fig=image_dict_index
                    )
                )
                if not image_data_holder.video_sample:
                    # Append composite images to list of images to be uploaded on WandB
                    wandb_img_list.append(
                        wandb.Image(
                            image_data_holder.composite_image[0], caption=caption
                        )
                    )
                else:
                    # Append composite images to list of images to be uploaded on WandB
                    save_stdout = sys.stdout
                    save_stderr = sys.stderr
                    sys.stdout = open("trash", "w")
                    sys.stderr = sys.stdout
                    wandb_video_list.append(
                        wandb.Video(image_data_holder.final_gif, caption=caption, fps=1)
                    )
                    sys.stdout = save_stdout
                    sys.stderr = save_stderr

            # Log sample images to wandb, do not commit yet
            if len(wandb_img_list) > 0:
                wandb.log({"Sample Images": wandb_img_list}, commit=False)
            else:
                wandb.log({"Sample Videos": wandb_video_list}, commit=False)

        # Commit epoch loss, and sample images if they exist.
        wandb.log(
            {
                "Epoch Loss": epoch_loss,
                "Epoch": current_epoch,
            }
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

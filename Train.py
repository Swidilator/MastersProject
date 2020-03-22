import torch
import os
from typing import Tuple, List, Any
import argparse
import json

from CRN.CRN_Framework import CRNFramework
from GAN import GANFramework
from support_scripts.sampling import sample_from_model
from support_scripts.utils import MastersModel
from support_scripts.utils import ModelSettingsManager

if __name__ == "__main__":
    # Initialise settings manager to read args and set up environment
    manager: ModelSettingsManager = ModelSettingsManager()

    model_frame: Optional[MastersModel] = None

    assert "train" in manager.args

    if manager.model == "CRN":
        model_frame: CRNFramework = CRNFramework.from_model_settings_manager(manager)
        pass
    elif manager.model == "GAN":
        model_frame: GANFramework = GANFramework.from_model_settings_manager(manager)
    else:
        raise SystemExit

    if manager.args["load_saved_model"]:
        if manager.args["load_saved_model"] == "Latest":
            model_frame.load_model(manager.args["model_save_dir"])
        else:
            model_frame.load_model(
                manager.args["model_save_dir"], manager.args["load_saved_model"]
            )

    if not manager.args["wandb"]:
        os.environ["WANDB_MODE"] = "dryrun"

    # Training
    if manager.args["train"]:
        wandb.init(
            project=manager.model.lower(), config={**manager.args, **manager.model_conf}
        )

        # Have WandB watch the components of the model
        for val in model_frame.wandb_trainable_model:
            wandb.watch(val)

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

                indices_list: tuple = tuple([x for x in range(manager.args["sample"])])

                img_list: List[Tuple[Any, Any]] = model_frame.sample(
                    args["sample"], **sample_args
                )
                wandb_img_list: list = []

                if not os.path.exists(manager.args["image_output_dir"]):
                    os.makedirs(manager.args["image_output_dir"])

                model_image_dir: str = os.path.join(
                    manager.args["image_output_dir"], manager.args["model"]
                )
                if not os.path.exists(model_image_dir):
                    os.makedirs(model_image_dir)

                img_pair: Tuple[Any, Any]
                for j, img_pair in enumerate(img_list):
                    filename = os.path.join(
                        manager.args["image_output_dir"],
                        "epoch_{epoch}_figure_{_figure_}.png".format(
                            epoch=current_epoch, _figure_=j
                        ),
                    )

                    output_image: Image = process_sampled_image(img_pair, filename)

                    caption: str = str(
                        "Epoch: {epoch}, figure: {fig}".format(
                            epoch=current_epoch, fig=j
                        )
                    )
                    wandb_img_list.append(wandb.Image(output_image, caption=caption))

                # Log sample images to wandb, do not commit yet
                wandb.log({"Sample Images": wandb_img_list}, commit=False)

            # Commit epoch loss, and sample images if they exist.
            wandb.log(
                {
                    "Epoch Loss": epoch_loss * manager.args["batch_size_pair"][1],
                    "Epoch": current_epoch,
                }
            )

            # Delete output of training
            del epoch_loss, _

            # Save model if requested to save every epoch
            if manager.args["save_every_epoch"]:
                if not os.path.exists(manager.args["model_save_dir"]):
                    os.makedirs(manager.args["model_save_dir"])
                model_frame.save_model(manager.args["model_save_dir"], current_epoch)

        # If not saving every epoch, save model only once at the end
        if not manager.args["save_every_epoch"]:
            if not os.path.exists(manager.args["model_save_dir"]):
                os.makedirs(manager.args["model_save_dir"])
            model_frame.save_model(manager.args["model_save_dir"])

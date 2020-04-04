import os
from typing import Tuple, Any, Optional
from PIL.Image import Image
from tqdm import tqdm

import wandb

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

    # Generate folder for run
    model_folder_name: str = "{model_name}_{run_name}".format(
        model_name=manager.args["model"],
        run_name=manager.args["run_name"].replace(" ", "_"),
    )
    full_save_path: str = os.path.join(
        manager.args["model_save_dir"], model_folder_name
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
            resume=(True if manager.args["starting_epoch"] > 1 else False),
        )

        # Have WandB watch the components of the model
        for val in model_frame.wandb_trainable_model:
            wandb.watch(val)

        # Indices list for sampling
        indices_list: tuple = tuple(range(manager.args["sample"]))

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
                if manager.args["model"] == "GAN":
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

                save_folder_name: str = "{model_name}_{run_name}".format(
                    model_name=manager.args["model"],
                    run_name=manager.args["run_name"].replace(" ", "_"),
                )
                model_image_dir: str = os.path.join(
                    manager.args["image_output_dir"], save_folder_name
                )
                if not os.path.exists(model_image_dir):
                    os.makedirs(model_image_dir)

                img_dict: dict
                for j, img_dict in enumerate(
                    tqdm(output_dicts, desc="Saving / Uploading Images")
                ):
                    filename = os.path.join(
                        model_image_dir,
                        "figure_{_figure_}_epoch_{epoch}.png".format(
                            epoch=current_epoch, _figure_=sample_list[j]
                        ),
                    )
                    img_dict["composite_img"].save(filename)
                    caption: str = str(
                        "Epoch: {epoch}, figure: {fig}".format(
                            epoch=current_epoch, fig=j
                        )
                    )
                    wandb_img_list.append(
                        wandb.Image(img_dict["composite_img"], caption=caption)
                    )

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

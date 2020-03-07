import torch
import os
from typing import Tuple, List, Any
import argparse
import json
from ast import literal_eval
from matplotlib import pyplot as plt

from CRN.CRN_Framework import CRNFramework
from GAN.GAN_Framework import GANFramework
from typing import Optional
from Training_Framework import MastersModel
import wandb
from PIL import Image

# from torchviz import make_dot

# When sampling, this combines images for saving as side-by-side comparisons
def process_sampled_image(img_pair: tuple, output_path: str) -> Image:
    img_width: int = img_pair[0].size[0]
    img_height: int = img_pair[0].size[1]

    new_image: Image = Image.new("RGB", (img_width * 2, img_height))
    new_image.paste(img_pair[0], (0, 0))
    new_image.paste(img_pair[1], (img_width, 0))
    new_image.save(output_path)
    return new_image


if __name__ == "__main__":
    # Parse settings
    parser = argparse.ArgumentParser(description="Masters model main file")

    parser.add_argument("model", action="store")
    parser.add_argument("dataset", action="store")
    parser.add_argument("dataset_path", action="store")
    parser.add_argument("input_image_height_width", action="store", type=eval)
    parser.add_argument("batch_size_pair", action="store", type=eval)
    parser.add_argument("training_machine_name", action="store")

    parser.add_argument("--model-conf-file", action="store", default="model_conf.json")
    parser.add_argument("--train", action="store", default=0, type=int)
    parser.add_argument("--starting-epoch", action="store", default=1, type=int)
    parser.add_argument("--sample", action="store", default=0, type=int)
    parser.add_argument("--training-subset", action="store", default=0, type=int)
    parser.add_argument("--wandb", action="store_true", default=False)
    parser.add_argument("--model-save-dir", action="store", default="./Models/")
    parser.add_argument("--image-output-dir", action="store", default="./Images/")
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--gpu-no", action="store", default=0)
    parser.add_argument("--save-every-epoch", action="store_true", default=False)
    parser.add_argument("--load-saved-model", action="store", default=None)
    parser.add_argument("--no-tanh", action="store_true", default=False)
    parser.add_argument("--num-workers", action="store", default=6, type=int)
    parser.add_argument("--num-classes", action="store", default=20, type=int)
    parser.add_argument("--input-image-noise", action="store_true", default=False)
    parser.add_argument("--flip-training-images", action="store_true", default=False)
    parser.add_argument("--deterministic", action="store_true", default=False)

    args: dict = vars(parser.parse_args())

    # Determinism
    if args["deterministic"]:
        torch.manual_seed(0)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Import model configuration file
    with open(args["model_conf_file"], "r") as model_conf_file:
        model_conf: dict = json.load(model_conf_file)

    # Print configuration options
    for arg, val in {**args, **model_conf[args["model"]]}.items():
        print("{arg}: {val}".format(arg=arg, val=val))

    if not args["cpu"]:
        if torch.cuda.is_available():
            device = torch.device("cuda:{gpu}".format(gpu=args["gpu_no"]))
            print("Device: CUDA")
        else:
            device = torch.device("cpu")
            print("Device: CPU")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    model_frame: Optional[MastersModel] = None

    model_frame_args: dict = {
        "device": device,
        "data_path": args["dataset_path"],
        "input_image_height_width": args["input_image_height_width"],
        "batch_size_slice": args["batch_size_pair"][0],
        "batch_size_total": args["batch_size_pair"][1],
        "num_classes": args["num_classes"],
        "num_loader_workers": args["num_workers"],
        "subset_size": args["training_subset"],
        "should_flip_train": args["flip_training_images"],
        "use_tanh": not args["no_tanh"],
        "use_input_noise": args["input_image_noise"],
    }

    if args["model"] == "CRN":
        settings = {
            "input_tensor_size": (
                model_conf["CRN"]["CRN_INPUT_TENSOR_SIZE_HEIGHT"],
                model_conf["CRN"]["CRN_INPUT_TENSOR_SIZE_WIDTH"],
            ),
            "num_output_images": model_conf["CRN"]["CRN_NUM_OUTPUT_IMAGES"],
            "num_inner_channels": model_conf["CRN"]["CRN_NUM_INNER_CHANNELS"],
            "history_len": model_conf["CRN"]["CRN_HISTORY_LEN"],
        }
        model_frame: CRNFramework = CRNFramework(**model_frame_args, **settings)

    elif args["model"] == "GAN":
        settings = {
            "base_learning_rate": model_conf["GAN"]["GAN_BASE_LEARNING_RATE"],
            "num_discriminators": model_conf["GAN"]["GAN_NUM_DISCRIMINATORS"],
            "use_local_enhancer": model_conf["GAN"]["GAN_USE_LOCAL_ENHANCER"],
            "use_noisy_labels": model_conf["GAN"]["GAN_USE_NOISY_LABELS"],
            "feature_matching_weight": model_conf["GAN"]["GAN_FEATURE_MATCHING_WEIGHT"],
            "feature_extractions_file_path": model_conf["GAN"][
                "GAN_FEATURE_EXTRACTIONS_FILE_PATH"
            ],
        }
        model_frame: GANFramework = GANFramework(**model_frame_args, **settings)
    else:
        raise SystemExit

    if args["load_saved_model"]:
        if args["load_saved_model"] == "Latest":
            model_frame.load_model(args["model_save_dir"])
        else:
            model_frame.load_model(args["model_save_dir"], args["load_saved_model"])

    if not args["wandb"]:
        os.environ["WANDB_MODE"] = "dryrun"

    # Training
    if args["train"]:
        wandb.init(
            project=args["model"].lower(), config={**args, **model_conf[args["model"]]}
        )

        # Have WandB watch the components of the model
        for val in model_frame.wandb_trainable_model:
            wandb.watch(val)

        for current_epoch in range(args["starting_epoch"], args["train"] + 1):
            print("Epoch:", current_epoch)

            # Decay learning rate
            if args["model"] == "GAN" and model_conf["GAN"]["GAN_DECAY_LEARNING_RATE"]:
                model_frame.adjust_learning_rate(
                    current_epoch,
                    model_conf["GAN"]["GAN_DECAY_STARTING_EPOCH"],
                    args["train"],
                    model_conf["GAN"]["GAN_BASE_LEARNING_RATE"],
                )

            # Perform single epoch of training
            epoch_loss, _ = model_frame.train(current_epoch=current_epoch)

            # Sample images from the model
            if args["sample"]:
                img_list: List[Tuple[Any, Any]] = model_frame.sample(args["sample"])
                wandb_img_list: list = []

                if not os.path.exists(args["image_output_dir"]):
                    os.makedirs(args["image_output_dir"])

                model_image_dir: str = os.path.join(
                    args["image_output_dir"], args["model"]
                )
                if not os.path.exists(model_image_dir):
                    os.makedirs(model_image_dir)

                img_pair: Tuple[Any, Any]
                for j, img_pair in enumerate(img_list):
                    filename = os.path.join(
                        args["image_output_dir"],
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
                    "Epoch Loss": epoch_loss * args["batch_size_pair"][1],
                    "Epoch": current_epoch,
                }
            )

            # Delete output of training
            del epoch_loss, _

            # Save model if requested to save every epoch
            if args["save_every_epoch"]:
                if not os.path.exists(args["model_save_dir"]):
                    os.makedirs(args["model_save_dir"])
                model_frame.save_model(args["model_save_dir"], current_epoch)

        # If not saving every epoch, save model only once at the end
        if not args["save_every_epoch"]:
            if not os.path.exists(args["model_save_dir"]):
                os.makedirs(args["model_save_dir"])
            model_frame.save_model(args["model_save_dir"])

    # Sample images if not training
    if args["sample"] and not args["train"]:
        if not os.path.exists(args["image_output_dir"]):
            os.makedirs(args["image_output_dir"])
        # model_frame.load_model(MODEL_PATH)
        img_list: list = model_frame.sample(args["sample"])
        for i, img_pair in enumerate(img_list):
            filename = os.path.join(
                args["image_output_dir"], "Sample_figure_{i}.png".format(i=i)
            )
            process_sampled_image(img_pair, filename)

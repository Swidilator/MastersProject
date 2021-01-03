import argparse
import json
from os import path

import torch


class ModelSettingsManager:
    def __init__(self):

        # Process generic input arguments
        self.args: dict = self.__process_args__()
        self.args["device"] = self.__set_torch_device__()

        # Process model specific configuration
        self.model_conf: dict = self.__process_model_conf__()

        # Set torch / nvidia determinism based on args
        self.__set_determinism__()

        for arg, val in {**self.args, **self.model_conf}.items():
            print("{arg}: {val}".format(arg=arg, val=val))

    def __process_args__(self) -> dict:
        # Parse settings
        # fmt: off
        parser = argparse.ArgumentParser(description="Masters model main file")

        parser.add_argument("model", action="store")
        parser.add_argument("dataset_path", action="store")
        parser.add_argument("input_image_height_width", action="store", type=eval)
        parser.add_argument("batch_size", action="store", type=int)
        parser.add_argument("training_machine_name", action="store")
        parser.add_argument("run_name", action="store")

        parser.add_argument("--model-conf-file", action="store", default="model_conf.json")
        parser.add_argument("--wandb", action="store_true", default=False)
        parser.add_argument("--train", action="store", default=0, type=int)
        parser.add_argument("--starting-epoch", action="store", default=1, type=int)
        parser.add_argument("--sample", action="store", default=0, type=int)
        parser.add_argument("--sample-mode", action="store", default="random")
        parser.add_argument("--sample-only", action="store_true", default=False)
        parser.add_argument("--training-subset-size", action="store", default=0, type=int)
        parser.add_argument("--base-model-save-dir", action="store", default="./Models/")
        parser.add_argument("--model-save-prefix", action="store", default="Epoch_")
        parser.add_argument("--base-image-save-dir", action="store", default="./Images/")
        parser.add_argument("--cpu", action="store_true", default=False)
        parser.add_argument("--gpu-no", action="store", default=0)
        parser.add_argument("--save-every-num-epochs", action="store", default=0, type=int)
        parser.add_argument("--load-saved-model", action="store", default=None)
        parser.add_argument("--log-every-n-steps", action="store", default=8)
        parser.add_argument("--use-amp", action="store", default=False)
        parser.add_argument("--num-data-workers", action="store", default=6, type=int)
        parser.add_argument("--flip-training-images", action="store_true", default=False)
        parser.add_argument("--deterministic", action="store_true", default=False)
        parser.add_argument("--max-run-hours", action="store", default=0.0, type=float)
        parser.add_argument("--num-frames-per-training-video", action="store", default=1, type=int)
        parser.add_argument("--num-frames-per-sampling-video", action="store", default=16, type=int)
        parser.add_argument("--prior-frame-seed-type", action="store", default="zero", type=str)
        parser.add_argument("--video-frame-offset", action="store", default="random", type=str)
        parser.add_argument("--use-mask-for-instances", action="store_true", default=False)
        parser.add_argument("--use-saved-feature-encodings", action="store_true", default=False)

        args: dict = vars(parser.parse_args())

        args.update(self.__clean_arg_path__("dataset_path", args))
        args.update(self.__clean_arg_path__("model_conf_file", args))
        args.update(self.__clean_arg_path__("base_model_save_dir", args))
        args.update(self.__clean_arg_path__("base_image_save_dir", args))

        # Add folder names for ease of access and standardisation
        args.update(
            {
                "model_save_dir": path.join(
                    args["base_model_save_dir"],
                    "{model}_{run_name}".format(
                        model=args["model"], run_name=args["run_name"].replace(" ", "_")
                    ),
                )
            }
        )
        args.update(
            {
                "image_save_dir": path.join(
                    args["base_image_save_dir"],
                    "{model}_{run_name}".format(
                        model=args["model"], run_name=args["run_name"].replace(" ", "_")
                    ),
                )
            }
        )
        args.update(
            {
                "flownet_save_path": path.join(
                    args["base_model_save_dir"],
                    "FlowNet2_checkpoints/FlowNet2_checkpoint.pth.tar"
                )
            }
        )

        if (args["sample_mode"] != "random") and (args["sample_mode"] != "fixed"):
            raise ValueError("--sample-mode should be 'random' or 'fixed'.")

        # fmt: on
        return args

    def __process_model_conf__(self) -> dict:
        with open("model_conf_default.json", "r") as model_conf_default_file:
            model_conf: dict = json.load(model_conf_default_file)
        with open(self.args["model_conf_file"], "r") as model_conf_file:
            model_conf.update(json.load(model_conf_file))

        return model_conf

    def __clean_arg_path__(self, key: str, args: dict) -> dict:
        return {key: path.abspath(args[key])}

    def __set_determinism__(self) -> None:
        if self.args["deterministic"]:
            torch.manual_seed(0)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def __set_torch_device__(self) -> torch.device:
        device: torch.device = torch.device("cpu")

        if not self.args["cpu"] and torch.cuda.is_available():
            device = torch.device("cuda:{gpu}".format(gpu=self.args["gpu_no"]))

        print("Device: {device}".format(device=device))
        return device

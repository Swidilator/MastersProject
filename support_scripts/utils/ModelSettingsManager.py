import argparse
import json
import torch
from os import path


class ModelSettingsManager:
    def __init__(self):

        # Process generic input arguments
        self.args: dict = self.__process_args__()
        self.model: str = self.args["model"]

        # Process model specific configuration
        self.model_conf: dict = self.__process_model_conf__()

        # Set torch / nvidia determinism based on args
        self.__set_determinism__()

        # Set torch device
        self.device: torch.device = self.__set_torch_device__()

        for arg, val in {**self.args, **self.model_conf}.items():
            print("{arg}: {val}".format(arg=arg, val=val))

    def __process_args__(self) -> dict:
        # Parse settings
        parser = argparse.ArgumentParser(description="Masters model main file")

        parser.add_argument("model", action="store")
        parser.add_argument("dataset", action="store")
        parser.add_argument("dataset_path", action="store")
        parser.add_argument("input_image_height_width", action="store", type=eval)
        parser.add_argument("batch_size_pair", action="store", type=eval)
        parser.add_argument("training_machine_name", action="store")
        parser.add_argument("run_name", action="store")

        parser.add_argument(
            "--model-conf-file", action="store", default="model_conf.json"
        )
        parser.add_argument("--train", action="store", default=0, type=int)
        parser.add_argument("--starting-epoch", action="store", default=1, type=int)
        parser.add_argument("--sample", action="store", default=0, type=int)
        parser.add_argument(
            "--sample-type", action="store", default="from_tuple", type=str
        )
        parser.add_argument("--training-subset", action="store", default=0, type=int)
        parser.add_argument("--wandb", action="store_true", default=False)
        parser.add_argument("--model-save-dir", action="store", default="./Models/")
        parser.add_argument("--model-save-prefix", action="store", default="Epoch_")
        parser.add_argument("--image-output-dir", action="store", default="./Images/")
        parser.add_argument("--cpu", action="store_true", default=False)
        parser.add_argument("--gpu-no", action="store", default=0)
        parser.add_argument("--save-every-num-epochs", action="store", default=0, type=int)
        parser.add_argument("--load-saved-model", action="store", default=None)
        parser.add_argument("--no-tanh", action="store_true", default=False)
        parser.add_argument("--num-workers", action="store", default=6, type=int)
        parser.add_argument("--num-classes", action="store", default=20, type=int)
        parser.add_argument("--input-image-noise", action="store_true", default=False)
        parser.add_argument(
            "--flip-training-images", action="store_true", default=False
        )
        parser.add_argument("--deterministic", action="store_true", default=False)
        parser.add_argument("--sample-mode", action="store", default="random")
        parser.add_argument("--sample-only", action="store_true", default=False)

        args: dict = vars(parser.parse_args())

        args.update(self.__clean_arg_path__("dataset_path", args))
        args.update(self.__clean_arg_path__("model_conf_file", args))
        args.update(self.__clean_arg_path__("model_save_dir", args))
        args.update(self.__clean_arg_path__("image_output_dir", args))

        if (args["sample_mode"] != "random") and (args["sample_mode"] != "fixed"):
            raise ValueError("--sample-mode should be 'random' or 'fixed'.")

        return args

    def __process_model_conf__(self) -> dict:
        with open(self.args["model_conf_file"], "r") as model_conf_file:
            model_conf: dict = json.load(model_conf_file)

        return model_conf[self.model]

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
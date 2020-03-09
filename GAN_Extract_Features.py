import torch
import numpy as np
import pandas as pd
import kmeans_pytorch
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import pickle

import os

import argparse
import json
from ast import literal_eval

from GAN.GAN_Framework import GANFramework

if __name__ == "__main__":
    # Parse settings
    parser = argparse.ArgumentParser(description="Masters model main file")

    parser.add_argument("dataset", action="store")
    parser.add_argument("dataset_path", action="store")
    parser.add_argument("input_image_height_width", action="store", type=eval)
    parser.add_argument("batch_size_pair", action="store", type=eval)

    parser.add_argument("--training-subset", action="store", default=0, type=int)
    parser.add_argument("--model-save-dir", action="store", default="./Models/")
    parser.add_argument("--model-file-name", action="store", default="GAN_Latest.pt")
    parser.add_argument(
        "--features-file-path", action="store", default="./extracted_features.csv"
    )
    parser.add_argument("--use-existing-data", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--gpu-no", action="store", default=0)
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
    with open("model_conf.json", "r") as model_conf_file:
        model_conf: dict = json.load(model_conf_file)

    # Print configuration options
    for arg, val in {**args, **model_conf["GAN"]}.items():
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

    settings = {
        "base_learning_rate": model_conf["GAN"]["GAN_BASE_LEARNING_RATE"],
        "num_discriminators": model_conf["GAN"]["GAN_NUM_DISCRIMINATORS"],
        "use_local_enhancer": model_conf["GAN"]["GAN_USE_LOCAL_ENHANCER"],
        "use_noisy_labels": model_conf["GAN"]["GAN_USE_NOISY_LABELS"],
        "feature_matching_weight": model_conf["GAN"]["GAN_FEATURE_MATCHING_WEIGHT"],
        "feature_extractions_file_path": model_conf["GAN"][
            "GAN_FEATURE_EXTRACTIONS_FILE_PATH"
        ],
        "use_sigmoid_discriminator": model_conf["GAN"][
            "GAN_USE_SIGMOID_DISCRIMINATOR"
        ],
    }

    if not args["use_existing_data"]:
        model_frame: GANFramework = GANFramework(**model_frame_args, **settings)

        # Load model
        model_frame.load_model(args["model_save_dir"], args["model_file_name"])
        # Extract features
        features: pd.DataFrame = model_frame.extract_features()

        # Save features to file
        features.to_csv(args["features_file_path"], index=False)
        del model_frame
    else:
        features: pd.DataFrame = pd.read_csv(args["features_file_path"])
        features = features.round({"Class": 0})

    clustered_means: torch.Tensor = torch.empty(0, 4)

    for i, val in enumerate(features["Class"].unique().tolist()):
        subset: pd.DataFrame = features[features["Class"] == val]
        subset = subset[["Value_1", "Value_2", "Value_3"]]
        np_subset: np.ndarray = subset.to_numpy()
        t_subset: torch.Tensor = torch.tensor(np_subset).to(device)

        cluster_ids_x, cluster_centers = kmeans_pytorch.kmeans(
            X=t_subset, num_clusters=10, distance="euclidean", device=device
        )

        formatted_means: torch.Tensor = torch.empty(10, 4).float()
        formatted_means[:, 0] = val
        formatted_means[:, 1:4] = cluster_centers

        clustered_means = torch.cat((clustered_means, formatted_means))

    # Dump to pickle to preserve tensor data type
    with open("cluster_centres_tensor.pkl", "wb") as f:
        pickle.dump(clustered_means.cpu(), f)

    np_clustered_means: np.ndarray = clustered_means.cpu().numpy()

    output_dataframe = pd.DataFrame(
        data=np.float_(np_clustered_means),
        index=np.arange(0, np_clustered_means.shape[0]),
        columns=["Class", "Mean_1", "Mean_2", "Mean_3"]
    )
    output_dataframe.to_csv("cluster_centres.csv", index=False)

    raise SystemExit

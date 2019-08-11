import torch
import os
from matplotlib import pyplot as plt

from CRN import CRNFramework
from Helper_Stuff import *
import wandb

# from torchviz import make_dot

if __name__ == "__main__":
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    MAX_INPUT_HEIGHT_WIDTH: tuple = (128, 256)
    INPUT_TENSOR_SIZE: tuple = (4, 8)
    NUM_OUTPUT_IMAGES: int = 1
    NUM_CLASSES: int = 35
    NUM_INNER_CHANNELS = 128
    BATCH_SIZE: int = 4
    HISTORY_LEN: int = 20

    PREFER_CUDA: bool = True
    NUM_LOADER_WORKERS: int = 2
    MODEL_PATH: str = "./Models/"

    CITYSCAPES_PATH: str = os.environ["CITYSCAPES_PATH"]
    print("Dataset path: {cityscapes}".format(cityscapes=CITYSCAPES_PATH))

    TRAINING_MACHINE: str = os.environ["TRAINING_MACHINE"]
    print("Training Machine: {machine}".format(machine=TRAINING_MACHINE))

    DATA_PATH: str = CITYSCAPES_PATH

    if PREFER_CUDA:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("Device: CUDA")
        else:
            device = torch.device("cpu")
            print("Device: CPU")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    crn_frame: CRNFramework = CRNFramework(
        device=device,
        data_path=DATA_PATH,
        input_tensor_size=INPUT_TENSOR_SIZE,
        max_input_height_width=MAX_INPUT_HEIGHT_WIDTH,
        num_output_images=NUM_OUTPUT_IMAGES,
        num_inner_channels=NUM_INNER_CHANNELS,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        num_loader_workers=NUM_LOADER_WORKERS,
        history_len=HISTORY_LEN,
    )

    TRAIN: tuple = (True, 100)
    SAMPLE: tuple = (False, 20)
    WANDB: bool = False
    SAVE_EVERY_EPOCH: bool = False
    LOAD_BEFORE_TRAIN: bool = True
    UPDATE_PL_LAMBDAS: bool = True

    # Training
    if TRAIN[0]:
        if WANDB:
            wandb.init(
                project="crn",
                config={
                    "Batch Size": BATCH_SIZE,
                    "Inner Channels": NUM_INNER_CHANNELS,
                    "Output Size": MAX_INPUT_HEIGHT_WIDTH,
                    "Training Machine": TRAINING_MACHINE,
                },
            )

        if LOAD_BEFORE_TRAIN:
            crn_frame.load_model(MODEL_PATH)

        # Watch if set
        no_except(wandb.watch, crn_frame.crn)

        for i in range(TRAIN[1]):
            loss, _ = crn_frame.train(UPDATE_PL_LAMBDAS)
            no_except(wandb.log, {"Epoch Loss": loss * BATCH_SIZE, "Epoch": i})
            print(i, loss, crn_frame.loss_net.loss_layer_scales)
            if SAVE_EVERY_EPOCH:
                crn_frame.save_model(MODEL_PATH)
        if not SAVE_EVERY_EPOCH:
            crn_frame.save_model(MODEL_PATH)
        quit()

    # Sampling
    if SAMPLE[0]:
        crn_frame.load_model(MODEL_PATH)
        img_list: sample_output = crn_frame.sample(SAMPLE[1])
        for i, img in enumerate(img_list):
            print(img_list[i])
            plt.figure(i)
            plt.imshow(img)
            plt.show()

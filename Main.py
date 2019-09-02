import torch
import os
from matplotlib import pyplot as plt

from CRN.CRN import CRNFramework
from GAN.GAN import GANFramework
from Helper_Stuff import *
import wandb

# from torchviz import make_dot

if __name__ == "__main__":
    # Determinism
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # General Model Settings
    MAX_INPUT_HEIGHT_WIDTH: tuple = (128, 256)
    NUM_CLASSES: int = 35
    NUM_INNER_CHANNELS = 512
    BATCH_SIZE: int = 2
    HISTORY_LEN: int = 100

    # CRN
    INPUT_TENSOR_SIZE: tuple = (4, 8)
    NUM_OUTPUT_IMAGES: int = 1

    # System Settings
    PREFER_CUDA: bool = True
    NUM_LOADER_WORKERS: int = 6
    MODEL_PATH: str = "./Models/"
    CITYSCAPES_PATH: str = os.environ["CITYSCAPES_PATH"]
    DATA_PATH: str = CITYSCAPES_PATH
    TRAINING_MACHINE: str = os.environ["TRAINING_MACHINE"]
    print("Dataset path: {cityscapes}".format(cityscapes=CITYSCAPES_PATH))
    print("Training Machine: {machine}".format(machine=TRAINING_MACHINE))

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

    # Run Specific Settings
    TRAIN: tuple = (False, 100)
    SAMPLE: tuple = (True, 1)
    WANDB: bool = False
    SAVE_EVERY_EPOCH: bool = True
    LOAD_BEFORE_TRAIN: bool = False
    # CRN
    UPDATE_PL_LAMBDAS: bool = True

    # Choose Model

    # model_frame: CRNFramework = CRNFramework(
    #     device=device,
    #     data_path=DATA_PATH,
    #     input_tensor_size=INPUT_TENSOR_SIZE,
    #     max_input_height_width=MAX_INPUT_HEIGHT_WIDTH,
    #     num_output_images=NUM_OUTPUT_IMAGES,
    #     num_inner_channels=NUM_INNER_CHANNELS,
    #     num_classes=NUM_CLASSES,
    #     batch_size=BATCH_SIZE,
    #     num_loader_workers=NUM_LOADER_WORKERS,
    #     history_len=HISTORY_LEN,
    # )

    model_frame: GANFramework = GANFramework(
        device=device,
        data_path=DATA_PATH,
        max_input_height_width=MAX_INPUT_HEIGHT_WIDTH,
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        num_loader_workers=NUM_LOADER_WORKERS,
    )

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
            model_frame.load_model(MODEL_PATH)

        # Watch if set
        no_except(wandb.watch, model_frame.wandb_trainable_model)

        for i in range(TRAIN[1]):
            if i % 5 == 0:
                loss_total, _ = model_frame.train(UPDATE_PL_LAMBDAS)
            else:
                loss_total, _ = model_frame.train(False)
            no_except(wandb.log, {"Epoch Loss": loss_total * BATCH_SIZE, "Epoch": i})
            # print(i, loss_total, model_frame.loss_net.loss_layer_scales)
            del loss_total
            if SAVE_EVERY_EPOCH:
                model_frame.save_model(MODEL_PATH)
        if not SAVE_EVERY_EPOCH:
            model_frame.save_model(MODEL_PATH)
        # quit()

    # Sampling
    if SAMPLE[0]:
        # model_frame.load_model(MODEL_PATH)
        img_list: sample_output = model_frame.sample(SAMPLE[1])
        for i, img in enumerate(img_list):
            print(img_list[i])
            plt.figure(i)
            plt.imshow(img)
            plt.show()

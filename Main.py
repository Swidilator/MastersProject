import torch
import os
from typing import Tuple, List, Any
from matplotlib import pyplot as plt

from CRN.CRN_Framework import CRNFramework
from GAN.GAN_Framework import GANFramework
from typing import Optional
from Training_Framework import MastersModel
import wandb
from PIL import Image

# from torchviz import make_dot

if __name__ == "__main__":
    # Determinism
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
            device = torch.device("cuda:0")
            print("Device: CUDA")
        else:
            device = torch.device("cpu")
            print("Device: CPU")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    # Model Settings
    MAX_INPUT_HEIGHT_WIDTH: tuple = (256, 512)
    NUM_CLASSES: int = 36
    BATCH_SIZE_SLICE: int = 1
    BATCH_SIZE_TOTAL: int = 16
    USE_TANH: bool = True
    USE_INPUT_NOISE: bool = False
    # CRN
    CRN_INPUT_TENSOR_SIZE: tuple = (4, 8)
    CRN_NUM_OUTPUT_IMAGES: int = 1
    CRN_NUM_INNER_CHANNELS = 1024
    CRN_HISTORY_LEN: int = 0

    # Run Specific Settings
    TRAIN: tuple = (True, 60)
    SAMPLE: tuple = (True, 1)
    WANDB: bool = False
    SAVE_EVERY_EPOCH: bool = True
    LOAD_BEFORE_RUN: bool = False
    SUBSET_SIZE: int = 256
    IMAGE_OUTPUT_DIR: str = "./Images/"
    # CRN
    CRN_UPDATE_PL_LAMBDAS: bool = False

    # Choose Model
    MODEL: str = "GAN"

    model_frame: Optional[MastersModel] = None

    if MODEL == "CRN":
        model_frame: CRNFramework = CRNFramework(
            device=device,
            data_path=DATA_PATH,
            batch_size_slice=BATCH_SIZE_SLICE,
            batch_size_total=BATCH_SIZE_TOTAL,
            num_loader_workers=NUM_LOADER_WORKERS,
            subset_size=SUBSET_SIZE,
            use_tanh=USE_TANH,
            use_input_noise=USE_INPUT_NOISE,
            settings={
                "input_tensor_size": CRN_INPUT_TENSOR_SIZE,
                "max_input_height_width": MAX_INPUT_HEIGHT_WIDTH,
                "num_output_images": CRN_NUM_OUTPUT_IMAGES,
                "num_inner_channels": CRN_NUM_INNER_CHANNELS,
                "num_classes": NUM_CLASSES,
                "history_len": CRN_HISTORY_LEN,
            },
        )

    elif MODEL == "GAN":
        model_frame: GANFramework = GANFramework(
            device=device,
            data_path=DATA_PATH,
            batch_size_slice=BATCH_SIZE_SLICE,
            batch_size_total=BATCH_SIZE_TOTAL,
            num_loader_workers=NUM_LOADER_WORKERS,
            subset_size=SUBSET_SIZE,
            use_tanh=USE_TANH,
            use_input_noise=USE_INPUT_NOISE,
            settings={
                "max_input_height_width": MAX_INPUT_HEIGHT_WIDTH,
                "num_classes": NUM_CLASSES,
            },
        )
    else:
        quit()

    if LOAD_BEFORE_RUN:
        model_frame.load_model(MODEL_PATH)

    if not WANDB:
        os.environ["WANDB_MODE"] = "dryrun"

    # Training
    if TRAIN[0]:
        wandb.init(
            project=MODEL.lower(),
            config={
                "Batch Size Total": BATCH_SIZE_TOTAL,
                "Batch Size Slice": BATCH_SIZE_SLICE,
                "Inner Channels": CRN_NUM_INNER_CHANNELS,
                "Output Size": MAX_INPUT_HEIGHT_WIDTH,
                "Training Machine": TRAINING_MACHINE,
                "Training Samples": SUBSET_SIZE,
            },
        )

        # Watch if set
        for val in model_frame.wandb_trainable_model:
            wandb.watch(val)

        for i in range(TRAIN[1]):
            print("Epoch:", i)
            if i % 5 == 0:
                loss_total, _ = model_frame.train(CRN_UPDATE_PL_LAMBDAS)
            else:
                loss_total, _ = model_frame.train(False)
            wandb.log(
                {"Epoch Loss": loss_total * BATCH_SIZE_TOTAL, "Epoch": i}, commit=False
            )
            # print(i, loss_total, model_frame.loss_net.loss_layer_scales)
            del loss_total
            if SAMPLE[0]:
                # model_frame.load_model(MODEL_PATH)
                img_list: List[Tuple[Any, Any]] = model_frame.sample(SAMPLE[1])

                if not os.path.exists(IMAGE_OUTPUT_DIR):
                    os.makedirs(IMAGE_OUTPUT_DIR)

                img_pair: Tuple[Any, Any]
                for j, img_pair in enumerate(img_list):
                    width: int = img_pair[0].size[0]
                    height: int = img_pair[0].size[1]

                    new_im: Image = Image.new("RGB", (width * 2, height))
                    new_im.paste(img_pair[0], (0, 0))
                    new_im.paste(img_pair[1], (width, 0))

                    filename = "{path}figure_{i}_{j}.png".format(
                        path=IMAGE_OUTPUT_DIR, i=i, j=j
                    )
                    new_im.save(filename)

                    caption: str = str("Sample Image " + str(i) + str(" ") + str(j))
                    wandb.log(
                        {"Sample Image": [wandb.Image(new_im, caption=caption)]},
                        commit=False,
                    )
                wandb.log(commit=True)
            if SAVE_EVERY_EPOCH:
                model_frame.save_model(MODEL_PATH)
        if not SAVE_EVERY_EPOCH:
            model_frame.save_model(MODEL_PATH)
        # quit()

    # Sampling
    # if SAMPLE[0]:
    #     # model_frame.load_model(MODEL_PATH)
    #     img_list: sample_output = model_frame.sample(SAMPLE[1])
    #     for i, img in enumerate(img_list):
    #         print(img_list[i])
    #         plt.figure(i)
    #         plt.imshow(img)
    #         plt.show()

    if SAMPLE[0] and not TRAIN[0]:
        if not os.path.exists(IMAGE_OUTPUT_DIR):
            os.makedirs(IMAGE_OUTPUT_DIR)
        # model_frame.load_model(MODEL_PATH)
        img_list: List[Any] = model_frame.sample(SAMPLE[1])
        for i, img_pair in enumerate(img_list):
            width: int = img_pair[0].size[0]
            height: int = img_pair[0].size[1]

            new_im: Image = Image.new("RGB", (width * 2, height))
            new_im.paste(img_pair[0], (0, 0))
            new_im.paste(img_pair[1], (width, 0))

            filename = "{path}figure_{i}.png".format(
                path=IMAGE_OUTPUT_DIR, i=i
            )
            new_im.save(filename)

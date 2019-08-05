import torch
import os
from matplotlib import pyplot as plt

from CRN import CRNFramework
from Data_Types import *

# from torchviz import make_dot

if __name__ == "__main__":
    MAX_INPUT_HEIGHT_WIDTH: tuple = (128, 256)
    INPUT_TENSOR_SIZE: tuple = (4, 8)
    NUM_OUTPUT_IMAGES: int = 1
    NUM_CLASSES: int = 35
    NUM_INNER_CHANNELS = 128
    BATCH_SIZE: int = 8

    PREFER_CUDA: bool = True
    NUM_LOADER_WORKERS: int = 2
    MODEL_PATH: str = "./Models/"

    CITYSCAPES_PATH: str = os.environ['CITYSCAPES_PATH']
    print("Dataset path: {cityscapes}".format(cityscapes=CITYSCAPES_PATH))

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
    )

    img_list: sample_output = crn_frame.sample(2)
    # crn_frame.save_model(MODEL_PATH)
    # crn_frame.load_model(MODEL_PATH)
    for i, img in enumerate(img_list):
        plt.figure(i)
        plt.imshow(img)
        plt.show()
    # crn_frame.train()
    quit()

    # data_set = CRNDataset(
    #     max_input_height_width=MAX_INPUT_HEIGHT_WIDTH,
    #     root="../CityScapes Samples/Train/",
    #     split="train",
    #     num_classes=NUM_CLASSES,
    # )
    #
    # data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
    #     data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_LOADER_WORKERS
    # )
    #
    # crn = CRN(
    #     input_tensor_size=INPUT_TENSOR_SIZE,
    #     final_image_size=MAX_INPUT_HEIGHT_WIDTH,
    #     num_output_images=NUM_OUTPUT_IMAGES,
    #     num_classes=NUM_CLASSES,
    # )
    # crn = crn.to(device)
    #
    # optimizer = torch.optim.SGD(crn.parameters(), lr=0.01, momentum=0.9)
    # loss_net: PerceptualLossNetwork = PerceptualLossNetwork()
    # loss_net = loss_net.to(device)
    #
    # for batch_idx, (img, msk) in enumerate(data_loader):
    #     optimizer.zero_grad()
    #     img = img.to(device)
    #     msk = msk.to(device)
    #     noise: torch.Tensor = torch.randn(
    #         BATCH_SIZE, 1, INPUT_TENSOR_SIZE[0], INPUT_TENSOR_SIZE[1], device=device
    #     )
    #     noise = noise.to(device)
    #
    #     out = crn(inputs=[msk, noise, BATCH_SIZE])
    #
    #     out = normalise(out)
    #
    #     loss: torch.Tensor = loss_net([out, img])
    #     loss.backward()
    #     optimizer.step()
    #     del loss, msk, noise, img
    #
    # torch.cuda.empty_cache()

    # to_image = torchvision.transforms.ToPILImage()
    # plt.figure(0)
    # plt.imshow(to_image(out[0, 0:3].cpu()))
    # plt.show()

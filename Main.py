import torchvision
import torch
from matplotlib import pyplot as plt
from Data_Management import CRNDataset
from CRN import CRN
from Perceptual_Loss import PerceptualLossNetwork
from torchviz import make_dot


def __single_channel_normalise__(channel: torch.Tensor, params: tuple) -> torch.Tensor:
    # channel = [H ,W]   params = (mean, std)
    return (channel - params[0]) / params[1]


def __single_image_normalise__(image: torch.Tensor, mean, std) -> torch.Tensor:
    for i in range(3):
        image[i] = __single_channel_normalise__(image[i], (mean[i], std[i]))
    return image


def normalise(input: torch.Tensor) -> torch.Tensor:
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if len(input.shape) == 4:
        for i in range(input.shape[0]):
            input[i] = __single_image_normalise__(input[i], mean, std)
    else:
        input = __single_image_normalise__(input, mean, std)
    return input


if __name__ == "__main__":
    MAX_INPUT_HEIGHT_WIDTH: tuple = (128, 256)
    INPUT_TENSOR_SIZE: tuple = (4, 8)
    NUM_OUTPUT_IMAGES: int = 1
    NUM_CLASSES: int = 35
    BATCH_SIZE: int = 1
    PREFER_CUDA: bool = True

    if PREFER_CUDA:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print('Device: CUDA')
        else:
            device = torch.device("cpu")
            print('Device: CPU')
    else:
        device = torch.device("cpu")
        print('Device: CPU')

    data_set = CRNDataset(
        max_input_height_width=MAX_INPUT_HEIGHT_WIDTH,
        root="../CityScapes Samples/Train/",
        split="train",
        num_classes=NUM_CLASSES
    )

    data_loader: torch.utils.data.DataLoader = torch.utils.data.DataLoader(
        data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0
    )

    crn = CRN(
        input_tensor_size=INPUT_TENSOR_SIZE,
        final_image_size=MAX_INPUT_HEIGHT_WIDTH,
        num_output_images=NUM_OUTPUT_IMAGES,
        num_classes=NUM_CLASSES
    )
    crn = crn.to(device)

    optimizer = torch.optim.SGD(crn.parameters(), lr=0.01, momentum=0.9)
    loss_net: PerceptualLossNetwork = PerceptualLossNetwork()
    loss_net = loss_net.to(device)

    for batch_idx, (img, msk) in enumerate(data_loader):
        optimizer.zero_grad()
        img = img.to(device)
        msk = msk.to(device)
        noise: torch.Tensor = torch.randn(BATCH_SIZE, 1, INPUT_TENSOR_SIZE[0], INPUT_TENSOR_SIZE[1], device=device)
        moise = noise.to(device)

        out = crn(
            inputs=[msk, noise, BATCH_SIZE]
        )

        out = normalise(out)

        loss: torch.Tensor = loss_net(
            [
                out,
                img
            ]
        )
        loss.backward()
        optimizer.step()
        del loss, msk, noise, img

    torch.cuda.empty_cache()

    # to_image = torchvision.transforms.ToPILImage()
    # plt.figure(0)
    # plt.imshow(to_image(out[0, 0:3].cpu()))
    # plt.show()
    # #del out, crn, msk, noise
    # torch.cuda.empty_cache()


    # plt.imshow(torchvision.transforms.ToPILImage(image[0].detach().numpy()))
    # plt.show()

    # rm = RefinementModule(
    #     prior_layer_channel_count=0,
    #     semantic_input_channel_count=35,
    #     output_channel_count=3,
    #     input_height_width=(32, 64),
    #     is_final_module=False,
    # )

    # output_test = rm(smnt, prior_layers=None)

    # plt.figure(0)
    # plt.imshow(torchvision.transforms.ToPILImage(smnt[0]))
    # plt.show()
    # plt.figure(1)
    # plt.imshow(output_test[0].detach().numpy())
    # plt.show()


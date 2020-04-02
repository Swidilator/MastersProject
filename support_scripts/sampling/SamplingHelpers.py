from PIL import Image
import random
import os
from typing import Tuple, List
from tqdm import tqdm

from GAN.GAN_Framework import GANFramework


# When sampling, this combines images for saving as side-by-side comparisons
def process_sampled_image(img_dict: dict) -> Image:
    img_width: int = img_dict["original_img"].size[0]
    img_height: int = img_dict["original_img"].size[1]

    key_list: list = list(img_dict.keys())
    ordered_keys: list = ["original_img", "output_img"]
    ordered_key_list: list = [
        *ordered_keys,
        *[x for x in key_list if x not in ordered_keys],
    ]

    num_images_hor: int = 2
    num_images_vert: int = (len(img_dict) + 1) // num_images_hor

    total_img_width: int = img_width * num_images_hor
    total_img_height: int = img_height * num_images_vert

    new_image: Image = Image.new("RGB", (total_img_width, total_img_height))
    for img_no_vert in range(total_img_height):
        for img_no_hor in range(total_img_width):
            index: int = (img_no_vert * num_images_hor) + img_no_hor
            if index < len(img_dict):
                new_image.paste(
                    img_dict[ordered_key_list[(img_no_vert * num_images_hor) + img_no_hor]],
                    (img_no_hor * img_width, img_no_vert * img_height),
                )
    # new_image.paste(img_list[0], (0, 0))
    # new_image.paste(img_list[1], (img_width, 0))
    return new_image


def create_image_directories(image_output_dir: str, model_name: str) -> str:
    if not os.path.exists(image_output_dir):
        os.makedirs(image_output_dir)

    model_image_dir: str = os.path.join(image_output_dir, model_name)
    if not os.path.exists(model_image_dir):
        os.makedirs(model_image_dir)

    return model_image_dir


def sample_from_model(
    model: GANFramework,
    sample_args: dict,
    mode: str,
    num_images: int = 1,
    indices: tuple = (0,),
) -> Tuple[List, List, List]:

    if mode == "random":
        sample_list: list = random.sample(
            range(len(model.__data_set_val__)), num_images
        )
    elif mode == "fixed":
        sample_list = list(indices)
    else:
        raise ValueError("Please choose from the following modes: 'random', 'fixed'.")

    img_list: List[dict,] = [model.sample(x, **sample_args) for x in sample_list]

    output_images: list = []
    output_dicts: list = []

    img_dict: dict
    for j, img_dict in enumerate(tqdm(img_list, desc="Sampling")):
        output_dicts.append(img_dict)
        output_images.append(process_sampled_image(img_dict))


    return output_dicts, output_images, sample_list

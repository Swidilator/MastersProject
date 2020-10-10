import os
import random
import torch
import numpy as np
from typing import Tuple, List, Union

from PIL import Image
from dominate.tags import output
from tqdm import tqdm

from support_scripts.utils import MastersModel


# When sampling, this combines images for saving as side-by-side comparisons
def __process_sampled_image__(img_dict: dict) -> Image:
    img_width: int = img_dict["original_img"].size[0]
    img_height: int = img_dict["original_img"].size[1]

    input_key_list: list = [
        key for key in list(img_dict.keys()) if type(img_dict[key]) is Image.Image
    ]
    output_key_list: list = list(img_dict["output_img_dict"].keys())
    true_output_key_list: list = [key for key in output_key_list if "output_img" in key]
    other_output_key_list: list = [
        key for key in output_key_list if "output_img" not in key
    ]

    input_img_dict: dict = {
        item[0]: item[1] for item in img_dict.items() if item[0] in input_key_list
    }
    output_img_dict: dict = img_dict["output_img_dict"]

    combined_dict: dict = {**input_img_dict, **output_img_dict}

    ordered_key_list: list = [
        "original_img",
        *true_output_key_list,
        *[key for key in input_key_list if "original_img" not in key],
        *other_output_key_list,
    ]

    num_images_hor: int = 2
    num_images_vert: int = (len(combined_dict) + 1) // num_images_hor

    total_img_width: int = img_width * num_images_hor
    total_img_height: int = img_height * num_images_vert

    new_image: Image = Image.new("RGB", (total_img_width, total_img_height))
    for img_no_vert in range(total_img_height):
        for img_no_hor in range(total_img_width):
            index: int = (img_no_vert * num_images_hor) + img_no_hor
            if index < len(combined_dict):
                new_image.paste(
                    combined_dict[
                        ordered_key_list[(img_no_vert * num_images_hor) + img_no_hor]
                    ],
                    (img_no_hor * img_width, img_no_vert * img_height),
                )

    img_dict.update({"composite_img": new_image})
    return img_dict


def create_image_directories(image_output_dir: str, model_name: str) -> str:
    model_image_dir: str = os.path.join(image_output_dir, model_name)
    os.makedirs(model_image_dir, exist_ok=True)
    return model_image_dir


def sample_from_model(
    model: MastersModel,
    sample_args: dict,
    mode: str,
    num_images: int = 1,
    indices: tuple = (0,),
) -> Tuple[List, List]:
    if mode == "random":
        sample_list: list = random.sample(range(len(model.data_set_val)), num_images)
    elif mode == "fixed":
        sample_list = list(indices)
    else:
        raise ValueError("Please choose from the following modes: 'random', 'fixed'.")

    with torch.no_grad():
        img_list: List[dict] = [model.sample(x, **sample_args) for x in sample_list]

    output_dicts: list = []

    img_dict: dict
    for j, img_dict in enumerate(tqdm(img_list, desc="Sampling")):
        output_dicts.append(__process_sampled_image__(img_dict))

    return output_dicts, sample_list


def sample_video_from_model(
    model: MastersModel,
    index_range: tuple,
    output_image_dir: str = None,
    output_image_number: int = 0,
    batch_size=2,
) -> Tuple[np.ndarray, np.ndarray]:
    # Tuples of indices
    batched_indices: list = []

    original_img_np: np.ndarray = None
    output_img_np: np.ndarray = None

    # TODO add support for vertical images.
    # standard_scale: int = 256

    for i in range(*index_range, batch_size):
        start_range_index = i
        end_range_index = i + batch_size
        if end_range_index > index_range[1]:
            end_range_index = index_range[1]
        batched_indices.append(tuple(range(start_range_index, end_range_index)))

    for tup_num, tup in enumerate(tqdm(batched_indices)):
        base_index: int = batch_size * tup_num + index_range[0]
        with torch.no_grad():
            image_dict_list: Union[list, dict] = model.sample(tup, video_dataset=True)
        if isinstance(image_dict_list, dict):
            image_dict_list = [image_dict_list]
        for dict_num, img_dict in enumerate(image_dict_list):
            if output_image_dir is not None:
                img_dict["output_img_dict"][f"output_img_{output_image_number}"].save(
                    os.path.join(
                        output_image_dir, f"{base_index + dict_num}".zfill(5) + ".png"
                    ), "PNG"
                )
            if output_img_np is None:
                output_img_np = np.array(
                    img_dict["output_img_dict"][f"output_img_{output_image_number}"].resize((512, 256), Image.BILINEAR)
                )[np.newaxis, :]
                original_img_np = np.array(img_dict["original_img"].resize((512, 256), Image.BILINEAR))[np.newaxis, :]
            else:
                tmp_output = np.array(
                    img_dict["output_img_dict"][f"output_img_{output_image_number}"].resize((512, 256), Image.BILINEAR)
                )[np.newaxis, :]
                output_img_np = np.append(output_img_np, tmp_output, axis=0)

                tmp_original = np.array(img_dict["original_img"].resize((512, 256), Image.BILINEAR))[np.newaxis, :]
                original_img_np = np.append(original_img_np, tmp_original, axis=0)

    return original_img_np, output_img_np

import os
import random

import imageio
import torch
import numpy as np
from typing import Tuple, List, Union, Dict, Optional

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from dataclasses import dataclass, field
from tqdm import tqdm

from support_scripts.utils import MastersModel

@dataclass
class SampleDataHolder:
    image_index: int
    video_sample: bool
    reference_image: List[Image.Image]
    mask_colour: List[Image.Image]
    output_image: List[Image.Image]  # If not a video_sample, can hold multiple outputs
    feature_selection: List[Image.Image] = field(default_factory=list)
    feature_selection_class_lists: List[list] = field(default_factory=list)
    composite_image: List[Image.Image] = field(default_factory=list)

    hallucinated_image: List[Image.Image] = field(default_factory=list)
    warped_image: List[Image.Image] = field(default_factory=list)
    combination_weights: List[Image.Image] = field(default_factory=list)
    output_flow: List[Image.Image] = field(default_factory=list)
    reference_flow: List[Image.Image] = field(default_factory=list)

    final_gif: np.ndarray = None


def __process_sampled_image__(image_data_holder: SampleDataHolder) -> Image:
    """
    When sampling, this combines images for saving as side-by-side comparisons

    """
    img_width: int = image_data_holder.reference_image[0].size[0]
    img_height: int = image_data_holder.reference_image[0].size[1]

    process_as_video: bool = image_data_holder.video_sample

    with open(
        "./support_scripts/sampling/NotoSans/NotoSans-Bold.ttf", "rb"
    ) as font_handle:
        font = ImageFont.truetype(font_handle, 14)

    # Create composite image for each video frame
    for video_frame in range(len(image_data_holder.reference_image)):
        ordered_image_list: list = [
            image_data_holder.reference_image[video_frame],
            *(
                image_data_holder.output_image
                if not process_as_video
                else [image_data_holder.output_image[video_frame]]
            ),
            image_data_holder.mask_colour[video_frame],
        ]
        # feature_selection
        if len(image_data_holder.feature_selection) > 0:
            ordered_image_list.append(image_data_holder.feature_selection[video_frame])

        # hallucinated_image
        if len(image_data_holder.hallucinated_image) > 0:
            single_image: Image = image_data_holder.hallucinated_image[video_frame]
            draw = ImageDraw.Draw(single_image)
            draw.text(
                (5, img_height - 20), "hallucinated_image", (255, 255, 255), font=font
            )
            ordered_image_list.append(single_image)

        # warped_image
        if len(image_data_holder.warped_image) > 0:
            single_image: Image = image_data_holder.warped_image[video_frame]
            draw = ImageDraw.Draw(single_image)
            draw.text((5, img_height - 20), "warped_image", (255, 255, 255), font=font)
            ordered_image_list.append(single_image)

        # combination_weights
        if len(image_data_holder.combination_weights) > 0:
            ordered_image_list.append(
                image_data_holder.combination_weights[video_frame]
            )

        # output_flow
        if len(image_data_holder.output_flow) > 0:
            single_image: Image = image_data_holder.output_flow[video_frame]
            draw = ImageDraw.Draw(single_image)
            draw.text((5, img_height - 20), "output_flow", (0, 0, 0), font=font)
            ordered_image_list.append(single_image)

        # reference_flow
        if len(image_data_holder.reference_flow) > 0:
            single_image: Image = image_data_holder.reference_flow[video_frame]
            draw = ImageDraw.Draw(single_image)
            draw.text((5, img_height - 20), "reference_flow", (0, 0, 0), font=font)
            ordered_image_list.append(single_image)

        # Two horizontal images, and calculate vertically
        num_images_hor: int = 2
        num_images_vert: int = (len(ordered_image_list) + 1) // num_images_hor

        total_img_width: int = img_width * num_images_hor
        total_img_height: int = img_height * num_images_vert

        # Create blank image of correct size, and paste individual images into them
        new_image: Image = Image.new("RGB", (total_img_width, total_img_height))
        for img_no_vert in range(total_img_height):
            for img_no_hor in range(total_img_width):
                index: int = (img_no_vert * num_images_hor) + img_no_hor
                if index < len(ordered_image_list):
                    new_image.paste(
                        ordered_image_list[index],
                        (img_no_hor * img_width, img_no_vert * img_height),
                    )

        # Update data holder with new composite image
        image_data_holder.composite_image.append(new_image)


def create_image_directories(image_output_dir: str, model_name: str) -> str:
    """
    Create directory for images to be stored in

    """
    model_image_dir: str = os.path.join(image_output_dir, model_name)
    os.makedirs(model_image_dir, exist_ok=True)
    return model_image_dir


def sample_from_model(
    model: MastersModel,
    mode: str,
    num_images: int = 1,
    indices: tuple = (0,),
) -> Tuple[List[SampleDataHolder], List]:
    """
    Sample a set of images/unmade videos from the given model, either from a set of indices,
    or a random number.

    """
    if mode == "random":
        sample_list: list = random.sample(range(len(model.data_set_val)), num_images)
    elif mode == "fixed":
        sample_list = list(indices)
    else:
        raise ValueError("Please choose from the following modes: 'random', 'fixed'.")

    # Sample from model
    with torch.no_grad():
        image_data_holders: List[SampleDataHolder] = [
            model.sample(x) for x in sample_list
        ]

    data_holder: SampleDataHolder
    for j, data_holder in enumerate(tqdm(image_data_holders, desc="Sampling")):

        # Create the composite images
        __process_sampled_image__(data_holder)

        # Create video if sampling for video
        if data_holder.video_sample:
            num_frames: int = len(data_holder.composite_image)
            img_shape: tuple = tuple(reversed(data_holder.composite_image[0].size))

            final_gif: np.ndarray = np.empty((num_frames, 3, *img_shape))

            for frame_no, comp_img in enumerate(data_holder.composite_image):
                final_gif[frame_no] = np.array(comp_img).transpose(2, 0, 1)

            data_holder.final_gif = final_gif.astype(np.uint8)

    return image_data_holders, sample_list


def sample_video_from_model(
    model: MastersModel,
    index_range: tuple,
    output_image_dir: str = None,
) -> Tuple[np.ndarray, np.ndarray]:
    # Tuples of indices
    reference_img_np: Optional[np.ndarray] = None
    fake_img_np: Optional[np.ndarray] = None

    # TODO add support for vertical images.
    # standard_scale: int = 256

    for i, image_index in enumerate(tqdm(range(index_range[0], index_range[1]))):
        with torch.no_grad():
            image_data_holder: SampleDataHolder = model.sample(image_index, demovideo_dataset=False)

            for frame_no in range(len(image_data_holder.output_image)):
                if output_image_dir is not None:
                    image_data_holder.output_image[frame_no].save(
                        os.path.join(
                            output_image_dir, f"{str(image_index).zfill(5)}_{str(frame_no).zfill(5)}".zfill(5) + ".png"
                        ),
                        "PNG",
                    )
                if fake_img_np is None:
                    fake_img_np = np.array(
                        image_data_holder.output_image[frame_no].resize(
                            (512, 256), Image.BILINEAR
                        )
                    )[np.newaxis, :]
                    reference_img_np = np.array(
                        image_data_holder.reference_image[0].resize(
                            (512, 256), Image.BILINEAR
                        )
                    )[np.newaxis, :]
                else:
                    tmp_output = np.array(
                        image_data_holder.output_image[frame_no].resize(
                            (512, 256), Image.BILINEAR
                        )
                    )[np.newaxis, :]
                    fake_img_np = np.append(fake_img_np, tmp_output, axis=0)

                    tmp_original = np.array(
                        image_data_holder.reference_image[0].resize(
                            (512, 256), Image.BILINEAR
                        )
                    )[np.newaxis, :]
                    reference_img_np = np.append(reference_img_np, tmp_original, axis=0)

    return reference_img_np, fake_img_np

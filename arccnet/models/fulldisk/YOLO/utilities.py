import os

import numpy as np
from p_tqdm import p_map
from PIL import Image, ImageDraw
from scipy.ndimage import rotate

from astropy.io import fits

from arccnet.models import labels
from arccnet.visualisation import utils as ut_v


def to_yolo(encoded_label, top_right, bottom_left, img_width, img_height):
    """
    Converts bounding box coordinates and class name into YOLO format.

    Parameters
    ----------
    encoded_label : str
        The index of the class associated with the object in the bounding box.
    top_right : tuple of float
        The (x, y) coordinates of the top-right corner of the bounding box.
    bottom_left : tuple of float
        The (x, y) coordinates of the bottom-left corner of the bounding box.
    img_width : int or float
        The width of the image in pixels.
    img_height : int or float
        The height of the image in pixels.

    Returns
    -------
    str
        A string in YOLO format representing the class ID and normalized bounding box.
        The format is: 'class_id x_center y_center width height', where all values except
        the class ID are normalized to the range [0, 1].

    Notes
    -----
    The YOLO format requires the center coordinates, width, and height of the bounding box
    to be normalized by the image dimensions. The class ID is determined using the `class_dict`
    lookup, and if the class name is not found in the dictionary, a default value of -1 is used.

    Example
    -------
    >>> class_dict = {"dog": 0, "cat": 1}
    >>> to_yolo("dog", (200, 150), (100, 50), 500, 400)
    '0 0.3 0.25 0.2 0.25'
    """
    x1, y1 = bottom_left
    x2, y2 = top_right
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return f"{encoded_label} {x_center} {y_center} {width} {height}"


def process_fits_row(row, local_path_root, base_dir, dataset_type, resize_dim=(640, 640)):
    """
    Process a single row in the DataFrame, handling the FITS file and saving the processed image and YOLO label.
    """
    arccnet_path_root = row["path"].split("/fits")[0]
    image_path = row["path"].replace(arccnet_path_root, local_path_root)
    label = row["yolo_label"]

    # Define output directories
    base_image_dir = os.path.join(base_dir, "images", dataset_type)
    base_label_dir = os.path.join(base_dir, "labels", dataset_type)
    os.makedirs(base_image_dir, exist_ok=True)
    os.makedirs(base_label_dir, exist_ok=True)

    # Process FITS file
    with fits.open(image_path) as img_fit:
        data = img_fit[1].data
        header = img_fit[1].header

    data = np.nan_to_num(data, nan=0.0)
    data = ut_v.hardtanh_transform_npy(data)
    crota2 = header["CROTA2"]
    data = rotate(data, crota2, reshape=False, mode="constant", cval=0)

    # Normalize and scale the image data
    data = (data - np.min(data)) / (np.max(data) - np.min(data))
    data = (data * 255).astype(np.uint8)

    # Convert to PIL Image and resize
    img = Image.fromarray(data)
    img_resized = img.resize(resize_dim)

    # Save the resized image as PNG
    basename = os.path.basename(image_path)
    png_filename = os.path.splitext(basename)[0] + ".png"
    img_resized.save(os.path.join(base_image_dir, png_filename))

    # Save the YOLO label in a .txt file
    label_filename = os.path.splitext(basename)[0] + ".txt"
    with open(os.path.join(base_label_dir, label_filename), "w") as label_file:
        label_file.write(label)


def process_and_save_fits(local_path_root, dataframe, base_dir, dataset_type, resize_dim=(640, 640)):
    """
    Process all rows in the DataFrame using parallel processing to handle FITS files.
    """
    print(f"Processing {dataset_type} dataset:\n")
    p_map(
        process_fits_row,
        dataframe.to_dict("records"),
        [local_path_root] * len(dataframe),
        [base_dir] * len(dataframe),
        [dataset_type] * len(dataframe),
        [resize_dim] * len(dataframe),
    )


def draw_yolo_labels_on_image(image_path, output_path=None):
    """
    Draws YOLO labels on the image by finding the corresponding label file,
    which assumes the label file is in a parallel 'labels' directory and has
    the same base name as the image file but with a '.txt' extension.

    Parameters:
    - image_path: Path to the input image.
    - output_path: Path where the output image will be saved.
                   If None, display the image.
    """
    class_names = [name for name, _ in sorted(labels.fulldisk_labels_ARs.items(), key=lambda item: item[1])]

    # Load the image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    width, height = img.size

    # Normalize the image path and replace 'images' with 'labels', change the extension to .txt
    label_path = image_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"

    # Read the label file
    with open(label_path) as file:
        for line in file:
            class_id, x_center, y_center, bbox_width, bbox_height = map(float, line.split())

            # Convert YOLO coordinates to PIL rectangle format
            x1 = (x_center - bbox_width / 2) * width
            y1 = (y_center - bbox_height / 2) * height
            x2 = (x_center + bbox_width / 2) * width
            y2 = (y_center + bbox_height / 2) * height

            # Draw the bounding box rectangle and the label
            draw.rectangle([x1, y1, x2, y2], outline="orange", width=1)
            draw.text((x1, y1), class_names[int(class_id)], fill="yellow")

    return img

from arccnet.models import labels


def to_yolo(class_name, top_right, bottom_left, img_width, img_height):
    """
    Converts bounding box coordinates and class name into YOLO format.

    Parameters
    ----------
    class_name : str
        The name of the class associated with the object in the bounding box.
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
    class_id = labels.fulldisk_labels_ARs.get(class_name, -1)  # Returns -1 if class_name is not found
    x1, y1 = bottom_left
    x2, y2 = top_right
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    return f"{class_id} {x_center} {y_center} {width} {height}"

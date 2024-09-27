import numpy as np

label_to_index = {
    "QS": 0,
    "IA": 1,
    "Alpha": 2,
    "Beta": 3,
    "Beta-Gamma": 4,
    "Beta-Delta": 5,
    "Beta-Gamma-Delta": 6,
    "Gamma": 7,
    "Gamma-Delta": 8,
}

index_to_label = {v: k for k, v in label_to_index.items()}

greek_mapping = {
    "Alpha": "α",
    "Beta": "β",
    "Gamma": "γ",
    "Delta": "δ",
}


def convert_to_greek_label(names_array):
    def map_to_greek(name):
        parts = name.split("-")
        greek_parts = [greek_mapping.get(part, part) for part in parts]
        return "-".join(greek_parts)

    return np.array([map_to_greek(name) for name in names_array])

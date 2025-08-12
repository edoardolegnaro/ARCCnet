import numpy as np

<<<<<<< HEAD
label_to_index = {
=======
LABEL_TO_INDEX = {
>>>>>>> main
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

<<<<<<< HEAD
index_to_label = {v: k for k, v in label_to_index.items()}

greek_mapping = {
=======
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}

GREEK_MAPPING = {
>>>>>>> main
    "Alpha": "α",
    "Beta": "β",
    "Gamma": "γ",
    "Delta": "δ",
}


def convert_to_greek_label(names_array):
    def map_to_greek(name):
        parts = name.split("-")
<<<<<<< HEAD
        greek_parts = [greek_mapping.get(part, part) for part in parts]
=======
        greek_parts = [GREEK_MAPPING.get(part, part) for part in parts]
>>>>>>> main
        return "-".join(greek_parts)

    return np.array([map_to_greek(name) for name in names_array])


<<<<<<< HEAD
# label mappings for training
qs_ia_ar_mapping = {
=======
# Label mappings for training
QS_IA_AR_MAPPING = {
>>>>>>> main
    "QS": "QS",
    "IA": "IA",
    "Alpha": "AR",
    "Beta": "AR",
    "Beta-Delta": "AR",
    "Beta-Gamma": "AR",
    "Beta-Gamma-Delta": "AR",
    "Gamma": "AR",
    "Gamma-Delta": "AR",
}

<<<<<<< HEAD
ia_ar_mapping = {
=======
IA_AR_MAPPING = {
>>>>>>> main
    "QS": None,
    "IA": "IA",
    "Alpha": "AR",
    "Beta": "AR",
    "Beta-Delta": "AR",
    "Beta-Gamma": "AR",
    "Beta-Gamma-Delta": "AR",
    "Gamma": "AR",
    "Gamma-Delta": "AR",
}

<<<<<<< HEAD
qs_ia_mapping = {
=======
QS_IA_MAPPING = {
>>>>>>> main
    "QS": "QS",
    "IA": "IA",
    "Alpha": None,
    "Beta": None,
    "Beta-Delta": None,
    "Beta-Gamma": None,
    "Beta-Gamma-Delta": None,
    "Gamma": None,
    "Gamma-Delta": None,
}

<<<<<<< HEAD
qs_ia_a_b_bg_mapping = {
=======
QS_IA_A_B_BG_MAPPING = {
>>>>>>> main
    "QS": "QS",
    "IA": "IA",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma",
    "Gamma": None,
    "Gamma-Delta": None,
}

<<<<<<< HEAD
ia_a_b_bg_mapping = {
    "QS": None,
    "IA": "IA",
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma",
    "Gamma": None,
    "Gamma-Delta": None,
}

a_b_bg_mapping = {
=======
A_B_BG_MAPPING = {
>>>>>>> main
    "QS": None,
    "IA": None,
    "Alpha": "Alpha",
    "Beta": "Beta",
    "Beta-Delta": "Beta",
    "Beta-Gamma": "Beta-Gamma",
    "Beta-Gamma-Delta": "Beta-Gamma",
    "Gamma": None,
    "Gamma-Delta": None,
}

<<<<<<< HEAD

label_mapping_dict = {
    "qs-ia-ar": qs_ia_ar_mapping,
    "ia-ar": ia_ar_mapping,
    "qs-ia": qs_ia_mapping,
    "qs-ia-a-b-bg": qs_ia_a_b_bg_mapping,
    "ia-a-b-bg": ia_a_b_bg_mapping,
    "a-b-bg": a_b_bg_mapping,
}

fulldisk_labels_ARs = {"Alpha": 0, "Beta": 1, "Beta-Gamma": 2, "Beta-Delta": 3, "Beta-Gamma-Delta": 4}
=======
LABEL_MAPPING_DICT = {
    "qs-ia-ar": QS_IA_AR_MAPPING,
    "ia-ar": IA_AR_MAPPING,
    "qs-ia": QS_IA_MAPPING,
    "qs-ia-a-b-bg": QS_IA_A_B_BG_MAPPING,
    "a-b-bg": A_B_BG_MAPPING,
}
>>>>>>> main

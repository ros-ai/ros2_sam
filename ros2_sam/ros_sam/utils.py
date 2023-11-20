import os
import pathlib
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import requests
from ament_index_python import get_package_share_path


def show_mask(mask, ax, color=(30, 140, 255, 150)) -> None:
    if color == "random":
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.asarray(color) / 255
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375) -> None:
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(
        pos_points[:, 0],
        pos_points[:, 1],
        color="green",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )
    ax.scatter(
        neg_points[:, 0],
        neg_points[:, 1],
        color="red",
        marker="*",
        s=marker_size,
        edgecolor="white",
        linewidth=1.25,
    )


def show_box(box, ax) -> None:
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )


class SAMDownloader:
    def __init__(self) -> None:
        self._model_url_dict = {
            "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
            "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
            "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        }

        self._checkpoint_dir = pathlib.Path(
            get_package_share_path("ros2_sam"), "models"
        )

    @property
    def checkpoint_dir(self) -> str:
        return self._checkpoint_dir.absolute()

    @property
    def model_url_dict(self) -> Dict[str, str]:
        return self._model_url_dict

    def check_model_availability(
        self,
        model_type: str = "vit_h",
    ) -> bool:
        return os.path.isfile(
            os.path.join(
                self._checkpoint_dir.absolute(),
                os.path.basename(self._model_url_dict[model_type]),
            )
        )

    def download(
        self,
        model_type: str = "vit_h",
    ) -> bool:
        model_url = self._model_url_dict[model_type]
        if not self.check_model_availability(model_type):
            if not self._checkpoint_dir.exists():
                self._checkpoint_dir.mkdir(parents=False)
            r = requests.get(model_url, stream=True)
            path = os.path.join(
                self._checkpoint_dir.absolute(),
                os.path.basename(model_url),
            )
            with open(path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
                        f.flush()
        return True

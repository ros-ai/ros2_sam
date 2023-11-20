# Free up GPU memory before loading the model
import gc
import glob
import os

import torch
from segment_anything import SamPredictor, sam_model_registry


class SAM:
    def __init__(
        self, checkpoint_dir: str, model_type: str = "vit_h", device: str = "cuda"
    ) -> None:
        checkpoints = glob.glob(os.path.join(checkpoint_dir, f"sam_{model_type}_*.pth"))

        if len(checkpoints) == 0:
            raise RuntimeError(
                f"No matching checkpoints for SAM model '{model_type}' was found in '{checkpoint_dir}'"
            )

        if len(checkpoints) > 1:
            raise RuntimeError(
                f"No unique checkpoint for SAM model '{model_type}' found in '{checkpoint_dir}'"
            )

        gc.collect()
        torch.cuda.empty_cache()

        sam_model = sam_model_registry[model_type](checkpoint=checkpoints[0])
        self._device = device

        if self._device is not None:
            sam_model.to(device=self._device)

        self._predictor = SamPredictor(sam_model)

    def __del__(self):
        gc.collect()
        torch.cuda.empty_cache()

    def segment(self, img, points, point_labels, boxes=None, multimask=True):
        self._predictor.set_image(img)
        return self._predictor.predict(
            point_coords=points,
            point_labels=point_labels,
            box=boxes,
            multimask_output=multimask,
        )

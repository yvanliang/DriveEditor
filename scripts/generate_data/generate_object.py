import json
import os
import cv2
import numpy as np
from nuscenes.utils.geometry_utils import view_points
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from segment_anything import sam_model_registry, SamPredictor

# 生成目标引导图，目标所在位置和原始位置相同

OUT_PATH = "./data/gm_data/obj_im_separate"


def scale_bbox(box: np.ndarray,
               im_size: tuple,
               scale: float = 2.0,
               min_size: int = 300) -> tuple[np.ndarray, np.ndarray]:
    """
    :param box: [[x1, y1],[x2, y2],...]
    :param im_size: (height, width)
    :param scale: scale factor
    :param min_size: min size of the box
    :return: scaled box for cropping original image, and original box: object box at cropped image

    以box的中心为中心，将box扩大scale倍，最小尺寸为min_size
    """
    x_min, y_min = box.min(axis=0)
    x_max, y_max = box.max(axis=0)
    x_center, y_center = (x_min + x_max) / 2, (y_min + y_max) / 2
    x_min_scale = max(x_center - max((x_max - x_center) * scale, min_size / 2), 0)
    y_min_scale = max(y_center - max((y_max - y_center) * scale, min_size / 2), 0)
    x_max_scale = min(x_center + max((x_max - x_center) * scale, min_size / 2), im_size[1])
    y_max_scale = min(y_center + max((y_max - y_center) * scale, min_size / 2), im_size[0])
    box_scale = np.array([x_min_scale, y_min_scale, x_max_scale, y_max_scale]).astype(np.int32)
    x_min_origin = max(x_min - x_min_scale, 0)
    y_min_origin = max(y_min - y_min_scale, 0)
    x_max_origin = min(x_max - x_min_scale, x_max_scale - x_min_scale)
    y_max_origin = min(y_max - y_min_scale, y_max_scale - y_min_scale)
    box_origin = np.array([x_min_origin, y_min_origin, x_max_origin, y_max_origin]).astype(np.int32)
    return box_scale, box_origin


class GenerateObject:
    def __init__(self, version: str, data_root: str, out_path: str = OUT_PATH, vis: bool = True):
        self.channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        self.exclude_category = ['movable_object.barrier', 'animal', 'human.pedestrian.adult', 'human.pedestrian.child',
                                 'human.pedestrian.construction_worker', 'human.pedestrian.personal_mobility',
                                 'human.pedestrian.police_officer', 'human.pedestrian.stroller',
                                 'human.pedestrian.wheelchair', 'vehicle.bicycle']
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        self.vis = vis
        self.out_path = out_path
        self.num_sample_per_data = 4
        self.max_dist = 36
        sam = sam_model_registry["vit_h"](checkpoint="checkpoints/sam_vit_h_4b8939.pth")
        sam.to(device='cuda')
        self.predictor = SamPredictor(sam)

        with open(os.path.join(data_root, "info_train.json"), 'r') as f:
            object_train = json.load(f)
        with open(os.path.join(data_root, "info_val.json"), 'r') as f:
            object_val = json.load(f)
        self.object_info = object_train + object_val

    def process(self):
        for info in tqdm(self.object_info, desc='scene'):
            for sd_data in info['sd_data']:
                ann_token, sd_token = sd_data['ann_token'], sd_data['sd_token']
                im, mask = self._generate_ins_condition_im(ann_token, sd_token)
                filename = os.path.join(self.out_path, info['instance_token'], f'{sd_token}_{ann_token}.png')
                os.makedirs(os.path.dirname(filename), exist_ok=True)
                cv2.imwrite(filename, im)
                cv2.imwrite(filename.replace('.png', '_m.png'), mask)

    def _generate_ins_condition_im(self, ann_token, sd_token):
        im_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sd_token)
        boxes = [b for b in boxes if b.token == ann_token]
        im = cv2.imread(im_path)
        if len(boxes) > 0 and ann_token != '':
            box = boxes[0]
            pt = view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2, :].T
            center_pt = view_points(box.center[:, None], view=camera_intrinsic, normalize=True)[:2, 0]
            box_scale, box_origin = scale_bbox(pt, im.shape, min_size=200)
            mask, score, vis_im = self._run_sam(im, box_scale, box_origin, center_pt)
            if score < 0.75:
                print(f'Warning: anno {ann_token} SAM score is {score}')
        else:
            vis_im = np.ones_like(im) * 255
            mask = np.zeros(im.shape[:2], dtype=np.uint8)
        return vis_im, mask

    def _run_sam(self, im: np.ndarray, box_scale: np.ndarray, box_origin: np.ndarray, center_pt: np.ndarray):
        """
        :param im: original image
        :param box_scale: scaled box for cropping original image
        :param box_origin: object box at cropped image
        :return: mask and score
        """

        im_crop = im[box_scale[1]:box_scale[3], box_scale[0]:box_scale[2]]
        im_crop_rgb = cv2.cvtColor(im_crop, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(im_crop_rgb)
        masks, scores, _ = self.predictor.predict(box=box_origin[None, :], multimask_output=False)
        mask = masks[0].astype(np.uint8) * 255
        score = scores[0]

        mask_all = np.zeros(im.shape[:2], dtype=np.uint8)
        mask_all[box_scale[1]:box_scale[3], box_scale[0]:box_scale[2]] = mask

        im[np.where(mask_all < 128)] = 255
        masked_image = cv2.bitwise_and(im, im, mask_all)

        return mask_all, score, masked_image


if __name__ == '__main__':
    pp = GenerateObject(version='v1.0-trainval', data_root='./data/')
    pp.process()

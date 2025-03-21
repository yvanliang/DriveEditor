import pickle
import pytorch_lightning as pl
import numpy as np
import torch
import cv2

from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


class NuScenesDataset(Dataset):
    def __init__(self, fps_id, num_frames, out_size):
        super().__init__()
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.num_frames_3d = 21
        self.out_size = out_size
        self.out_size_3d = (576, 576)
        self.obj_in_cond_scale = 0.89
        self.base_rotation_quaternion = Quaternion([0.5, 0.5, -0.5, 0.5])

        self.transform_img_3d = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.out_size_3d, antialias=True),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])
        self.transform_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(self.out_size, antialias=True),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])
        self.transform_img_resize = transforms.Resize(self.out_size, antialias=True)
        self.transform_mask = transforms.Compose([
            transforms.Resize([int(s / 8) for s in out_size], interpolation=InterpolationMode.NEAREST),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ])
        self.transform_depth = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x / (256. * 50.)),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
            transforms.Resize(self.out_size, antialias=True),
        ])

        self.video_data: list[dict] = []
        with open('checkpoints/train_data.pkl', 'rb') as f:
            self.video_data = pickle.load(f)


    def __getitem__(self, index):
        if self.video_data[index]['category_name'] == '':
            return self.get_blank(index)
        else:
            return self.get_gm(index)

    def get_blank(self, index):
        data = self.video_data[index]
        video_im,  video_cond_im, video_mask, video_position = [], [], [], []

        for frame_idx in range(self.num_frames):
            pos = data['pos'][frame_idx]

            # get gt image
            im = data['im'][frame_idx]

            # get target mask
            ratio = self.out_size[0] / im.shape[0]
            cond_im = self.transform_img(np.copy(im))

            mask = np.zeros(im.shape[:2], dtype=np.float32)
            mask[pos[1]:pos[3], pos[0]:pos[2]] = 1.

            pos4cond = np.round((pos * ratio)).astype(int)
            cond_im[:, pos4cond[1]:pos4cond[3], pos4cond[0]:pos4cond[2]] = \
                torch.zeros((3, pos4cond[3] - pos4cond[1], pos4cond[2] - pos4cond[0]))

            video_position.append({})
            video_im.append(self.transform_img(im))
            video_cond_im.append(cond_im)
            video_mask.append(torch.tensor(mask).unsqueeze(0))

        video_multiview = torch.randn((self.num_frames_3d, 3, 576, 576))
        obj_im_clip = torch.ones(3, 224, 224)
        cond_im_3d = torch.ones(1, 3, 576, 576)
        video_depth = torch.ones(self.num_frames, 6, *self.out_size) * -1.

        # get other info
        cond_aug_eval = torch.tensor(0.02)
        cond_aug = rand_log_normal(shape=[1, ], loc=-3.0, scale=0.5).squeeze()
        video_cond_im = torch.stack(video_cond_im, dim=0)
        cond_frame = video_cond_im + cond_aug * torch.randn_like(video_cond_im)
        cond_frame_eval = video_cond_im + cond_aug_eval * torch.randn_like(video_cond_im)
        cond_frames_3d = cond_im_3d + cond_aug * torch.randn_like(cond_im_3d)
        cond_frames_3d_eval = cond_im_3d + cond_aug_eval * torch.randn_like(cond_im_3d)
        video_im = torch.stack(video_im, dim=0)
        video_mask = self.transform_mask(torch.stack(video_mask, dim=0))
        video_mask_fuse = torch.zeros_like(video_mask)

        return {
            'jpg': video_im,
            'jpg_3d': video_multiview,
            'cond_frames': cond_frame,
            'cond_frames_eval': cond_frame_eval,
            'cond_frames_without_noise': [video_cond_im[0], obj_im_clip],
            'cond_frames_3d': cond_frames_3d,
            'cond_frames_3d_eval': cond_frames_3d_eval,
            'cond_frames_without_noise_3d': cond_im_3d,
            'depth': video_depth,
            'mask_concat': video_mask,
            'mask_fuse': video_mask_fuse,
            # time embedding
            'fps_id': torch.tensor(self.fps_id),
            'motion_bucket_id': torch.tensor(127),
            'cond_aug': cond_aug,
            'cond_aug_3d': cond_aug.repeat(self.num_frames_3d).unsqueeze(-1),
            'polars_rad_3d': torch.zeros(self.num_frames_3d).unsqueeze(-1),
            'azimuths_rad_3d': torch.zeros(self.num_frames_3d).unsqueeze(-1),
            'indices_3d': torch.zeros(self.num_frames),
            # additional_model_inputs
            'image_only_indicator': torch.zeros(self.num_frames),
            'num_video_frames': self.num_frames,
            'image_only_indicator_3d': torch.zeros(self.num_frames_3d),
            'num_video_frames_3d': self.num_frames_3d,
            'obj_pos': video_position,
            'valid_mask': torch.zeros(self.num_frames),
            'obj_ratio': torch.zeros(self.num_frames),
            'index': index,
        }

    def get_gm(self, index):
        data = self.video_data[index]
        video_im, video_cond_im, video_mask, video_mask_fuse = [], [], [], []
        video_enc_gt, video_cond_im_enc, video_position = [], [], []
        video_depth_concat, video_depth, video_depth_mask = [], [], []
        video_elevations, video_azimuths, valid_mask = [], [], []

        obj_im_clip, cond_im_3d, obj_ratio = None, None, torch.tensor(0.85)
        for frame_idx in range(self.num_frames):
            box = data['boxes'][frame_idx]
            camera_intrinsic = data['camera_intrinsic']
            valid_mask.append(1. if box is not None else 0.)

            # get gt image
            im = data['im'][frame_idx]

            # get cube image
            depth_cube = list()
            for depth_im in data['depth'][frame_idx]:
                depth_cube.append(self.transform_depth(depth_im))
            depth_cube = torch.concat(depth_cube, dim=0)

            # get 3d info
            elevations, azimuths = self._get_matrix(box)
            video_elevations.append(elevations)
            video_azimuths.append(azimuths)

            obj_separate_im, obj_separate_mask, obj_height = None, None, -1
            if frame_idx == data['keyframe_idx']:
                obj_separate_im = transforms.ToTensor()(data['obj_separate_im']) * 2.0 - 1.0
                obj_separate_mask = cv2.morphologyEx(data['obj_separate_mask'], cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
                obj_separate_mask = (
                        transforms.ToTensor()(obj_separate_mask) * 2.0 - 1.0
                ).repeat(3, 1, 1)

                # get ratio
                corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :].T
                pos_fuse = scale_bbox(corners, (900, 1600), 1.05, 1.05)
                indices = torch.argwhere(obj_separate_mask[0] > 0)
                (y_min, x_min), (y_max, x_max) = indices.min(dim=0).values, indices.max(dim=0).values
                h_obj, w_obj = y_max - y_min, x_max - x_min
                obj_ratio = h_obj / (pos_fuse[3] - pos_fuse[1])

                # get object image for clip condition
                obj_im_clip, _ = self._get_obj_im(obj_separate_im, obj_separate_mask, margin=32)

                # get object image for 3d model
                obj_scale = get_scale_by_obj_azimuth(azimuths)
                margin = 150 if 'movable_object' in data['category_name'] else 20
                cond_im_3d, obj_im_3d_shape = self._get_obj_im(obj_separate_im, obj_separate_mask, margin=margin,
                                                               out_size=576, scale_ratio=obj_scale)
                cond_im_3d = cond_im_3d.unsqueeze(0)

                obj_separate_im = self.transform_img_resize(obj_separate_im)
                obj_separate_mask = self.transform_img_resize(obj_separate_mask)
                obj_height = obj_im_3d_shape[0]

            # get target mask
            ratio = self.out_size[0] / im.shape[0]
            mask = np.zeros(im.shape[:2], dtype=np.float32)
            mask_fuse = np.zeros(im.shape[:2], dtype=np.float32)
            cond_im = self.transform_img(np.copy(im))
            if box is not None:
                corners = view_points(box.corners(), camera_intrinsic, normalize=True)[:2, :].T
                scale_y = torch.rand((1,)).item() * 0.7 + 1.2
                scale_x = torch.rand((1,)).item() * 0.8 + 1.2
                pos = scale_bbox(corners, im.shape[:2], scale_x, scale_y)
                pos_fuse = scale_bbox(corners, im.shape[:2], 1.05, 1.05)

                # target mask
                mask[pos[1]:pos[3], pos[0]:pos[2]] = 1.
                mask_fuse[pos_fuse[1]:pos_fuse[3], pos_fuse[0]:pos_fuse[2]] = 1.

                # cond image
                pos4cond = np.round((pos * ratio)).astype(int)
                cond_im[:, pos4cond[1]:pos4cond[3], pos4cond[0]:pos4cond[2]] = \
                    torch.zeros((3, pos4cond[3] - pos4cond[1], pos4cond[2] - pos4cond[0]))
                if obj_separate_mask is not None and obj_separate_im is not None:
                    cond_im[torch.where(obj_separate_mask > 0.)] = obj_separate_im[torch.where(obj_separate_mask > 0.)]

                # 3d info
                x_min, y_min = corners.min(axis=0)
                x_max, y_max = corners.max(axis=0)
                center_norm = np.array([(y_min + y_max) / 2, (x_min + x_max) / 2]) / np.array([900, 1600])

                video_position.append({'yx': torch.tensor(center_norm, dtype=torch.float32),
                                       'box_height': torch.tensor(pos_fuse[3] - pos_fuse[1]) * ratio,
                                       'obj_height': torch.tensor(obj_height),
                                       })
            else:
                video_position.append({})

            video_im.append(self.transform_img(im))
            video_cond_im.append(cond_im)
            video_mask.append(torch.tensor(mask).unsqueeze(0))
            video_mask_fuse.append(torch.tensor(mask_fuse).unsqueeze(0))
            video_depth.append(depth_cube)

        video_multiview = torch.randn((self.num_frames_3d, 3, 576, 576))

        v = None
        video_elevations = [v := x if x is not None else v for x in video_elevations]
        video_elevations = [v := x if x is not None else v for x in video_elevations[::-1]][::-1]
        video_elevations = torch.tensor(video_elevations)
        v = None
        video_azimuths = [v := x if x is not None else v for x in video_azimuths]
        video_azimuths = [v := x if x is not None else v for x in video_azimuths[::-1]][::-1]
        video_azimuths = torch.tensor(video_azimuths)

        # get other info
        cond_aug_eval = torch.tensor(0.02)
        cond_aug_eval_3d = torch.tensor(1e-5)
        cond_aug = rand_log_normal(shape=[1, ], loc=-3.0, scale=0.5).squeeze()
        video_cond_im = torch.stack(video_cond_im, dim=0)
        cond_frame = video_cond_im + cond_aug * torch.randn_like(video_cond_im)
        cond_frame_eval = video_cond_im + cond_aug_eval * torch.randn_like(video_cond_im)
        cond_frames_3d = cond_im_3d + cond_aug * torch.randn_like(cond_im_3d)
        cond_frames_3d_eval = cond_im_3d + cond_aug_eval_3d * torch.randn_like(cond_im_3d)
        video_im = torch.stack(video_im, dim=0)
        video_mask = self.transform_mask(torch.stack(video_mask, dim=0))
        video_mask_fuse = self.transform_mask(torch.stack(video_mask_fuse, dim=0))
        video_depth = torch.stack(video_depth, dim=0)
        valid_mask = torch.tensor(valid_mask)

        video_elevations, video_azimuths, video_3d_indices = (
            self._get_3d_full(video_elevations, video_azimuths, data['keyframe_idx']))

        return {
            'jpg': video_im,
            'jpg_3d': video_multiview,
            'cond_frames': cond_frame,
            'cond_frames_eval': cond_frame_eval,
            'cond_frames_without_noise': [video_cond_im[0], obj_im_clip],
            'cond_frames_3d': cond_frames_3d,
            'cond_frames_3d_eval': cond_frames_3d_eval,
            'cond_frames_without_noise_3d': cond_im_3d,
            'depth': video_depth,
            'mask_concat': video_mask,
            'mask_fuse': video_mask_fuse,
            # time embedding
            'fps_id': torch.tensor(self.fps_id),
            'motion_bucket_id': torch.tensor(127),
            'cond_aug': cond_aug,
            'cond_aug_3d': cond_aug.repeat(self.num_frames_3d).unsqueeze(-1),
            'polars_rad_3d': video_elevations.unsqueeze(-1),
            'azimuths_rad_3d': video_azimuths.unsqueeze(-1),
            'indices_3d': video_3d_indices,
            # additional_model_inputs
            'image_only_indicator': torch.zeros(self.num_frames),
            'num_video_frames': self.num_frames,
            'image_only_indicator_3d': torch.zeros(self.num_frames_3d),
            'num_video_frames_3d': self.num_frames_3d,
            'obj_pos': video_position,
            'valid_mask': valid_mask,
            'obj_ratio': obj_ratio.repeat(self.num_frames),
            'index': index,
        }

    def __len__(self):
        return len(self.video_data)

    def _get_matrix(self, box):
        elevations, azimuths = None, None
        if box is not None:
            sensor_rotation_object = box.orientation.rotation_matrix.T
            sensor_translation_object = -np.dot(sensor_rotation_object, box.center)

            elevations = np.arcsin(sensor_translation_object[2] / np.linalg.norm(sensor_translation_object))
            azimuths = np.arctan2(sensor_translation_object[1], sensor_translation_object[0])
            elevations = np.pi / 2 - elevations
            azimuths = np.pi + azimuths

        return elevations, azimuths

    def _get_obj_im(self, im_obj, mask_obj, margin=20, out_size=224, scale_ratio=1., center=None):
        indices = torch.argwhere(mask_obj[0] > 0)
        (y_min, x_min), (y_max, x_max) = indices.min(dim=0).values, indices.max(dim=0).values
        h_obj, w_obj = y_max - y_min, x_max - x_min
        factor = min((out_size - 2 * margin) / h_obj, (out_size - 2 * margin) / w_obj) / scale_ratio
        shape_obj = list(map(lambda x: int(x * factor), im_obj.shape[-2:]))
        im_obj = transforms.Resize(shape_obj, antialias=True)(im_obj)
        x_min, y_min, x_max, y_max = int(x_min * factor), int(y_min * factor), int(x_max * factor), int(y_max * factor)
        im_obj = im_obj[:, y_min:y_max, x_min:x_max]
        shape_obj = (y_max - y_min, x_max - x_min)
        if center is None:
            start_x = (out_size - shape_obj[1]) // 2
            start_y = (out_size - shape_obj[0]) // 2
            im = torch.ones(3, out_size, out_size)
            im[:, start_y:start_y + shape_obj[0], start_x:start_x + shape_obj[1]] = im_obj
        else:
            c_x, c_y = int(center[0] * factor) - x_min, int(center[1] * factor) - y_min
            left = max(-(c_x - out_size // 2), 0)
            top = max(-(c_y - out_size // 2), 0)
            right = max(c_x + out_size // 2 - im_obj.shape[2], 0)
            bottom = max(c_y + out_size // 2 - im_obj.shape[1], 0)
            im = torch.nn.functional.pad(im_obj, (left, right, top, bottom), value=1)
        return im, shape_obj

    def _get_3d_full(self, polars_rad, azimuths_rad, cond_idx):
        diff = azimuths_rad - azimuths_rad[cond_idx]
        diff = torch.where(diff < 0, diff + 2 * np.pi, diff)

        sample_azimuths_deg = [3., 6., 9., 12., 16., 23., 30., 45., 90., 135.,
                               225., 270., 315, 330., 337., 344., 348., 351., 354., 357., 0]
        sample_azimuths_rad = torch.tensor([np.deg2rad(a % 360) for a in sample_azimuths_deg])

        indices = torch.argmin(torch.abs(diff[:, None] - sample_azimuths_rad), dim=1).to(torch.int32)

        polars_rad = torch.ones(self.num_frames_3d) * torch.mean(polars_rad)
        return polars_rad, sample_azimuths_rad, indices


def get_obj_im_cond(im_obj, mask_obj, area, scale=0.85):
    h, w = scale * (area[3] - area[1]), scale * (area[2] - area[0])
    if h < 10 or w < 10:
        return None, None
    indices = torch.argwhere(mask_obj[0] > 0)
    (y_min, x_min), (y_max, x_max) = indices.min(dim=0).values, indices.max(dim=0).values
    h_obj, w_obj = y_max - y_min, x_max - x_min
    factor = h / h_obj
    shape_obj = list(map(lambda x: int(x * factor), im_obj.shape[-2:]))
    im_obj = transforms.Resize(shape_obj, antialias=False)(im_obj)
    mask_obj = transforms.Resize(shape_obj, antialias=False)(mask_obj)
    x_min, y_min, x_max, y_max = int(x_min * factor), int(y_min * factor), int(x_max * factor), int(y_max * factor)
    mask_obj = mask_obj[:, y_min:y_max, x_min:x_max]
    im_obj = im_obj[:, y_min:y_max, x_min:x_max]
    shape_obj = (y_max - y_min, x_max - x_min)
    center = [int((area[3] + area[1]) / 2), int((area[2] + area[0]) / 2)]

    s_x = int(max(center[1] - shape_obj[1] / 2, 0))
    crop_s_x = int(max(0 - (center[1] - shape_obj[1] / 2), 0))
    e_x = min(s_x + shape_obj[1], 1600)
    crop_e_x = max(s_x + shape_obj[1] - 1600, 0)
    s_y = int(max(center[0] - shape_obj[0] / 2, 0))
    crop_s_y = int(max(0 - (center[0] - shape_obj[0] / 2), 0))
    e_y = min(s_y + shape_obj[0], 900)
    crop_e_y = max((s_y + shape_obj[0] - 900), 0)

    s_x_obj = 0 + crop_s_x
    e_x_obj = shape_obj[1] - crop_e_x
    s_y_obj = 0 + crop_s_y
    e_y_obj = shape_obj[0] - crop_e_y

    if s_x > 1600 or s_y > 900:
        return None, None

    cond_im = torch.ones(3, 900, 1600)
    cond_mask = torch.zeros(3, 900, 1600)
    obj_im_crop = im_obj[:, s_y_obj:e_y_obj, s_x_obj:e_x_obj]
    mask_crop = mask_obj[:, s_y_obj:e_y_obj, s_x_obj:e_x_obj]
    tmp = cond_im[:, s_y:e_y, s_x:e_x]
    tmp[torch.where(mask_crop > 0)] = obj_im_crop[torch.where(mask_crop > 0)]
    cond_im[:, s_y:e_y, s_x:e_x] = tmp
    cond_mask[:, s_y:e_y, s_x:e_x] = mask_crop
    cond_im = cond_im.contiguous()
    cond_mask = cond_mask.contiguous()
    return cond_im, cond_mask


def get_scale_by_obj_azimuth(obj_azimuth, scale_min=1.2, scale_max=2.8):
    degree_ori = np.degrees(obj_azimuth)
    degree = abs(degree_ori % 180 - 90)
    if degree <= 15:
        scale = scale_min
    elif 15 < degree <= 30:
        scale = 1.4
    elif 30 < degree <= 40:
        scale = 1.7
    elif 40 < degree <= 60:
        scale = 2.0
    elif 60 < degree <= 75:
        scale = 2.2
    elif 75 < degree <= 80:
        scale = 2.4
    elif 80 < degree <= 83:
        scale = 2.6
    else:
        scale = scale_max
    return scale


def scale_bbox(corners: np.ndarray,
               im_size: tuple,
               scale_x: float = 2.0,
               scale_y: float = 2.0,
               min_size: int = 0) -> np.ndarray:
    x_min, y_min = corners.min(axis=0)
    x_max, y_max = corners.max(axis=0)
    if min(x_max - x_min, y_max - y_min) > im_size[0] / 2:
        scale_x = min(scale_x, 1.5)
        scale_y = min(scale_y, 1.25)
    center_x_ori, center_y_ori = (x_min + x_max) / 2, (y_min + y_max) / 2
    half_scaled_w = max((x_max - x_min) / 2 * scale_x, min_size / 2)
    half_scaled_h = max((y_max - y_min) / 2 * scale_y, min_size / 2)
    center_range_x = half_scaled_w - (x_max - x_min) / 2
    center_range_y = half_scaled_h - (y_max - y_min) / 2
    center_x = int((torch.rand((1,)).item() * 2 - 1) * center_range_x) + center_x_ori
    center_y = int((torch.rand((1,)).item() * 2 - 1) * center_range_y) + center_y_ori
    x_min_scale = max(int(center_x - half_scaled_w), 0)
    y_min_scale = max(int(center_y - half_scaled_h), 0)
    x_max_scale = min(int(center_x + half_scaled_w), im_size[1])
    y_max_scale = min(int(center_y + half_scaled_h), im_size[0])
    return np.array([x_min_scale, y_min_scale, x_max_scale, y_max_scale]).astype(np.int32)


def rand_log_normal(shape, loc=0., scale=1.):
    log_sigma = loc + scale * torch.randn(shape)
    return log_sigma.exp()


class NuScenesLoader(pl.LightningDataModule):
    def __init__(self, num_workers, out_size, fps_id=None, num_frames=10, shuffle=True, batch_size=1,
                 *args, **kwargs):
        super().__init__()
        self.batch_size = batch_size
        self.fps_id = fps_id
        self.num_frames = num_frames
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.out_size = out_size
        self.nus_train, self.nus_val = None, None
        self.nus_test = None
        self.nus_predict = None

    def prepare_data(self):
        pass

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self.nus_train = NuScenesDataset(self.fps_id, self.num_frames, self.out_size)
            self.nus_val = NuScenesDataset(self.fps_id, self.num_frames, self.out_size)

        if stage == "test":
            self.nus_test = NuScenesDataset(self.fps_id, self.num_frames, self.out_size)

        if stage == "predict":
            self.nus_predict = NuScenesDataset(self.fps_id, self.num_frames, self.out_size)

    def train_dataloader(self):
        return DataLoader(
            self.nus_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.nus_test,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.nus_val,
            batch_size=1,
            shuffle=False,
            num_workers=self.num_workers,
        )

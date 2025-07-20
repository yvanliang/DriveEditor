import json
import os

import cv2
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt, rcParams
from nuscenes.utils.geometry_utils import BoxVisibility, view_points, box_in_image
from nuscenes.nuscenes import NuScenes
from segment_anything import sam_model_registry, SamPredictor
from tqdm import tqdm

OUT_PATH = "vis_result_new"


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


class MakeObjectData:
    def __init__(self, version: str, data_root: str, out_path: str = OUT_PATH, vis: bool = True):
        self.channels = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        self.exclude_category = ['movable_object.barrier', 'static_object.bicycle_rack',
                                 'movable_object.pushable_pullable', 'human.pedestrian.police_officer',
                                 'human.pedestrian.construction_worker', 'vehicle.motorcycle', 'human.pedestrian.adult']
        self.nusc = NuScenes(version=version, dataroot=data_root, verbose=True)
        self.vis = vis
        self.out_path = out_path
        self.num_sample_per_data = 4
        self.max_dist = 32
        self.min_size = 70

    def process(self):
        for scene in tqdm(self.nusc.scene, desc='scene'):
            num_sample = len(self.nusc.field2token('sample', 'scene_token', scene['token']))
            ins_per_channel, idx2sample_token = self._gather_info_each_scene(scene)
            ins_per_channel_filter = self._filter_info_each_scene(ins_per_channel, idx2sample_token, num_sample, scene['name'])
            with open(os.path.join('data/vis', f"{scene['name']}.json"), 'w') as f:
                json.dump(ins_per_channel_filter, f, indent=4, separators=(',', ': '))

    def _gather_info_each_scene(self, scene):
        """对每个关键帧中的目标进行遍历，合并相同目标的关键帧"""
        sample_idx = 0
        sample_token = scene['first_sample_token']
        ins_per_channel = {k: dict() for k in self.channels}
        idx2sample_token = dict()

        while True:
            idx2sample_token[sample_idx] = sample_token
            info = self._gather_ins_info_each_sample(sample_token)
            # 聚合信息
            for cam in self.channels:
                for ins, value in info[cam].items():
                    if ins not in ins_per_channel[cam]:
                        ins_per_channel[cam][ins] = [{'sample_idx': sample_idx, **value}]
                    else:
                        ins_per_channel[cam][ins].append({'sample_idx': sample_idx, **value})

            if sample_token == scene['last_sample_token']:
                break
            sample_idx += 1
            sample_token = self.nusc.get('sample', sample_token)['next']
        return ins_per_channel, idx2sample_token

    def _gather_ins_info_each_sample(self, sample_token: str) -> dict:
        """对每个sample进行操作"""
        sample_rec = self.nusc.get('sample', sample_token)
        ins_in_sample_per_channel = dict()
        for cam in self.channels:
            ins_in_sample_per_channel[cam] = dict()
            _, boxes, cam_intrinsic = self.nusc.get_sample_data(sample_rec['data'][cam], box_vis_level=BoxVisibility.ANY)

            for box in boxes:
                ann_rec = self.nusc.get('sample_annotation', box.token)
                category = ann_rec['category_name']
                visibility = int(ann_rec['visibility_token'])

                # 计算距离
                sample_data_record = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
                pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
                dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_rec['translation']))

                # 计算是否完全在图像内
                sd_record = self.nusc.get('sample_data', sample_rec['data'][cam])
                imsize = (sd_record['width'], sd_record['height'])
                all_vis = box_in_image(box, cam_intrinsic, imsize, vis_level=BoxVisibility.ALL)

                # 计算目标在图中的大小
                pt = view_points(box.corners(), view=cam_intrinsic, normalize=True)[:2, :].T
                x_min, y_min = pt.min(axis=0)
                x_max, y_max = pt.max(axis=0)
                size = min(x_max - x_min, y_max - y_min)

                # 1.目标距离小于max_dist； 2.目标在图中大小大于200像素； 3.目标类别不在exclude_category中 4.目标可见性在60%以上
                if dist < self.max_dist and size > self.min_size and category not in self.exclude_category and visibility == 4:
                    ins_in_sample_per_channel[cam][ann_rec['instance_token']] = {'category_name': category,
                                                                                 'dist': dist, 'ann_token': ann_rec['token'],
                                                                                 'visibility': visibility, 'all_vis': all_vis}
        return ins_in_sample_per_channel

    def _filter_info_each_scene(self, ins_per_channel, idx2sample_token, num_sample, scene_name):
        """引入连续出现的要求，过滤目标"""
        ins_per_channel_filter = {k: list() for k in self.channels}
        for cam in self.channels:
            for idx in range(0, num_sample - self.num_sample_per_data):
                for ins, value in ins_per_channel[cam].items():
                    # 要求目标在连续self.num_sample_per_data帧中均存在
                    if set(range(idx, idx + self.num_sample_per_data)).issubset(set(v['sample_idx'] for v in value)):
                        # 寻找condition，要求完整出现在图中，且不被遮挡
                        has_condition = any([v['all_vis'] and v['visibility'] == 4 for v in value])
                        dist_list = []
                        for v in value:
                            if v['sample_idx'] in range(idx, idx + self.num_sample_per_data):
                                if v['all_vis'] and v['visibility'] == 4:
                                    dist_list.append(v['dist'])
                                else:
                                    dist_list.append(100)
                        min_dist_idx = dist_list.index(min(dist_list))
                        if min(dist_list) < 15 and max(dist_list) < 25:
                            anns_token = []
                            for v in value:
                                for i in range(idx, idx + self.num_sample_per_data):
                                    if v['sample_idx'] == i:
                                        anns_token.append(v['ann_token'])
                                        v['dist'] = 100

                            ins_per_channel_filter[cam].append({
                                'instance_token': ins,
                                'scene_name': scene_name,
                                "channel": cam,
                                'first_sample_idx': idx,
                                'first_sample_token': idx2sample_token[idx],
                                'category_name': value[0]['category_name'],
                                'min_dist_sample_idx': value[min_dist_idx]['sample_idx'] if has_condition else None,
                                'min_dist_sample_token': idx2sample_token[value[min_dist_idx]['sample_idx']] if has_condition else None,
                                'min_dist_ann_token': value[min_dist_idx]['ann_token'] if has_condition else None,
                                'anns_token': anns_token})

        return ins_per_channel_filter

    def _get_other_channel_ann(self, ins_per_channel_filter):
        """获取其他通道的图像作为新视角条件"""
        other_channel_anno = dict()
        for cam_global in self.channels:
            other_channel_anno[cam_global] = dict()
            for ins in ins_per_channel_filter[cam_global]:
                ann_tokens = self.nusc.field2token('sample_annotation', 'instance_token', ins['instance_token'])
                for ann_token in ann_tokens:
                    ann_record = self.nusc.get('sample_annotation', ann_token)
                    sample_rec = self.nusc.get('sample', ann_record['sample_token'])
                    for cam in self.channels:
                        _, boxes, cam_intrinsic = self.nusc.get_sample_data(sample_rec['data'][cam],
                                                                            box_vis_level=BoxVisibility.ALL, selected_anntokens=[ann_token])
                        if len(boxes) > 0:
                            assert len(boxes) < 2, 'Error: Found multiple annotations. Something is wrong!'
                            box = boxes[0]

                            # 计算目标是否被遮挡
                            visibility = int(ann_record['visibility_token'])

                            # 计算距离
                            ann_rec = self.nusc.get('sample_annotation', box.token)
                            sample_data_record = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
                            pose_record = self.nusc.get('ego_pose', sample_data_record['ego_pose_token'])
                            dist = np.linalg.norm(np.array(pose_record['translation']) - np.array(ann_rec['translation']))

                            if dist < self.max_dist and visibility == 4 and cam != cam_global:
                                if ins['instance_token'] not in other_channel_anno[cam_global]:
                                    other_channel_anno[cam_global][ins['instance_token']] = \
                                        [{'channel': cam, 'dist': dist, 'anno_token': ann_rec['token']}]
                                else:
                                    other_channel_anno[cam_global][ins['instance_token']].append(
                                        {'channel': cam, 'dist': dist, 'anno_token': ann_rec['token']})

        # 仅保留最小距离的结果
        condition_anno_min_dist = dict()
        for cam_global in self.channels:
            condition_anno_min_dist[cam_global] = dict()
            for ins, value in other_channel_anno[cam_global].items():
                dist_list = [v['dist'] for v in value]
                min_dist_idx = dist_list.index(min(dist_list))
                condition_anno_min_dist[cam_global][ins] = value[min_dist_idx]

        # 将结果写入ins_per_channel_filter
        ins_per_channel_filter_final = dict()
        for cam_global in self.channels:
            ins_per_channel_filter_final[cam_global] = list()
            for ins_info in ins_per_channel_filter[cam_global]:
                # 对应instance存在其他视角的condition图片
                if ins_info['instance_token'] in condition_anno_min_dist[cam_global].keys():
                    ins_info_new = {**ins_info, 'other_channel': condition_anno_min_dist[cam_global][ins_info['instance_token']]}
                    ins_per_channel_filter_final[cam_global].append(ins_info_new)
                # 对应instance不存在其他视角的condition图片，自身有all_vis图片
                elif ins_info['min_dist_ann_token'] is not None:
                    ins_per_channel_filter_final[cam_global].append(ins_info)

        return ins_per_channel_filter_final

    def _save_image(self, ins_per_channel_filter, scene_name, extra_info=True):
        for cam_global in self.channels:
            os.makedirs(os.path.join(self.out_path, scene_name, cam_global), exist_ok=True)
            for ins in ins_per_channel_filter[cam_global]:
                ann_list = [*ins['anns_token'], ins['min_dist_ann_token']]
                fig, axes = plt.subplots(3, 4, figsize=(25, 15), subplot_kw={'aspect': 'auto'}, tight_layout=True)

                # 绘制cam_global视角下的目标
                for i, ann_token in enumerate(ann_list):
                    if ann_token is None:  # cam_global视角下没有condition图（但其他视角有）
                        continue
                    ann_record = self.nusc.get('sample_annotation', ann_token)
                    sample_record = self.nusc.get('sample', ann_record['sample_token'])
                    data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_record['data'][cam_global], selected_anntokens=[ann_token])
                    assert len(boxes) == 1
                    box = boxes[0]

                    im = Image.open(data_path)
                    axes[i // 2][(i % 2) * 2].imshow(im)
                    axes[i // 2][(i % 2) * 2].axis('off')
                    axes[i // 2][(i % 2) * 2].set_aspect('equal')
                    axes[i // 2][(i % 2) * 2].set_title(f'{i + 1}: {ann_token}')
                    c = np.array(self.nusc.colormap[box.name]) / 255.0
                    box.render(axes[i // 2][(i % 2) * 2], view=camera_intrinsic, normalize=True, colors=(c, c, c))

                    try:
                        pt = view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2, :].T
                        box_scale, box_origin = scale_bbox(pt, im.size[::-1], min_size=200)
                        mask, score, vis_im = self._run_sam(np.array(im), box_scale, box_origin)
                        axes[i // 2][(i % 2) * 2 + 1].imshow(vis_im)
                        axes[i // 2][(i % 2) * 2 + 1].axis('off')
                        axes[i // 2][(i % 2) * 2 + 1].set_aspect('equal')
                        axes[i // 2][(i % 2) * 2 + 1].set_title(f'{i + 1}. im_size: {vis_im.shape[:2]}')
                    except:
                        pass

                # 绘制其他视角的目标
                if 'other_channel' in ins:
                    cam = ins['other_channel']['channel']
                    ann_token = ins['other_channel']['anno_token']
                    ann_record = self.nusc.get('sample_annotation', ann_token)
                    sample_record = self.nusc.get('sample', ann_record['sample_token'])
                    data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_record['data'][cam], selected_anntokens=[ann_token])
                    assert len(boxes) == 1
                    box = boxes[0]

                    im = Image.open(data_path)
                    axes[2][2].imshow(im)
                    axes[2][2].axis('off')
                    axes[2][2].set_aspect('equal')
                    axes[2][2].set_title(f"6 distance: {ins['other_channel']['dist']}")
                    c = np.array(self.nusc.colormap[box.name]) / 255.0
                    box.render(axes[2][2], view=camera_intrinsic, normalize=True, colors=(c, c, c))

                    try:
                        pt = view_points(box.corners(), view=camera_intrinsic, normalize=True)[:2, :].T
                        box_scale, box_origin = scale_bbox(pt, im.size[::-1], min_size=200)
                        mask, score, vis_im = self._run_sam(np.array(im), box_scale, box_origin)
                        axes[2][3].imshow(vis_im)
                        axes[2][3].axis('off')
                        axes[2][3].set_aspect('equal')
                        axes[2][3].set_title(f'6. im_size: {vis_im.shape[:2]}')
                    except:
                        pass

                if extra_info:
                    rcParams['font.family'] = 'monospace'
                    category = ins['category_name']

                    information = ' \n'.join(['category: {}'.format(category)])
                    plt.annotate(information, (0, 0), (0, -20), xycoords='axes fraction', textcoords='offset points', va='top')

                file_name = f"{ins['first_sample_idx']:02d}_{ins['category_name']}_{ins['instance_token']}.png"
                plt.savefig(os.path.join(self.out_path, scene_name, cam_global, file_name))
                plt.close()

    def _run_sam(self, im: np.ndarray, box_scale: np.ndarray, box_origin: np.ndarray):
        """
        :param im: original image
        :param box_scale: scaled box for cropping original image
        :param box_origin: object box at cropped image
        :return: mask and score
        """

        def post_process(img: np.ndarray, mask_im: np.ndarray, scale: float = 1.4):
            img[np.where(mask_im < 128)] = 255
            masked_image = cv2.bitwise_and(img, img, mask_im)
            x, y, w, h = cv2.boundingRect(mask)
            im_w = int(w * scale)
            im_h = int(h * scale)

            size = max(im_w, im_h)

            square_canvas = np.ones((size, size, 3), dtype=np.uint8) * 255  # 白色背景

            start_x = (size - w) // 2
            start_y = (size - h) // 2
            square_canvas[start_y:start_y + h, start_x:start_x + w] = masked_image[y:y + h, x:x + w]
            return square_canvas

        im_crop_rgb = im[box_scale[1]:box_scale[3], box_scale[0]:box_scale[2]]

        if self.predictor is not None:
            self.predictor.set_image(im_crop_rgb)
            masks, scores, _ = self.predictor.predict(box=box_origin[None, :], multimask_output=False)
            mask = masks[0].astype(np.uint8) * 255
            score = scores[0]
        else:
            mask = np.zeros(im_crop_rgb.shape[:2], dtype=np.uint8)
            mask[box_origin[1]:box_origin[3], box_origin[0]:box_origin[2]] = 255
            score = 1.

        im = post_process(im_crop_rgb, mask)
        return mask, score, im


if __name__ == '__main__':
    pp = MakeObjectData(version='v1.0-trainval', data_root='./dataset/nuscenes/')
    pp.process()
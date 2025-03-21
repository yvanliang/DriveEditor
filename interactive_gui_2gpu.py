import math
import os
import tempfile
import cv2
import torch
import numpy as np
import gradio as gr
import pickle

from einops import repeat, rearrange
from nuscenes.utils.geometry_utils import view_points
from omegaconf import OmegaConf
from pyquaternion import Quaternion
from copy import deepcopy
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from sgm import instantiate_from_config

CUBE_FACES = [
    [0, 1, 2, 3],  # front
    [4, 5, 6, 7],  # back
    [0, 4, 7, 3],  # side
    [1, 5, 6, 2],  # side
    [0, 1, 5, 4],  # top
    [2, 3, 7, 6],  # bottom
]


class GradioShow:
    def __init__(
            self,
            num_frames: int = 10,
            step: int = 25,
            out_size: tuple = (576, 1024),
    ):
        model_config = "configs/sample.yaml"
        self.out_size = out_size
        self.out_size_3d = (576, 576)
        self.num_frames = num_frames
        self.num_frames_3d = 21
        self.device0 = "cuda:0"  # model
        self.device2 = "cuda:1"  # 3d model
        self.model = load_model(
            model_config,
            self.device0,
            num_steps=step,
            num_frames=num_frames,
            verbose=True,
        )
        for name, layer in self.model.model.diffusion_model.named_children():
            if "_3d" in name:
                layer.to(self.device2)

        with open("checkpoints/data.pkl", "rb") as f:
            self.video_data = pickle.load(f)
        self.video_data.sort(key=lambda x: x['name'])
        self.ratio = self.out_size[0] / 900
        self.idx, self.camera_intrinsic, self.data = None, None, None
        self.box = []
        self.selected_object_idx = 0
        self.im, self.im_with_box, self.im_result = [], [], []
        self.to_tensor = transforms.ToTensor()
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
            transforms.Lambda(lambda x: x / (256. * 50.)),  # 最大深度值为80，大于50则clip为1
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
            transforms.Resize(self.out_size, antialias=True),
        ])

    def run(self):
        with gr.Blocks(title="DriveEditor Editing Tool") as demo:
            gr.HTML("""<h1 style="text-align: center;">DriveEditor: A Unified 3D Information-Guided Framework for 
            Controllable Object Editing in Driving Scenes</h1>""")
            with gr.Row():
                with gr.Column():
                    with gr.Group():
                        gr.Markdown("**Sample/Box Selection**")
                        data_names = [f"{i + 1}.{d['name']}" for i, d in enumerate(self.video_data)]
                        data_idx = gr.Dropdown(data_names, type="index", label="Data Selection")
                        editing = gr.Radio(
                            choices=["Deletion", "Insertion", "Repositioning", "Replacement"],
                            label="Editing Operation",
                            value="Deletion",
                        )

                    with gr.Group(visible=False) as box_group:
                        gr.Markdown("**Box Editing Options**")
                        box_id = gr.Dropdown(None, type="value", label="Select BBox")
                        with gr.Row():
                            with gr.Column():
                                x_offset = gr.Slider(minimum=-30, maximum=30, step=0.1, value=0.0, interactive=True,
                                                     label="x (+ -> move right)")
                                y_offset = gr.Slider(minimum=-30, maximum=30, step=0.1, value=0.0, interactive=True,
                                                     label="y (+ -> move down)")
                                z_offset = gr.Slider(minimum=-30, maximum=30, step=0.1, value=0.0, interactive=True,
                                                     label="z (+ -> move front)")
                            with gr.Column():
                                l_offset = gr.Slider(minimum=0, maximum=10, step=0.1, value=0.0, interactive=True,
                                                     label="l (head-tail)")
                                w_offset = gr.Slider(minimum=0, maximum=10, step=0.1, value=0.0, interactive=True,
                                                     label="w (left-right)")
                                h_offset = gr.Slider(minimum=0, maximum=10, step=0.1, value=0.0, interactive=True,
                                                     label="h (top-bottom)")
                        yaw_offset = gr.Slider(minimum=-180, maximum=180, value=0.0, interactive=True,
                                               label="yaw")
                        edited_viz = gr.Image(label="Edited Box Visualization")
                        with gr.Group():
                            with gr.Row():
                                with gr.Column():
                                    rst_anno_btn = gr.Button(value="Reset Annotation")
                                with gr.Column():
                                    rst_anno_all_btn = gr.Button(value="Reset All Annotations")

                    with gr.Group(visible=False) as object_group:
                        gr.Markdown("**Object Editing Options**")
                        with gr.Row():
                            object_select = gr.Gallery(label="Select for replacement", columns=[2], rows=[1],
                                                       object_fit="contain", height="auto", allow_preview=False)

                with gr.Column():
                    with gr.Group():
                        gr.Markdown("**Editing Result**")
                        all_viz = gr.Video(label="Boxes Visualization", interactive=False, loop=True, autoplay=True,
                                           show_download_button=False, show_share_button=False, format="mp4")
                        result_viz = gr.Video(label="Result Visualization", interactive=False, loop=True, autoplay=True,
                                              show_download_button=False, show_share_button=False, format="mp4")
                        run_btn = gr.Button(value="Generate")

                    with gr.Group():
                        gr.Markdown("**Editing Options**")
                        decoding_t = gr.Number(1, label="images to decode at one time", minimum=1, maximum=10, step=1)
                        seed_input = gr.Number(42, label="seed", minimum=1, maximum=1000, step=1)
                        result_draw_box = gr.Checkbox(label="Draw Box on Result", value=False)

                    gr.Markdown("""
                    **Notice:**
                    1. We provide pre-set object bounding boxes. For convenience, we use the first frame as a keyframe, 
                    so users can only modify the object position in subsequent frames.
                    2. When modifying the object position, it is necessary to set a reasonable and continuous position. 
                    Due to the video being shot within one second, significant changes may cause distortion.
                    """)

            def get_select_index(evt: gr.SelectData):
                self.selected_object_idx = evt.index

            object_select.select(get_select_index, None, None)

            @editing.select(inputs=[editing, box_id], outputs=[box_group, object_group, object_select,
                                                               all_viz, edited_viz, x_offset, y_offset, z_offset,
                                                               l_offset, w_offset, h_offset, yaw_offset])
            def update_editing(editing, box_id):
                return_val = reset_box_all(box_id, editing)
                if editing == "Deletion":
                    return gr.update(visible=False), gr.update(visible=False), self.data['im'], *return_val
                elif editing == "Insertion":
                    return gr.update(visible=True), gr.update(visible=True), self.data['im_Insertion'], *return_val
                elif editing == "Repositioning":
                    return gr.update(visible=True), gr.update(visible=False), self.data['im'], *return_val
                elif editing == "Replacement":
                    return gr.update(visible=False), gr.update(visible=True), self.data['im_Replacement'], *return_val

            # on load data
            @data_idx.input(inputs=[data_idx, editing], outputs=[all_viz, result_viz, box_id, object_select])
            def load_data(data_idx, edit):
                self.idx = data_idx
                self.data = self.video_data[self.idx]
                self.camera_intrinsic = self.data['camera_intrinsic']
                self.box, self.im, self.im_with_box, self.im_result = [], [], [], []
                self.selected_object_idx = 0
                for data in self.data['data']:
                    im = cv2.resize(data['im'], self.out_size[::-1], interpolation=cv2.INTER_LINEAR)
                    im_draw = np.copy(im)
                    if edit == "Deletion" or edit == "Replacement":
                        box = data['box']
                    else:
                        box = data[f'box_{edit}']
                    if box is not None:
                        corners = self.ratio * view_points(box.corners(), self.camera_intrinsic, True)[:2, :].T
                        draw_box(im_draw, corners, colors=(255, 158, 0))
                    self.im_with_box.append(im_draw)
                    self.im.append(im)
                    self.box.append(box)

                if edit in ("Replacement", "Insertion"):
                    select_data = self.data[f'im_{edit}']
                else:
                    select_data = self.data['im']

                return (concatenate_images(self.im_with_box),
                        None,
                        gr.update(choices=[i for i in range(1, len(self.box))], value=1),
                        select_data)

            # on load box info
            @box_id.input(inputs=[box_id],
                          outputs=[all_viz, edited_viz, x_offset, y_offset, z_offset, l_offset, w_offset, h_offset,
                                   yaw_offset])
            def load_box_info(box_id):
                box = self.box[box_id]
                if box is not None:
                    x, y, z = box.center
                    w, l, h = box.wlh
                    yaw = np.degrees(box.orientation.yaw_pitch_roll[0])
                else:
                    gr.Warning(f"There is no object in image {box_id}!")
                    x, y, z, w, l, h, yaw = 0, 0, 0, 0, 0, 0, 0
                return (
                    concatenate_images(self.im_with_box),
                    self.im_with_box[box_id],
                    gr.update(value=x, minimum=-30, maximum=30),
                    gr.update(value=y, minimum=-30, maximum=30),
                    gr.update(value=z, minimum=-30, maximum=30),
                    gr.update(value=l, minimum=0, maximum=10),
                    gr.update(value=w, minimum=0, maximum=10),
                    gr.update(value=h, minimum=0, maximum=10),
                    gr.update(value=yaw, minimum=-180, maximum=180)
                )

            def reset_box_func(box_id, edit):
                if edit == "Deletion" or edit == "Replacement":
                    box = self.data['data'][box_id]['box']
                else:
                    box = self.data['data'][box_id][f'box_{edit}']
                if box is not None:
                    self.box[box_id] = deepcopy(box)
                    x, y, z = box.center
                    w, l, h = box.wlh
                    yaw = np.degrees(box.orientation.yaw_pitch_roll[0])
                    im = np.copy(self.im[box_id])
                    corners = self.ratio * view_points(box.corners(), self.camera_intrinsic, normalize=True)[:2, :].T
                    draw_box(im, corners, colors=(255, 158, 0))
                    self.im_with_box[box_id] = im
                else:
                    x, y, z, w, l, h, yaw = 0, 0, 0, 0, 0, 0, 0
                return (
                    concatenate_images(self.im_with_box),
                    self.im_with_box[box_id],
                    gr.update(value=x, minimum=-30, maximum=30),
                    gr.update(value=y, minimum=-30, maximum=30),
                    gr.update(value=z, minimum=-30, maximum=30),
                    gr.update(value=l, minimum=0, maximum=10),
                    gr.update(value=w, minimum=0, maximum=10),
                    gr.update(value=h, minimum=0, maximum=10),
                    gr.update(value=yaw, minimum=-180, maximum=180)
                )

            @rst_anno_all_btn.click(inputs=[box_id, editing],
                                    outputs=[all_viz, edited_viz, x_offset, y_offset, z_offset, l_offset, w_offset,
                                             h_offset, yaw_offset])
            def reset_box_all(box_id, edit):
                return_value, return_val = None, None
                for i in range(0, len(self.box)):
                    return_val = reset_box_func(i, edit)
                    if i == box_id:
                        return_value = return_val
                if return_val is not None and return_value is not None:
                    return return_val[0], *return_value[1:]
                else:
                    return None, None, None, None, None, None, None, None, None

            @rst_anno_btn.click(inputs=[box_id, editing],
                                outputs=[all_viz, edited_viz, x_offset, y_offset, z_offset, l_offset, w_offset,
                                         h_offset, yaw_offset])
            def reset_box(box_id, edit):
                return reset_box_func(box_id, edit)

            def edit_and_show(xo, yo, zo, lo, wo, ho, yawo, box_id):
                box = self.data['data'][box_id]['box']
                if box is not None:
                    box = deepcopy(box)
                    box.center = np.array([float(xo), float(yo), float(zo)])
                    box.wlh = np.array([float(wo), float(lo), float(ho)])
                    yaw_ori, pitch_ori, roll_ori = box.orientation.yaw_pitch_roll
                    box.orientation = Quaternion(axis=[1, 0, 0], angle=roll_ori) * \
                                      Quaternion(axis=[0, 1, 0], angle=pitch_ori) * \
                                      Quaternion(axis=[0, 0, 1], angle=np.radians(float(yawo)))
                    self.box[box_id] = box
                    im = np.copy(self.im[box_id])
                    corners = self.ratio * view_points(box.corners(), self.camera_intrinsic, normalize=True)[:2, :].T
                    draw_box(im, corners, colors=(255, 158, 0))
                    self.im_with_box[box_id] = im
                else:
                    gr.Warning(f"There is no object in image {box_id}!")

                return concatenate_images(self.im_with_box), self.im_with_box[box_id]

            @run_btn.click(inputs=[decoding_t, result_draw_box, seed_input, editing],
                           outputs=[result_viz])
            def predict(decoding_t, result_draw_box_value, seed_input_value, editing):
                set_seed(int(seed_input_value))
                result = self.predict(decoding_t, result_draw_box_value, editing)
                return result

            @result_draw_box.input(inputs=[result_draw_box], outputs=[result_viz])
            def draw_box_on_result(result_draw_box):
                if result_draw_box:
                    vid = self.draw_box_on_result()
                    return concatenate_images(vid)
                else:
                    return concatenate_images(self.im_result)

            share_kwargs = {
                "inputs": [x_offset, y_offset, z_offset, l_offset, w_offset, h_offset, yaw_offset, box_id],
                "outputs": [all_viz, edited_viz],
            }
            x_offset.input(edit_and_show, **share_kwargs)
            y_offset.input(edit_and_show, **share_kwargs)
            z_offset.input(edit_and_show, **share_kwargs)
            l_offset.input(edit_and_show, **share_kwargs)
            w_offset.input(edit_and_show, **share_kwargs)
            h_offset.input(edit_and_show, **share_kwargs)
            yaw_offset.input(edit_and_show, **share_kwargs)

        demo.launch(server_name="0.0.0.0", server_port=7890)

    def predict(self, decoding_t, result_draw_box_value, editing):
        if editing == "Deletion":
            (cond_im_all, cond_im_3d, mask_all, mask_fuse_all, elevations_all, azimuths_all, indices_3d_all,
             position_all, obj_im_clip, box_im_all, valid_mask, obj_ratio) = self.get_deletion()
        else:
            (cond_im_all, cond_im_3d, mask_all, mask_fuse_all, elevations_all, azimuths_all, indices_3d_all,
             position_all, obj_im_clip, box_im_all, valid_mask, obj_ratio) = self.get_editing(mode=editing)

        H, W = cond_im_all.shape[2:]
        assert cond_im_all.shape[1] == 3
        shape = (self.num_frames, 4, H // 8, W // 8)
        shape_3d = (self.num_frames_3d, 4, 576 // 8, 576 // 8)

        value_dict = dict(
            motion_bucket_id=127,
            fps_id=10,
            cond_aug=torch.tensor(0.02).repeat(self.num_frames),
            cond_aug_3d=torch.tensor(1e-5).repeat(self.num_frames_3d),
            polars_rad_3d=elevations_all,
            azimuths_rad_3d=azimuths_all,
            indices_3d=indices_3d_all,
            obj_pos=position_all,
            cond_frames_without_noise=[cond_im_all[0:1], obj_im_clip.unsqueeze(0)],
            cond_frames=cond_im_all + torch.tensor(0.02) * torch.randn_like(cond_im_all),
            cond_frames_without_noise_3d=cond_im_3d,
            cond_frames_3d=cond_im_3d + torch.tensor(1e-5) * torch.randn_like(cond_im_3d),
            mask_concat=mask_all,
            mask_fuse=mask_fuse_all,
            depth=box_im_all,
        )

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(self.model.conditioner) +
                    get_unique_embedder_keys_from_conditioner(self.model.conditioner_3d),
                    value_dict,
                    [1, self.num_frames],
                    device=self.device0,
                )
                c, uc = self.model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                c_3d, uc_3d = self.model.conditioner_3d.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames_3d",
                        "cond_frames_without_noise_3d",
                    ],
                )

                for k in ["crossattn"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=self.num_frames)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=self.num_frames)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=self.num_frames)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=self.num_frames)

                for k in ["crossattn", "concat"]:
                    uc_3d[k] = repeat(uc_3d[k], "b ... -> b t ...", t=self.num_frames_3d)
                    uc_3d[k] = rearrange(uc_3d[k], "b t ... -> (b t) ...", t=self.num_frames_3d)
                    c_3d[k] = repeat(c_3d[k], "b ... -> b t ...", t=self.num_frames_3d)
                    c_3d[k] = rearrange(c_3d[k], "b t ... -> (b t) ...", t=self.num_frames_3d)

                for d in position_all:
                    for k, v in d.items():
                        if isinstance(v, torch.Tensor):
                            d[k] = v.to(self.device0)

                randn = torch.randn(shape, device=self.device0)
                randn_3d = torch.randn(shape_3d, device=self.device0)

                additional_model_inputs = {}
                additional_model_inputs["image_only_indicator"] = torch.zeros(2, self.num_frames).to(self.device0)
                additional_model_inputs["image_only_indicator_3d"] = torch.zeros(2, self.num_frames_3d).to(self.device0)
                additional_model_inputs["num_video_frames"] = self.num_frames
                additional_model_inputs["num_video_frames_3d"] = self.num_frames_3d
                additional_model_inputs["valid_mask"] = torch.concat([valid_mask] * 2).to(self.device0)
                additional_model_inputs["mask_fuse"] = torch.concat([mask_fuse_all] * 2).to(self.device0)
                additional_model_inputs["obj_ratio"] = torch.concat([obj_ratio] * 2).to(self.device0)
                additional_model_inputs['obj_pos'] = position_all * 2
                additional_model_inputs['indices_3d'] = torch.concat(
                    [
                        indices_3d_all,
                        indices_3d_all + self.num_frames_3d
                    ]
                ).to(self.device0)

                for d in (c, uc, c_3d, uc_3d):
                    for k, v in d.items():
                        if isinstance(v, torch.Tensor):
                            d[k] = v.to(self.device0)
                        elif isinstance(v, list):
                            d[k] = [vv.to(self.device0) for vv in v]

                def denoiser(input, sigma, c, input_3d, sigma_3d, c_3d):
                    return self.model.denoiser(
                        self.model.model, input, input_3d, sigma, sigma_3d, c, c_3d, **additional_model_inputs
                    )

                samples_z, _ = self.model.sampler(denoiser, randn, randn_3d,
                                                  cond=c, uc=uc, cond_3d=c_3d, uc_3d=uc_3d)

                self.model.en_and_decode_n_samples_a_time = decoding_t

                torch.cuda.empty_cache()
                samples_x = self.model.decode_first_stage(samples_z)
                samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                vid = (
                    (rearrange(samples, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )

                self.im_result = [np.ascontiguousarray(v) for v in vid]
                if result_draw_box_value:
                    vid = self.draw_box_on_result()
                return concatenate_images(vid)

    def get_deletion(self):
        cond_im_all, mask_all, position_all = [], [], []
        for frame_idx, (im, data) in enumerate(zip(self.im, self.data['data'])):
            # get target mask
            cond_im = self.to_tensor(np.copy(im)) * 2. - 1.

            box_ori = self.data['data'][frame_idx]['box']
            corners_ori = view_points(box_ori.corners(), self.camera_intrinsic, normalize=True)[:2, :].T
            pos = scale_bbox(corners_ori, (900, 1600), scale_x=1.9, scale_y=1.9)
            pos = pos.astype(np.int32)

            # target mask
            mask = np.zeros(im.shape[:2], dtype=np.float32)
            pos4mask = np.round((pos * self.ratio)).astype(int)
            mask[pos4mask[1]:pos4mask[3], pos4mask[0]:pos4mask[2]] = 1.

            pos4cond = np.round((pos * self.ratio)).astype(int)
            cond_im[:, pos4cond[1]:pos4cond[3], pos4cond[0]:pos4cond[2]] = \
                torch.zeros((3, pos4cond[3] - pos4cond[1], pos4cond[2] - pos4cond[0]))

            position_all.append({})
            cond_im_all.append(cond_im)
            mask_all.append(torch.tensor(mask).unsqueeze(0))

        obj_im_clip = torch.ones(3, 224, 224)
        cond_im_3d = torch.ones(1, 3, 576, 576)
        box_im_all = torch.ones(self.num_frames, 6, *self.out_size) * -1.

        # get other info
        cond_im_all = torch.stack(cond_im_all, dim=0)
        mask_all = self.transform_mask(torch.stack(mask_all, dim=0))
        mask_fuse_all = torch.zeros_like(mask_all)

        elevations_all = torch.zeros(self.num_frames_3d).unsqueeze(-1)
        azimuths_all = torch.zeros(self.num_frames_3d).unsqueeze(-1)
        indices_3d_all = torch.zeros(self.num_frames_3d).unsqueeze(-1)

        valid_mask = torch.zeros(self.num_frames)
        obj_ratio = torch.zeros(self.num_frames)

        return (cond_im_all, cond_im_3d, mask_all, mask_fuse_all, elevations_all, azimuths_all, indices_3d_all,
                position_all, obj_im_clip, box_im_all, valid_mask, obj_ratio)

    def get_editing(self, mode):
        video_cond_im, video_mask, video_mask_fuse = [], [], []
        video_enc_gt, video_cond_im_enc, video_position = [], [], []
        video_depth_concat, video_depth, video_depth_mask = [], [], []
        video_elevations, video_azimuths = [], []
        valid_mask = []
        keyframe_idx = 0

        if mode in ("Replacement", "Insertion"):
            obj_separate_im = self.data[f'im_{mode}'][self.selected_object_idx]
            obj_separate_mask = self.data[f'mask_{mode}'][self.selected_object_idx]
        elif mode == "Repositioning":
            if f'im_{mode}' in self.data:
                obj_separate_im = self.data[f'im_{mode}'][keyframe_idx]
                obj_separate_mask = self.data[f'mask_{mode}'][keyframe_idx]
            else:
                obj_separate_im = self.data['im']
                obj_separate_mask = self.data['mask']
        else:
            obj_separate_im = self.data['im']
            obj_separate_mask = self.data['mask']

        obj_separate_im = transforms.ToTensor()(obj_separate_im) * 2.0 - 1.0
        obj_separate_mask = (transforms.ToTensor()(obj_separate_mask) * 2.0 - 1.0).repeat(3, 1, 1)

        if mode not in ('Repositioning', 'Insertion'):
            box_keyframe = self.data['data'][keyframe_idx]['box']
        else:
            box_keyframe = self.box[keyframe_idx]
        _, azimuths = self._get_matrix(box_keyframe)
        corners = view_points(box_keyframe.corners(), self.camera_intrinsic, normalize=True)[:2, :].T
        pos_fuse = scale_bbox(corners, (900, 1600), 1.05, 1.05)
        indices = torch.argwhere(obj_separate_mask[0] > 0)
        (y_min, x_min), (y_max, x_max) = indices.min(dim=0).values, indices.max(dim=0).values
        h_obj, w_obj = y_max - y_min, x_max - x_min
        obj_ratio = h_obj / (pos_fuse[3] - pos_fuse[1])

        # get object image for clip condition
        obj_im_clip, _ = self._get_obj_im(obj_separate_im, obj_separate_mask, margin=32)

        # get object image for 3d model
        obj_scale = get_scale_by_obj_azimuth(azimuths)
        margin = 150 if 'movable_object' in self.data['category_name'] else 20
        cond_im_3d, obj_im_3d_shape = self._get_obj_im(obj_separate_im, obj_separate_mask, margin=margin,
                                                       out_size=576, scale_ratio=obj_scale)
        cond_im_3d = cond_im_3d.unsqueeze(0)

        obj_separate_im, obj_separate_mask = (
            get_obj_im_cond(obj_separate_im, obj_separate_mask, pos_fuse, scale=obj_ratio))

        obj_separate_im = self.transform_img_resize(obj_separate_im)
        obj_separate_mask = self.transform_img_resize(obj_separate_mask)
        obj_height = obj_im_3d_shape[0]

        for frame_idx, (im, box, data) in enumerate(zip(self.im, self.box, self.data['data'])):
            if mode not in ('Repositioning', 'Insertion'):
                box = self.data['data'][frame_idx]['box']

            valid_mask.append(1. if box is not None else 0.)

            # get 3d info
            elevations, azimuths = self._get_matrix(box)
            video_elevations.append(elevations)
            video_azimuths.append(azimuths)

            # get target mask
            mask = np.zeros((900, 1600), dtype=np.float32)
            mask_fuse = np.zeros((900, 1600), dtype=np.float32)
            cond_im = self.transform_img(np.copy(im))
            if box is not None:
                corners = view_points(box.corners(), self.camera_intrinsic, normalize=True)[:2, :].T
                if mode == 'Repositioning':
                    box_ori = self.data['data'][frame_idx]['box']
                    corners_ori = view_points(box_ori.corners(), self.camera_intrinsic, normalize=True)[:2, :].T
                    pos = scale_bbox(np.concatenate([corners, corners_ori]), (900, 1600), 1.3, 1.3)
                else:
                    pos = scale_bbox(corners, (900, 1600), 1.3, 1.3)
                pos_fuse = scale_bbox(corners, (900, 1600), 1.05, 1.05)

                # target mask
                mask[pos[1]:pos[3], pos[0]:pos[2]] = 1.
                mask_fuse[pos_fuse[1]:pos_fuse[3], pos_fuse[0]:pos_fuse[2]] = 1.

                # cond image
                pos4cond = np.round((pos * self.ratio)).astype(int)
                cond_im[:, pos4cond[1]:pos4cond[3], pos4cond[0]:pos4cond[2]] = \
                    torch.zeros((3, pos4cond[3] - pos4cond[1], pos4cond[2] - pos4cond[0]))
                if frame_idx == keyframe_idx:
                    cond_im[torch.where(obj_separate_mask > 0.)] = obj_separate_im[torch.where(obj_separate_mask > 0.)]

                # 3d info
                x_min, y_min = corners.min(axis=0)
                x_max, y_max = corners.max(axis=0)
                center_norm = np.array([(y_min + y_max) / 2, (x_min + x_max) / 2]) / np.array([900, 1600])

                video_position.append({'yx': torch.tensor(center_norm, dtype=torch.float32)[None, :],
                                       'box_height': torch.tensor(pos_fuse[3] - pos_fuse[1]) * self.ratio,
                                       'obj_height': torch.tensor(obj_height),
                                       })

                box_im = process_box(box.corners().T, corners, self.camera_intrinsic)
                box_im = torch.concat([self.transform_depth(b.astype(np.float32)) for b in box_im])
            else:
                video_position.append({})
                box_im = torch.ones(self.out_size).unsqueeze(0).repeat(6, 1, 1) * -1.

            video_cond_im.append(cond_im)
            video_mask.append(torch.tensor(mask).unsqueeze(0))
            video_mask_fuse.append(torch.tensor(mask_fuse).unsqueeze(0))
            video_depth.append(box_im)

        # set missing values
        v = None
        video_elevations = [v := x if x is not None else v for x in video_elevations]
        video_elevations = [v := x if x is not None else v for x in video_elevations[::-1]][::-1]
        video_elevations = torch.tensor(video_elevations)
        v = None
        video_azimuths = [v := x if x is not None else v for x in video_azimuths]
        video_azimuths = [v := x if x is not None else v for x in video_azimuths[::-1]][::-1]
        video_azimuths = torch.tensor(video_azimuths)

        # get other info
        video_cond_im = torch.stack(video_cond_im, dim=0)
        video_mask = self.transform_mask(torch.stack(video_mask, dim=0))
        video_mask_fuse = self.transform_mask(torch.stack(video_mask_fuse, dim=0))
        video_depth = torch.stack(video_depth, dim=0)
        valid_mask = torch.tensor(valid_mask)

        video_elevations, video_azimuths, video_3d_indices = (
            self._get_3d_full(video_elevations, video_azimuths, keyframe_idx))

        return (
            video_cond_im, cond_im_3d, video_mask, video_mask_fuse, video_elevations, video_azimuths, video_3d_indices,
            video_position, obj_im_clip, video_depth, valid_mask, obj_ratio.repeat(self.num_frames))

    def draw_box_on_result(self):
        vid = []
        for im, box in zip(self.im_result, self.box):
            if box is not None:
                corners = view_points(box.corners(), self.camera_intrinsic, normalize=True)[:2, :].T
                corners *= self.ratio
                im_draw = np.copy(im)
                draw_box(im_draw, corners, colors=(255, 158, 0))
            else:
                im_draw = im
            vid.append(im_draw)
        return vid

    def _get_matrix(self, box):
        elevations, azimuths = None, None
        if box is not None:
            sensor_rotation_object = box.orientation.rotation_matrix.T
            sensor_translation_object = -np.dot(sensor_rotation_object, box.center)

            # get elevation, azimuth, distance and view_angle
            elevations = np.arcsin(sensor_translation_object[2] / np.linalg.norm(sensor_translation_object))
            azimuths = np.arctan2(sensor_translation_object[1], sensor_translation_object[0])
            elevations = np.deg2rad(90) - elevations
            azimuths = np.deg2rad(180) + azimuths

        return elevations, azimuths

    def _get_3d_full(self, polars_rad, azimuths_rad, cond_idx):
        diff = azimuths_rad - azimuths_rad[cond_idx]
        diff = torch.where(diff < 0, diff + 2 * np.pi, diff)

        sample_azimuths_deg = [3., 6., 9., 12., 16., 23., 30., 45., 90., 135.,
                               225., 270., 315, 330., 337., 344., 348., 351., 354., 357., 0]
        sample_azimuths_rad = torch.tensor([np.deg2rad(a % 360) for a in sample_azimuths_deg])

        indices = torch.argmin(torch.abs(diff[:, None] - sample_azimuths_rad), dim=1).to(torch.int32)

        polars_rad = torch.ones(self.num_frames_3d) * torch.mean(polars_rad)
        return polars_rad, sample_azimuths_rad, indices

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


def draw_box(im, corners, colors=None, thickness=2):
    if colors is None:
        colors = (255, 255, 255)

    def draw_rect(selected_corners):
        prev = selected_corners[-1]
        for corner in selected_corners:
            cv2.line(im, [int(prev[0]), int(prev[1])],
                     [int(corner[0]), int(corner[1])], colors, thickness=thickness)
            prev = corner

    # Draw the sides
    for i in range(4):
        cv2.line(im, [int(corners[i][0]), int(corners[i][1])],
                 [int(corners[i + 4][0]), int(corners[i + 4][1])], colors, thickness=thickness)

    # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
    draw_rect(corners[:4])
    draw_rect(corners[4:])

    center_bottom_forward = np.mean(corners[2:4], axis=0)
    center_bottom = np.mean(corners[[2, 3, 7, 6]], axis=0)
    cv2.line(im, [int(center_bottom[0]), int(center_bottom[1])],
             [int(center_bottom_forward[0]), int(center_bottom_forward[1])], colors, thickness=thickness)


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


def concatenate_images(images):
    height, width = images[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_filename = os.path.join(tempfile.gettempdir(), 'output.mp4')
    out = cv2.VideoWriter(video_filename, fourcc, 7, (width, height))
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        out.write(img)

    out.release()
    return video_filename


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


def load_model(
        config: str,
        device: str,
        num_steps: int,
        num_frames: int,
        verbose: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    model = instantiate_from_config(config.model).to(device).eval()

    return model


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_frames" or key == "cond_frames_3d":
            batch[key] = value_dict[key].to(device)
        elif key == "cond_frames_without_noise":
            batch[key] = [v.to(device) for v in value_dict["cond_frames_without_noise"]]
        else:
            batch[key] = value_dict[key].to(device)

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
        elif key not in batch_uc and isinstance(batch[key], list):
            batch_uc[key] = [torch.clone(v) for v in batch[key]]
    return batch, batch_uc


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def process_box(corners, corners_2d, camera_intrinsic):
    def draw_line(im):
        def draw_rect(selected_corners, selected_colors):
            prev, prev_color = selected_corners[-1], selected_colors[-1]
            for corner, corner_color in zip(selected_corners, selected_colors):
                draw_line(prev, corner, prev_color, corner_color)
                prev, prev_color = corner, corner_color

        def draw_line(start, end, color_start, color_end, num_points: int = 250, thickness: int = 2):
            step_distance = (end - start) / num_points
            step_color = (color_end - color_start) / num_points

            current_point = start
            current_color = color_start
            for _ in range(num_points):
                next_point = current_point + step_distance
                color = current_color + step_color
                cv2.line(im, tuple(current_point.astype(int)), tuple(next_point.astype(int)), color.item(), thickness)
                current_point = next_point
                current_color = color

        # Draw the sides
        for i in range(4):
            draw_line(corners_2d[i], corners_2d[i + 4], corners[i, 2], corners[i + 4, 2])

        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        draw_rect(corners_2d[:4], corners[:4, 2])
        draw_rect(corners_2d[4:], corners[4:, 2])

    im_size = (800, 450)
    corners_2d /= 2  # downsample
    corners_2d = corners_2d.astype(np.int32)
    depth_map_list = []

    for i, face in enumerate(CUBE_FACES):
        corners_2d_face = corners_2d[face]
        corners_3d_face = corners[face]

        depth_map = np.zeros(im_size[::-1], dtype=np.float32)

        polygon_mask = np.zeros(im_size[::-1], dtype=np.uint8)
        cv2.fillPoly(polygon_mask, [corners_2d_face], 255)
        coords = np.column_stack(np.where(polygon_mask == 255))[:, ::-1]

        normal = np.cross(corners_3d_face[1] - corners_3d_face[0], corners_3d_face[2] - corners_3d_face[0])
        d_coeff = -np.dot(normal, corners_3d_face[0])

        a = camera_intrinsic[0, 0]
        b = camera_intrinsic[1, 1]
        c = camera_intrinsic[0, 2]
        d = camera_intrinsic[1, 2]
        m, n, o = normal
        k = coords[:, 0] * 2  # downsample
        l = coords[:, 1] * 2  # downsample
        z = d_coeff / (m * (c - k) / a + n * (d - l) / b - o)
        x = -(c - k) / a * z
        y = -(d - l) / b * z
        intersection_points = np.column_stack((x, y, z))

        depth_map[coords[:, 1], coords[:, 0]] = intersection_points[:, 2]
        draw_line(depth_map)
        depth_map = (depth_map * 256.).astype(np.uint16)
        depth_map_list.append(depth_map)

    return depth_map_list


if __name__ == "__main__":
    show = GradioShow()
    show.run()

# DriveEditor
A diffusion-based unified framework capable of repositioning, inserting, replacing, and deleting objects in driving scenario videos.

![sample](assets/figure.png)

[Arxiv](https://arxiv.org/abs/2412.19458) | [Project Page](https://yvanliang.github.io/DriveEditor/)

## :gear: Installation
<a name="installation"></a>
### 1. Clone this repo.
 ```shell
git clone git@github.com:yvanliang/DriveEditor.git
 ```
### 2. Set up the conda environment for inference.
 ```shell
 conda create -n DriveEditor python=3.10 -y
 conda activate DriveEditor
 pip install torch==2.1.1 torchvision==0.16.1 xformers==0.0.23 --index-url https://download.pytorch.org/whl/cu118
 pip install -r requirements.txt
 pip install .
 ```

## :nut_and_bolt: Model Preparation
Download the pretrained models from [Google Drive](https://drive.google.com/file/d/1QuDnHjMS6KYwq-HglzzCSucc1HZaDW3J/edit) and place it in the `checkpoints` directory.

## :zap: Use This model
1. Download the demo data from [Google Drive](https://drive.google.com/file/d/16fmFFqW-OKzZH3S95VHHY0j45Hqn4p0B/edit) and place it in the `checkpoints` directory.


2. We provide a gradio demo for editing driving scenario videos. To run the demo, a GPU with more than 32GB of VRAM is required. Execute the following command:

    ```
    python interactive_gui.py
    ```

3. If you don't have a GPU with more than 32GB of VRAM but have two 24GB VRAM GPUs, you can use both GPUs for inference, although it will take more time. First, modify `sgm/modules/diffusionmodules/video_model.py`:

   - At line 684, add:

     ```python
     h_out_3d = h_out_3d.to(x.device)
     hs_3d_all = [t.to(x.device) for t in hs_3d_all]
     ```

    - At line 794, add:

      ```python
      x = x.to("cuda:1")
      timesteps = timesteps.to("cuda:1")
      context = context.to("cuda:1")
      y = y.to("cuda:1")
      if time_context is not None:
          time_context = time_context.to("cuda:1")
      image_only_indicator = image_only_indicator.to("cuda:1")
      ```

   - Then run the following command:
       ```
       python interactive_gui_2gpu.py
       ```

## :star2: Acknowledgements
We appreciate the releasing code of [Stable Video Diffusion](https://github.com/Stability-AI/generative-models) and [ChatSim](https://github.com/yifanlu0227/ChatSim).

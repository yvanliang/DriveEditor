import torch
from safetensors.torch import load_file, save_file

model_3d_path = 'checkpoints/sv3d_p.safetensors'
ckpt_file_3d = load_file(model_3d_path)
model_path = 'checkpoints/svd.safetensors'
ckpt_file = load_file(model_path)

ckpt_new = dict()
for k, v in ckpt_file.items():
    if 'conditioner.embedders.0.open_clip' in k:
        ckpt_new[k.replace('conditioner.embedders.0.open_clip', 'conditioner.embedders.0.model.open_clip')] = v
    else:
        ckpt_new[k] = v

for k, v in ckpt_file_3d.items():
    if 'label_emb.' in k:
        ckpt_new[k.replace('label_emb.', 'label_emb_3d.')] = v
    elif 'time_embed.' in k:
        ckpt_new[k.replace('time_embed.', 'time_embed_3d.')] = v
    elif 'model.diffusion_model.input_blocks.' in k:
        ckpt_new[k.replace('model.diffusion_model.input_blocks.', 'model.diffusion_model.input_blocks_3d.')] = v
    elif 'model.diffusion_model.middle_block.' in k:
        ckpt_new[k.replace('model.diffusion_model.middle_block.', 'model.diffusion_model.middle_block_3d.')] = v
    elif 'model.diffusion_model.output_blocks.' in k:
        ckpt_new[k.replace('model.diffusion_model.output_blocks.', 'model.diffusion_model.output_blocks_3d.')] = v
    elif 'model.diffusion_model.out.' in k:
        ckpt_new[k.replace('model.diffusion_model.out.', 'model.diffusion_model.out_3d.')] = v

k = 'model.diffusion_model.input_blocks.0.0.weight'
ckpt_new[k] = torch.cat((ckpt_file[k], torch.zeros((320, 1, 3, 3))), dim=1).to(ckpt_file[k].dtype)

save_file(ckpt_new, 'checkpoints/svd_3d_9input.safetensors')

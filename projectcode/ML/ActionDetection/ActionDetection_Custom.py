import torch

ckpt = torch.load("/opt/epic-kitchens/C1-Action-Recognition-TSN-TRN-TSM/models/tsm_rgb.ckpt", map_location="cpu")
print(ckpt.keys())  # see what’s inside

hp = ckpt["hyper_parameters"]
print(hp.keys())

print(hp["model"])

print(hp["data"])

print(hp["learning"])

print(hp["trainer"])

state_dict = ckpt["state_dict"]

fc_weight_key = [k for k in state_dict.keys() if "fc" in k and "weight" in k][-1]
print(fc_weight_key, state_dict[fc_weight_key].shape)
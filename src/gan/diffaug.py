import torch
import torch.nn.functional as F

AUG_POLICIES = ["color", "translation", "cutout"]

def rand_brightness(x): return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)
def rand_saturation(x):
    x_mean = x.mean(dim=1, keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) * 2) + x_mean
def rand_contrast(x):
    x_mean = x.mean(dim=(1,2,3), keepdim=True)
    return (x - x_mean) * (torch.rand(x.size(0), 1, 1, 1, device=x.device) + 0.5) + x_mean

def color(x):
    for fn in (rand_brightness, rand_saturation, rand_contrast):
        x = fn(x)
    return x

def translation(x, ratio=0.125):
    B, C, H, W = x.shape
    shift_x = int(H * ratio + 0.5)
    shift_y = int(W * ratio + 0.5)
    tx = torch.randint(-shift_x, shift_x + 1, [B, 1, 1], device=x.device)
    ty = torch.randint(-shift_y, shift_y + 1, [B, 1, 1], device=x.device)
    grid_batch, grid_x, grid_y = torch.meshgrid(
        torch.arange(B, device=x.device), torch.arange(H, device=x.device), torch.arange(W, device=x.device), indexing="ij")
    grid_x = torch.clamp(grid_x + tx, 0, H - 1)
    grid_y = torch.clamp(grid_y + ty, 0, W - 1)
    x = x[grid_batch, :, grid_x, grid_y]
    return x

def cutout(x, ratio=0.5):
    B, C, H, W = x.shape
    cut_h, cut_w = int(H * ratio + 0.5), int(W * ratio + 0.5)
    cy = torch.randint(0, H, [B, 1, 1], device=x.device)
    cx = torch.randint(0, W, [B, 1, 1], device=x.device)
    yl, xl = torch.clamp(cy - cut_h // 2, 0, H), torch.clamp(cx - cut_w // 2, 0, W)
    yu, xu = torch.clamp(yl + cut_h, 0, H), torch.clamp(xl + cut_w, 0, W)
    for i in range(B):
        x[i, :, yl[i,0,0]:yu[i,0,0], xl[i,0,0]:xu[i,0,0]] = 0
    return x

def diff_augment(x, policies=AUG_POLICIES):
    for p in policies:
        if p == "color":
            x = color(x)
        elif p == "translation":
            x = translation(x)
        elif p == "cutout":
            x = cutout(x)
    return x

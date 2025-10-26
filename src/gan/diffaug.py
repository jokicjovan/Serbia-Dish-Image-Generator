import torch
import torch.nn.functional as F

AUG_POLICIES = ["color", "translation", "cutout"]

def rand_brightness(x): 
    return x + (torch.rand(x.size(0), 1, 1, 1, device=x.device) - 0.5)

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
    
    translation_x = torch.randint(-shift_x, shift_x + 1, size=(B,), device=x.device)
    translation_y = torch.randint(-shift_y, shift_y + 1, size=(B,), device=x.device)
    
    grid_x = torch.arange(W, dtype=torch.float32, device=x.device).view(1, 1, W).repeat(B, H, 1)
    grid_y = torch.arange(H, dtype=torch.float32, device=x.device).view(1, H, 1).repeat(B, 1, W)
    
    grid_x = grid_x + translation_x.view(B, 1, 1)
    grid_y = grid_y + translation_y.view(B, 1, 1)
    
    grid_x = 2.0 * grid_x / (W - 1) - 1.0
    grid_y = 2.0 * grid_y / (H - 1) - 1.0
    
    grid = torch.stack([grid_x, grid_y], dim=-1)
    x_translated = F.grid_sample(x, grid, mode='bilinear', padding_mode='border', align_corners=True)
    
    return x_translated

def cutout(x, ratio=0.5):
    B, C, H, W = x.shape
    cut_h, cut_w = int(H * ratio + 0.5), int(W * ratio + 0.5)
    mask = torch.ones_like(x)
    
    for i in range(B):
        cy = torch.randint(0, H, (1,), device=x.device).item()
        cx = torch.randint(0, W, (1,), device=x.device).item()
        y1 = max(0, cy - cut_h // 2)
        y2 = min(H, y1 + cut_h)
        x1 = max(0, cx - cut_w // 2)
        x2 = min(W, x1 + cut_w)
        mask[i, :, y1:y2, x1:x2] = 0
    
    return x * mask

def diff_augment(x, policies=AUG_POLICIES):
    for p in policies:
        if p == "color": x = color(x)
        elif p == "translation": x = translation(x)
        elif p == "cutout": x = cutout(x)
    return x

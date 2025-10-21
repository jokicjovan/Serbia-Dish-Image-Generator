import torch, torch.nn as nn, torch.nn.functional as F

def snlinear(in_f, out_f):
    return nn.utils.spectral_norm(nn.Linear(in_f, out_f))
def snconv(in_c, out_c, k, s=1, p=0):
    return nn.utils.spectral_norm(nn.Conv2d(in_c, out_c, k, s, p))

class CondBN(nn.Module):
    def __init__(self, ch, cond_dim):
        super().__init__()
        self.bn = nn.BatchNorm2d(ch, affine=False)
        self.gam = nn.Linear(cond_dim, ch)
        self.bet = nn.Linear(cond_dim, ch)
    def forward(self, x, y):  # y: [B, cond_dim]
        h = self.bn(x)
        g = self.gam(y).unsqueeze(-1).unsqueeze(-1)
        b = self.bet(y).unsqueeze(-1).unsqueeze(-1)
        return h * (1 + g) + b

class ResBlockG(nn.Module):
    def __init__(self, in_c, out_c, cond_dim, up=True):
        super().__init__()
        self.up = up
        self.cbn1 = CondBN(in_c, cond_dim)
        self.cbn2 = CondBN(out_c, cond_dim)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.skip = nn.Conv2d(in_c, out_c, 1) if in_c!=out_c else nn.Identity()
    def forward(self, x, y):
        h = self.cbn1(x, y); h = F.relu(h)
        if self.up: h = F.interpolate(h, scale_factor=2, mode="nearest")
        h = self.conv1(h)
        h = self.cbn2(h, y); h = F.relu(h); h = self.conv2(h)
        s = x
        if self.up: s = F.interpolate(s, scale_factor=2, mode="nearest")
        s = self.skip(s)
        return h + s

class Generator(nn.Module):
    def __init__(self, z_dim=128, cond_in=768, cond_hidden=256, base_ch=64, out_size=128):
        super().__init__()
        self.cond = nn.Sequential(
            nn.Linear(cond_in, 512), nn.ReLU(True),
            nn.Linear(512, cond_hidden)
        )
        self.fc = nn.Linear(z_dim + cond_hidden, 4*4*base_ch*16)
        ch = base_ch*16
        blocks, size = [], 4
        while size < out_size:
            blocks.append(ResBlockG(ch, ch//2, cond_dim=cond_hidden, up=True))
            ch //= 2; size *= 2
        self.blocks = nn.ModuleList(blocks)
        self.bn = nn.BatchNorm2d(ch, affine=True)
        self.conv_out = nn.Conv2d(ch, 3, 3, 1, 1)

    def forward(self, z, e):
        y = self.cond(e)                       # [B, cond_hidden]
        h = self.fc(torch.cat([z, y], dim=1))  # [B, 4*4*C]
        ch = h.shape[1] // (4*4)
        h = h.view(-1, ch, 4, 4)
        for b in self.blocks: h = b(h, y)
        h = F.relu(self.bn(h))
        x = torch.tanh(self.conv_out(h))
        return x

class ResBlockD(nn.Module):
    def __init__(self, in_c, out_c, down=True):
        super().__init__()
        self.down = down
        self.conv1 = snconv(in_c, out_c, 3, 1, 1)
        self.conv2 = snconv(out_c, out_c, 3, 1, 1)
        self.skip = snconv(in_c, out_c, 1) if in_c!=out_c else nn.Identity()
    def forward(self, x):
        h = F.relu(x); h = self.conv1(h)
        h = F.relu(h); h = self.conv2(h)
        s = x
        if self.down:
            h = F.avg_pool2d(h, 2)
            s = F.avg_pool2d(s, 2)
        s = self.skip(s)
        return h + s

class Discriminator(nn.Module):
    def __init__(self, cond_in=768, base_ch=64, in_size=128):
        super().__init__()
        ch = base_ch
        blocks = [ResBlockD(3, ch, down=True)]
        size, c = in_size, ch
        while size > 4:
            blocks.append(ResBlockD(c, c*2, down=True))
            c *= 2; size //= 2
        self.blocks = nn.ModuleList(blocks)
        self.conv_out = snconv(c, c, 3, 1, 1)
        self.lin = snlinear(c, 1)
        self.embed = snlinear(cond_in, c)   # projection term

    def forward(self, x, e):
        h = x
        for b in self.blocks: h = b(h)
        h = F.relu(self.conv_out(h))
        h = torch.sum(h, dim=(2,3))          # global sum pooling
        out = self.lin(h) + torch.sum(self.embed(e) * h, dim=1, keepdim=True)
        return out

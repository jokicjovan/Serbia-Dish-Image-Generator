import torch

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    @torch.no_grad()
    def update(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert name in self.shadow
                new_avg = self.shadow[name] * self.decay + p.data * (1.0 - self.decay)
                self.shadow[name] = new_avg.clone()

    def apply_shadow(self, model):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.backup[name] = p.data.clone()
                p.data = self.shadow[name].clone()

    def copy_to(self, model):
        """Copy shadow weights to model without creating backup."""
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.shadow[name].clone()

    def restore(self, model):
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.data = self.backup[name].clone()
        self.backup = {}

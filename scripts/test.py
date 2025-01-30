import torch
from mmseg.models import build_loss

# 1. Define your loss config (match your MMSegmentation config)
loss_cfg = dict(
    type='CrossEntropyLoss',
    use_sigmoid=False,  # Use softmax for 2 classes
    class_weight=[1.0, 9.0],  # [background, dog]
)

# 2. Build the loss function
loss_fn = build_loss(loss_cfg)

# 3. Generate synthetic data (GPU)
fake_labels = torch.randint(0, 2, (2, 512, 512)).cuda()  # (B, H, W) with class indices 0/1
fake_logits = torch.randn(2, 2, 512, 512).cuda()  # (B, num_classes, H, W)

# 4. Compute loss
loss = loss_fn(fake_logits, fake_labels)
print(f"Loss value: {loss.item()}")  # Should print a float without crashing

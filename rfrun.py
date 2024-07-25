

import torch
from dit import DiT_Llama
from rf import RF
from torchvision.utils import make_grid
from PIL import Image
import numpy as np

channels=1
num_classes=4
model = DiT_Llama(channels, 32, dim=256, n_layers=1, n_heads=8, num_classes=num_classes).cuda()
model.load_state_dict(torch.load(f'model_mydata_10500_final.pth'))
rf = RF(model)
rf.model.eval()
with torch.no_grad():
    cond = torch.arange(0, 16).cuda() % num_classes
    uncond = torch.ones_like(cond) * num_classes

    init_noise = torch.randn(16, channels, 32, 32).cuda()
    images = rf.sample(init_noise, cond, uncond)
    # image sequences to gif
    gif = []
    for image in images:
        # unnormalize
        image = image * 0.5 + 0.5
        image = image.clamp(0, 1)
        x_as_image = make_grid(image.float(), nrow=4)
        img = x_as_image.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        gif.append(Image.fromarray(img))

    gif[0].save(
        f"contents/sample_ttt.gif",
        save_all=True,
        append_images=gif[1:],
        duration=100,
        loop=0,
    )

    last_img = gif[-1]
    last_img.save(f"contents/sample_ttt_last.png")    
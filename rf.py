# implementation of Rectified Flow for simple minded people like me.
import argparse

import torch
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = ImageFolder(root, transform=transform)
        
    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, label
    
    def __len__(self):
        return len(self.dataset)
    

class RF:
    def __init__(self, model, ln=True):
        self.model = model
        self.ln = ln

    def forward(self, x, cond):
        b = x.size(0)
        if self.ln:
            nt = torch.randn((b,)).to(x.device)
            t = torch.sigmoid(nt)
        else:
            t = torch.rand((b,)).to(x.device)
        texp = t.view([b, *([1] * len(x.shape[1:]))])
        z1 = torch.randn_like(x)
        zt = (1 - texp) * x + texp * z1
        vtheta = self.model(zt, t, cond)
        batchwise_mse = ((z1 - x - vtheta) ** 2).mean(dim=list(range(1, len(x.shape))))
        tlist = batchwise_mse.detach().cpu().reshape(-1).tolist()
        ttloss = [(tv, tloss) for tv, tloss in zip(t, tlist)]
        return batchwise_mse.mean(), ttloss

    @torch.no_grad()
    def sample(self, z, cond, null_cond=None, sample_steps=50, cfg=2.0):
        b = z.size(0)
        dt = 1.0 / sample_steps
        dt = torch.tensor([dt] * b).to(z.device).view([b, *([1] * len(z.shape[1:]))])
        images = [z]
        for i in range(sample_steps, 0, -1):
            t = i / sample_steps
            t = torch.tensor([t] * b).to(z.device)

            vc = self.model(z, t, cond)
            if null_cond is not None:
                vu = self.model(z, t, null_cond)
                vc = vu + cfg * (vc - vu)

            z = z - dt * vc
            images.append(z)
        return images


if __name__ == "__main__":
    # train class conditional RF on mnist.
    import numpy as np
    import torch.optim as optim
    from PIL import Image
    from torch.utils.data import DataLoader
    from torchvision import datasets, transforms
    from torchvision.utils import make_grid
    from tqdm import tqdm

    import wandb
    from dit import DiT_Llama

    parser = argparse.ArgumentParser(description="use cifar?")
    parser.add_argument("--cifar", action="store_true")
    args = parser.parse_args()
    CIFAR = args.cifar
    #MYDATA = args.mydata
    MYDATA = True

    if MYDATA:
        dataset_name = "mydata"
        channels=1
        # 定义转换
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  # 确保所有图像都是 32x32
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize((0.5,), (0.5,))
        ])        
        # 使用自定义数据集
        custom_dataset = CustomDataset(root="data/mydata", transform=transform)
        dataloader = DataLoader(custom_dataset, batch_size=1, shuffle=True, drop_last=True)        
        # 获取类别数量
        num_classes = len(custom_dataset.dataset.classes)
        # 更新模型的类别数
        model = DiT_Llama(
            channels, 32, dim=256, n_layers=1, n_heads=8, num_classes=num_classes
        ).cuda()        
    elif CIFAR:
        dataset_name = "cifar"
        num_classes=10
        fdatasets = datasets.CIFAR10
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.RandomCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
        dataloader = DataLoader(mnist, batch_size=256, shuffle=True, drop_last=True)

        channels = 3
        model = DiT_Llama(
            channels, 32, dim=256, n_layers=10, n_heads=8, num_classes=10
        ).cuda()

    else:
        dataset_name = "mnist"
        num_classes=10
        fdatasets = datasets.MNIST
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Pad(2),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        mnist = fdatasets(root="./data", train=True, download=True, transform=transform)
        dataloader = DataLoader(mnist, batch_size=256, shuffle=True, drop_last=True)
        channels = 1
        model = DiT_Llama(
            channels, 32, dim=64, n_layers=6, n_heads=4, num_classes=10
        ).cuda()

    model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {model_size}, {model_size / 1e6}M")

    rf = RF(model)
    optimizer = optim.Adam(model.parameters(), lr=5e-4)
    criterion = torch.nn.MSELoss()

    #wandb.init(project=f"rf_{dataset_name}")

    for epoch in range(200):
        lossbin = {i: 0 for i in range(10)}
        losscnt = {i: 1e-6 for i in range(10)}
        for i, (x, c) in tqdm(enumerate(dataloader)):
            x, c = x.cuda(), c.cuda()
            optimizer.zero_grad()
            loss, blsct = rf.forward(x, c)
            loss.backward()
            optimizer.step()

            #wandb.log({"loss": loss.item()})

            # count based on t
            for t, l in blsct:
                lossbin[int(t * 10)] += l
                losscnt[int(t * 10)] += 1

        # log
        for i in range(10):
            print(f"Epoch: {epoch}, {i} range loss: {lossbin[i] / losscnt[i]}")

        #wandb.log({f"lossbin_{i}": lossbin[i] / losscnt[i] for i in range(10)})

        #一个epoch完了，评估一下
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
                f"contents/sample_{epoch}.gif",
                save_all=True,
                append_images=gif[1:],
                duration=100,
                loop=0,
            )

            last_img = gif[-1]
            last_img.save(f"contents/sample_{epoch}_last.png")

        rf.model.train()

    # 训练结束后保存最终模型
    torch.save(model.state_dict(), f'model_{dataset_name}_final.pth')
    print(f"Final model saved as model_{dataset_name}_final.pth")

    #test
    rf.model.eval()
    with torch.no_grad():
        cond = torch.arange(0, 16).cuda() % num_classes
        cond = torch.tensor([0,0,0,0,1,1,1,1,0,0,0,0,1,1,1,1]).cuda()
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
            f"contents/sample_test.gif",
            save_all=True,
            append_images=gif[1:],
            duration=100,
            loop=0,
        )

        last_img = gif[-1]
        last_img.save(f"contents/sample_test_last.png")


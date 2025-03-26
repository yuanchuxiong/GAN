# 1. 必要なライブラリをインポート
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime

# 2. Generator（生成器）の定義
class Generator(nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            # 1. 1x1 -> 4x4
            nn.ConvTranspose2d(z_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(True),

            # 2. 4x4 -> 8x8
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            # 3. 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            # 4. 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            # 5. 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            # 6. 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, 4, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            # 7. 128x128 -> 128x128（詳細を強調）
            nn.ConvTranspose2d(32, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(-1, 100, 1, 1)  # (batch_size, z_dim, 1, 1)
        return self.model(z)

# 3. Discriminator（識別器）の定義
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            # 1. 128x128 -> 64x64
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # 2. 64x64 -> 32x32
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            # 3. 32x32 -> 16x16
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            # 4. 16x16 -> 8x8
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            # 5. 8x8 -> 4x4
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True),

            # 6. 4x4 -> 1x1（確率を出力）
            nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x).view(-1)  # 出力は (batch_size,) を保証

# 4. データの読み込みと前処理（128x128にリサイズ）
transform = transforms.Compose([
    transforms.Resize(128),  # 高解像度に適応
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

# 5. 学習パラメータの設定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 使用するデバイス情報を表示
if torch.cuda.is_available():
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    print("Using CPU")

z_dim = 100
epochs = 300
lr = 0.0002

# 6. モデルの初期化
gen = Generator(z_dim=z_dim).to(device)
disc = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_g = optim.Adam(gen.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))

# 7. 学習ループ
for epoch in range(epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)

        # ====== Discriminatorの学習 ======
        optimizer_d.zero_grad()
        output_real = disc(real_images)
        loss_real = criterion(output_real, torch.ones_like(output_real))

        noise = torch.randn(real_images.size(0), z_dim, 1, 1, device=device)
        fake_images = gen(noise)
        output_fake = disc(fake_images.detach())  # detach() でGeneratorへの影響を避ける
        loss_fake = criterion(output_fake, torch.zeros_like(output_fake))

        loss_d = loss_real + loss_fake
        loss_d.backward()
        optimizer_d.step()

        # ====== Generatorの学習（計算グラフの問題を修正） ======
        for _ in range(2):  # Generatorを2回学習
            optimizer_g.zero_grad()
            noise = torch.randn(real_images.size(0), z_dim, 1, 1, device=device)
            fake_images = gen(noise)
            output_fake = disc(fake_images)
            loss_g = criterion(output_fake, torch.ones_like(output_fake))
            loss_g.backward()
            optimizer_g.step()

    # ログの出力
    current_time = datetime.now().strftime("%Y/%m/%d %H:%M:%S")
    print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f} | {current_time}")

    # 50エポックごとに画像を保存
    if (epoch + 1) % 50 == 0:
        with torch.no_grad():
            noise = torch.randn(16, z_dim, 1, 1, device=device)
            fake_images = gen(noise).cpu().detach()
            fake_images = (fake_images + 1) / 2
            grid = make_grid(fake_images, nrow=4)

            fig, ax = plt.subplots(figsize=(4, 4))
            ax.imshow(np.transpose(grid.numpy(), (1, 2, 0)))
            ax.axis("off")
            plt.show()

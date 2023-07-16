import torch
from pytorch_msssim import ssim, ms_ssim
from PIL import Image
import torchvision.transforms as TF
import pathlib
from tqdm import tqdm

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        path = self.files[i]
        img = Image.open(path).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)
        return path.name, img

device = torch.device('cuda')
transform = TF.Compose([
    TF.Resize((16, 16)),
    TF.ToTensor(),
])
real_path = pathlib.Path('all_images')
gan_path = pathlib.Path('gan_images')
log_path = 'ssim.log'

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm',
                    'tif', 'tiff', 'webp'}

real_files = sorted([file for ext in IMAGE_EXTENSIONS
                for file in real_path.glob('*.{}'.format(ext))])
real_dataset = ImagePathDataset(real_files, transforms=transform)
real_loader = torch.utils.data.DataLoader(real_dataset,
    batch_size=1000,
    shuffle=False,  
    drop_last=False,
)
real_images = torch.cat([batch[1] for _, batch in enumerate(real_loader)], dim=0).to(device)
print('real_images', real_images.shape)

gan_files = sorted([file for ext in IMAGE_EXTENSIONS
                for file in gan_path.glob('*.{}'.format(ext))])
gan_dataset = ImagePathDataset(gan_files, transforms=transform)
gan_loader = torch.utils.data.DataLoader(gan_dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False
)

ssim_metric = []
for filename, gan_image in tqdm(gan_loader):
    filename = filename[0]
    
    gan_images = gan_image.repeat(len(real_images), 1, 1, 1).to(device)
    _ssim = ssim(gan_images, real_images, size_average=True, data_range=1)
    ssim_value = _ssim.cpu().numpy()
    ssim_metric.append(f'{filename}\t{ssim_value}')

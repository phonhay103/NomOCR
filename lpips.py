import torch
from PIL import Image
import torchvision.transforms as TF
import pathlib
from tqdm import tqdm
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

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

device = torch.device('cuda:0')
transform = TF.Compose([
    TF.Resize((32, 32)),
    TF.ToTensor(),
])
real_path = pathlib.Path('all_images')
gan_path = pathlib.Path('gan_images')
log_path = 'lpips.txt'

IMAGE_EXTENSIONS = {'bmp', 'jpg', 'jpeg', 'pgm', 'png', 'ppm', 'tif', 'tiff', 'webp'}

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

lpips = LearnedPerceptualImagePatchSimilarity(net_type='alex', reduction='mean', normalize=True).to(device)

lpips_values = []
lpips_metric = []
for filename, gan_image in tqdm(gan_loader):
    filename = filename[0]
    
    gan_images = gan_image.repeat(len(real_images), 1, 1, 1).to(device)
    _lpips = lpips(gan_images, real_images)
    lpips_value = _lpips.detach().cpu().numpy()
    lpips_values.append(lpips_value)
    lpips_metric.append(f'{filename}\t{lpips_value}')

with open(log_path, 'w') as f:
    f.write('\n'.join(lpips_metric))

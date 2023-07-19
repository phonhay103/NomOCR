import numpy as np
import pandas as pd

# Load lpips values
with open('ssim_16_16.txt') as f:
    lpips = [line.split('\t') for line in f.read().splitlines()]

df = pd.DataFrame(lpips, columns=['filename', 'ssim'])
df = df.astype({'ssim': np.float32})

# Load gan labels
with open('../NOM_CGGAN_errors_mean.txt') as f:
    filenames = [line.split('\t')[0] for line in f]

# Get lpips
df = df[df.filename.isin(filenames)]
print(df['ssim'].mean())
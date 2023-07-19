import numpy as np
from tqdm import tqdm
import pandas as pd

# Load lpips values
with open('lpips_32_32.txt') as f:
    lpips = [line.split('\t') for line in f.read().splitlines()]

df = pd.DataFrame(lpips, columns=['filename', 'lpips'])
df = df.astype({'lpips': np.float32})

# Load gan labels
with open('../NOM_CGGAN_train.txt') as f:
    filenames = [line.split('\t')[0] for line in f]

# Get lpips
df = df[df.filename.isin(filenames)]
print(df.lpips.mean())
import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class DeepfakeDataset(Dataset):
    def __init__(self, data_root='./data', csv_file='./FINAL_DATASET.csv', split='train', transform=None):
        """
        Args:
            data_root (string): Directory containing the 'images' folder.
            csv_file (string): Path to the CSV file with metadata.
            split (string): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.data_root = data_root
        self.split = split
        self.transform = transform
        
        # Load the CSV file
        if not os.path.exists(csv_file):
            raise FileNotFoundError(f"CSV file not found: {csv_file}")
            
        self.df = pd.read_csv(csv_file)
        
        # Filter by dataset_split
        self.df = self.df[self.df['dataset_split'] == split].reset_index(drop=True)
        
        # Keep only rows where the physical image file exists
        def file_exists(row):
            img_path = os.path.join(self.data_root, 'images', f"{row['image_id']}.jpg")
            return os.path.exists(img_path)
            
        self.df = self.df[self.df.apply(file_exists, axis=1)].reset_index(drop=True)

        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Build the image path
        img_id = str(self.df.loc[idx, 'image_id'])
        
        img_path = os.path.join(self.data_root, 'images', f"{img_id}.jpg")
        
        # Convert to RGB since standard ImageNet models expect 3 channels
        image = Image.open(img_path).convert('RGB')
            
        label = int(self.df.loc[idx, 'label_numeric'])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

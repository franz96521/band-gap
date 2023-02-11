from torch.utils.data import Dataset
import os 
from PIL import Image
import numpy as np
import torch
class ImageDataset(Dataset):
    def __init__(self, root,transform,max_samples=None):
        self.root = root       
        self.files  = os.listdir(root)
        self.arriba = os.listdir(root+"/arriba")
        self.abajo = os.listdir(root+"/abajo")
        self.derecha = os.listdir(root+"/derecha")
        self.izquierda = os.listdir(root+"/izquierda")
        self.enfrente = os.listdir(root+"/enfrente")
        self.atras = os.listdir(root+"/atras")
        self.bandgaps = np.loadtxt(root+'/bandgaps.csv')
        self.transform1 = transform
        self.max_samples = max_samples
    def __len__(self):
        if self.max_samples is None:
            return len(self.arriba)
        return min(self.max_samples, len(self.arriba))
    
    def __getitem__(self, idx):
        arriba =self.transform1( Image.open(self.root+"/arriba/"+self.arriba[idx]).convert('RGB'))
        abajo = self.transform1(Image.open(self.root+"/abajo/"+self.abajo[idx]).convert('RGB'))
        derecha = self.transform1(Image.open(self.root+"/derecha/"+self.derecha[idx]).convert('RGB'))
        izquierda = self.transform1(Image.open(self.root+"/izquierda/"+self.izquierda[idx]).convert('RGB'))
        enfrente =self.transform1( Image.open(self.root+"/enfrente/"+self.enfrente[idx]).convert('RGB'))
        atras = self.transform1(Image.open(self.root+"/atras/"+self.atras[idx]).convert('RGB'))
        # x = torch.cat((arriba,abajo,derecha,izquierda,enfrente,atras),0)
        bandgap = self.bandgaps[idx].astype(np.float32)
        return arriba,abajo,derecha,izquierda,enfrente,atras, bandgap
    
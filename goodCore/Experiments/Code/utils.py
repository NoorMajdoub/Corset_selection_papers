import numpy as np
import json
from torch.utils.data import DataLoader,Dataset
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader


def save_dataset_metadata(dataset, filepath='/kaggle/working/meta.json',data_flag="chestmnist"):
    #pip install medmnist
    #from medmnist import INFO
    """
    Save metadata about a dataset to a JSON file.
    """
    info = INFO[data_flag]

    meta = {
        "task": info['task'],
        "n_channels": info['n_channels'],
        "label": info['label']   
    }

    with open(filepath, 'w') as f:
        json.dump(meta, f)

def get_metadata(meta_path,info,label_array):
    """
    Retrieve metadata from a JSON file based on the specified info and label array.
    Parameters:
    - meta_path (str): The path to the JSON file containing the metadata.
    - info (str): The type of metadata to retrieve(n_channels, task, label)
    - label_array (list): A list of indices for the labels to retrieve(cause multilabels)
    Returns:- list: A list of metadata values corresponding to the specified info and label array.  
    """
    with open(meta_path, 'r') as file:
        meta = json.load(file)
    if info =="label":
        return [meta['label'][str(idx)] for idx, val in enumerate(label_array) if val == 1]
    else:
        return meta[info]
    



def get_dataset(X,Y):
        """
        Create a PyTorch dataset from the given input data and labels.
        X and Y are the npz files you read 
        """
        X_tensor = torch.tensor(X)
        Y_tensor = torch.tensor(Y)
        #the transforms were applied before savnig X and Y (when you initial created them)
        dataset = TensorDataset(X_tensor, Y_tensor)
        return dataset


def get_model_radDino():
    #most relaibel model for now is the raddino
    model = AutoModel.from_pretrained("microsoft/rad-dino")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    return model



def generate_embeddings(model,dataset,batch_size=32):


    loader = DataLoader(
    dataset,
    batch_size=batch_size,  #bigger migth overflow dino
    shuffle=False,
    num_workers=4,
    pin_memory=True)
    features_corset = []
    labels_corset = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for imgs, lbls in loader:
            #print("o")
        # inputs = processor(images=imgs, return_tensors="pt")
            #inputs = {k: v.to(device) for k, v in inputs.items()}

        # outputs = model(**inputs) #instead of usingproce cause you already did el procesing with transofrm
            outputs = model(pixel_values=imgs.to(device))
            features = outputs.last_hidden_state.mean(dim=1) 
            features_corset.append(features)
            labels_corset.append(lbls)
            if len(features_corset)%50==0:   #looks stupid improve /L
                print(len(features_corset)/len(loader))
    features_corset = torch.cat(features_corset, dim=0)  #I keep this ?
    labels_corset = torch.cat(labels_corset, dim=0)
    return features_corset, labels_corset
def save_embeddings(features_corset, labels_corset, file_path):
    np.savez(file_path, 
         features=features_corset.cpu().numpy(), 
         labels=labels_corset.cpu().numpy())
def load_embeddings(file_path):
    data = np.load(file_path)
    features_corset = data['features']
    labels_corset = data['labels']
    return features_corset, labels_corset

def main():
    #get the saved npz files of the corpus
    X = np.load('')
    Y = np.load('')
    # make em a dataset
    dataset = get_dataset(X, Y)
    #your raddino for embedding model
    model = get_model_radDino()
    embeddings = generate_embeddings(model, dataset)

    #apply the corset selection alsp
    

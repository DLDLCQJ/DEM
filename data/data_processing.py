import os
from os.path import join as opj
import numpy as np
import pandas as pd
from torchvision import transforms
import torch
import SimpleITK as sitk
#import matplotlib.pyplot as plt
from skimage.transform import resize
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler,TensorDataset
from sklearn.model_selection import train_test_split

def pad_todesire_2D(img, desired_shape):
    X_before = int((desired_shape[0]-img.shape[0])/2)
    Y_before = int((desired_shape[1]-img.shape[1])/2)
    X_after = desired_shape[0]-img.shape[0]-X_before
    Y_after = desired_shape[1]-img.shape[1]-Y_before
    npad = ((X_before, X_after),
            (Y_before, Y_after))
    padded = np.pad(img, pad_width=npad, mode='constant', constant_values=0)
    return padded

def crop_center_2D(img,crop_shape):
    x,y = img.shape
    cropx = crop_shape[0]
    cropy = crop_shape[1]
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[startx:startx+cropx,starty:starty+cropy]

def augment():
    return transforms.Compose([
    transforms.ToPILImage(),
    #transforms.Resize(image_resize=128,image_resize=128),
    transforms.RandomHorizontalFlip(p = 0.5),
    transforms.RandomRotation(45,),
    transforms.RandomVerticalFlip(p = 0.5,),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def norm_meta(X, norm_type='demean_std'):
    if norm_type == 'demean_std':
        X_o = np.float32(X.copy())
        m = np.mean(X_o)
        s = np.std(X_o)
        normalized_X = np.divide((X_o - m), s)
    elif norm_type == 'minmax':
        perc1 = np.percentile(X, 1)
        perc99 = np.percentile(X, 99)
        normalized_X = np.divide((X - perc1), (perc99 - perc1))
        normalized_X[normalized_X < 0] = 0.0
        normalized_X[normalized_X > 1] = 1.0
    return torch.tensor(normalized_X,dtype=torch.float32)

def meta_engineer(df):
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=[object]).columns
    for col in numerical_cols:
        df[col].fillna(df[col].median(), inplace=True)
    for col in categorical_cols:
        df[col].fillna('Unknown', inplace=True)
    #df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    encoder = OneHotEncoder(drop='first', sparse=False)
    encoded_cols = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_cols, columns=encoder.get_feature_names_out(categorical_cols))
    df.drop(columns=categorical_cols, inplace=True)
    df = pd.concat([df, encoded_df], axis=1)
    return df

class MyDataset(Dataset):
    def __init__(self, imgs,labels,norm_type=None,meta=None,augment=None):
        super(MyDataset, self).__init__()
        self.imgs = torch.tensor(imgs, dtype=torch.float32) if isinstance(imgs, np.ndarray) else imgs
        self.labels = torch.tensor(labels, dtype=torch.float32) if isinstance(labels, np.ndarray) else labels
        
        self.augment = augment
        self.meta = meta
        self.norm_type = norm_type
        self.set_transform()
    def __len__(self):
        return self.labels.shape[0]
    
    def set_transform(self):
        self.transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                           std=[0.229, 0.224, 0.225])

    def __getitem__(self, idx):
        imgs, labels = self.imgs[idx], self.labels[idx]
        if self.meta is not None:
            meta = norm_meta(imgs, self.norm_type)
            return (meta, labels)
        else:
            if self.augment is not None:
                imgs = self.augment(imgs)
            imgs = self.transform(imgs)
            return imgs,labels
    def update_labels(self, new_labels):
        assert len(new_labels) == len(self.labels), 'label length is same'
        return MyDataset(self.imgs, new_labels, self.norm_type, self.meta, self.augment)


class Filter_Dataset(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices
        self.labels = self.dataset.labels[indices]
        self.imgs = self.dataset.imgs[indices]
    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        return self.dataset[actual_idx] 
    def update_labels(self, new_labels):
        assert len(new_labels) == len(self.labels), 'label length is same'
        return MyDataset(self.imgs, new_labels)

def loading_data(path,img_file, label_file, desired_shape):
    nifti_list = pd.read_csv(opj(path,img_file + '.csv'))
    imgs = []
    for nii_file in nifti_list.to_numpy():
        filename = os.path.split(nii_file[0])[1]
        img = sitk.ReadImage(nii_file)
        img_arr = np.squeeze(sitk.GetArrayFromImage(img)) #reg:181*217*181 / mvps:91*109*91 / jacobian:181*217*181
        print(img_arr.shape)
        for i, slice_number in enumerate(range(80,95)):        ##30-45
            new_filename= "third_"+str(slice_number)+"_"+filename     #For FreeSurfer reconstructed image
            img_2D = img_arr[slice_number, :, :].astype(np.float32)
            min_shape = min(img_2D.shape)
            cropped_shape = list((min_shape,min_shape))
            cropped_img = crop_center_2D(img_2D, cropped_shape)
            for j in range (2):
                if desired_shape[j] < cropped_shape[j]:
                    final_img =  resize(cropped_img, desired_shape, order=3, mode='reflect', anti_aliasing=True)
                else:
                    final_img = pad_todesire_2D(cropped_img, desired_shape)
            processed_img = np.array(final_img).astype(float)
            #print(processed_img.shape)
            # if i % 10 == 0:
            #     mid_slice_x_after = processed_img
            #     plt.imshow(mid_slice_x_after, cmap='gray', origin='lower')
            #     plt.xlabel('First axis')
            #     plt.ylabel('Second axis')
            #     plt.colorbar(label='Signal intensity')
            #     plt.show()
            imgs.append(processed_img)
    ##
    imgs = np.squeeze(np.array(imgs))
    imgs = imgs[:, np.newaxis, :, :]
    processed_img = np.repeat(imgs,3,axis=1)
    ##
    y = pd.read_csv(opj(path, label_file + '.csv')).iloc[:,1].values
    y = np.repeat(y, 15)

    return processed_img, y


def load_test_data(args,data_type):
    if data_type==1:
        nifti_list = pd.read_csv(opj(args.path,args.test_img_file1 + '.csv'))
        y = pd.read_csv(opj(args.path, args.test_label_file1 + '.csv')).iloc[:,1].values
    elif data_type==2:
        nifti_list = pd.read_csv(opj(args.path,args.test_img_file2 + '.csv'))
        y = pd.read_csv(opj(args.path, args.test_label_file2 + '.csv')).iloc[:,1].values
    ##
    imgs = []
    for nii_file in nifti_list.to_numpy():
        filename = os.path.split(nii_file[0])[1]
        img = sitk.ReadImage(nii_file)
        img_arr = np.squeeze(sitk.GetArrayFromImage(img)) #reg:181*217*181 / mvps:91*109*91
        ##
        for i, slice_number in enumerate(range(80,95)):
            new_filename= "third_"+str(slice_number)+"_"+filename     #For FreeSurfer reconstructed image
            img_2D = img_arr[slice_number, :, :].astype(np.float32)
            # save_path = opj(filename +"_2D", new_filename)
            # nib.save(nib.Nifti1Image(fdata_2D, affine = np.eye(4)), save_path)
            min_shape = min(img_2D.shape)
            cropped_shape = list((min_shape,min_shape))
            cropped_img = crop_center_2D(img_2D, cropped_shape)
            for j in range (2):
                if args.desired_shape[j] < cropped_shape[j]:
                    #final_img = crop_center_2D(cropped_img, args.desired_shape)
                    final_img =  resize(cropped_img, args.desired_shape, order=3, mode='reflect', anti_aliasing=True)
                else:
                    final_img = pad_todesire_2D(cropped_img, args.desired_shape)
            #print(final_img.shape)
            processed_img = np.array(final_img).astype(float)
            imgs.append(processed_img)
    ##
    processed_img = np.squeeze(np.array(imgs))
    processed_img = processed_img[:, np.newaxis, :, :]
    processed_img = np.repeat(processed_img,3,axis=1)
    ##
    y = np.repeat(y, 15)
    return processed_img, y


#-*- coding: UTF-8 -*-
"""
    Name: Qingneng Li (Eng: Lional)
    Time: 2019/12/23
    Place: SIAT, Shenzhen
    Item: Understanding of Moving average

"""
import numpy as np, os, cv2, h5py, pydicom
from skimage.transform import radon,iradon

class Data_Loader():
    def __init__(self, WL=300, WW=1500):
        self.intercept = -1024
        self.WL = WL
        self.WW = WW
        self.crop_size = (256, 256)
        self.prefix_path = 'E:/PET-DICOMS/AC-CT'

    def adjust_windows(self, img):
        # Chest window: [-160, 240] HU
        # Bone window: [-450, 1050] HU
        im = (img + self.intercept)
        UL = self.WL + self.WW / 2.0
        DL = self.WL - self.WW / 2.0
        UL_mask = im > UL
        DL_mask = im <= DL
        img_new = (im - DL) / self.WW
        img_new[UL_mask] = 1
        img_new[DL_mask] = 0
        return img_new

    def central_crop(self, img):
        if img.ndim == 2:
            H, W = img.shape[0], img.shape[1]
            LP = (W - self.crop_size[1])//2
            TP = (H - self.crop_size[0])//2
            im = img[TP:TP+self.crop_size[0],
                    LP:LP+self.crop_size[1]]
            return im

        if img.ndim == 3: #[H, W, C]
            H, W = img.shape[0], img.shape[1]
            LP = (W - self.crop_size[1])//2
            TP = (H - self.crop_size[0])//2
            im = img[TP:TP+self.crop_size[0],
                    LP:LP+self.crop_size[1], :]
            return im

        if img.ndim == 4: # [B,H,W,C]
            H, W = img.shape[1], img.shape[2]
            LP = (W - self.crop_size[1])//2
            TP = (H - self.crop_size[0])//2
            im = img[:, TP:TP+self.crop_size[0],
                    LP:LP+self.crop_size[1], :]
            return im

    def _read_PET(self, path):
        image = pydicom.read_file(path).pixel_array
        image = np.float32(image)
        image = cv2.resize(image, (434, 434))
        image = self.central_crop(image)
        # image = 255.0 * (1 - image / 32767.0)
        image = 255.0 * (1 - image/np.max(image))
        return image

    def _read_CT(self, path):
        image = pydicom.read_file(path).pixel_array
        image = np.float32(image)
        image = cv2.resize(image, (311, 311))
        image = self.central_crop(image)
        image = 255.0 * self.adjust_windows(image)
        return image

    def _read_UMAP(self, NAC, AC, reverse=True):
        if reverse:
            nac = 255.0 - NAC
            ac = 255.0 - AC
        else:
            nac = NAC
            ac = AC
        theta = np.linspace(0, 180, 256, False)
        ACF = radon(ac, theta) / (radon(nac, theta) + 1e-12)
        umap = iradon(ACF, theta)
        # umap = umap - np.mean(umap)
        # umap_mask = umap > 0
        # umap = umap * umap_mask
        umap = cv2.normalize(umap, None, 0, 255, cv2.NORM_MINMAX)
        return umap

    def save_jpg(self):
        file_names = os.listdir(self.prefix_path + '/NAC')

        for f in file_names:
            print(f)

            Path_NAC = self.prefix_path + '/NAC/' + f
            Path_AC = self.prefix_path + '/AC/' + f
            Path_CT = self.prefix_path + '/CT/' + f
            NAC = self._read_PET(Path_NAC)
            AC = self._read_PET(Path_AC)
            CT = self._read_CT(Path_CT)
            umap = self._read_UMAP(NAC, AC, True)

            # cv2.imwrite('./data/NAC-AC/'+f.replace('.dcm','.jpg'),
            #             np.concatenate((NAC, AC),1))
            # cv2.imwrite('./data/UMAP-CT/'+f.replace('.dcm','.jpg'),
            #             np.concatenate((umap, CT),1))
            cv2.imwrite('./data/UMAP/'+f.replace('.dcm','.jpg'), umap)
            cv2.imwrite('./data/NAC/'+f.replace('.dcm','.jpg'), NAC)
            cv2.imwrite('./data/AC/'+f.replace('.dcm','.jpg'), AC)
            cv2.imwrite('./data/CT/'+f.replace('.dcm','.jpg'), CT)

    def save_h5(self):
        file_names = os.listdir(self.prefix_path + '/NAC')
        NAC, AC, CT, umap = [], [], [], []
        for f in file_names:
            print(f)
            Path_NAC = prefix_path + '/NAC/' + f
            Path_AC = prefix_path + '/AC/' + f
            Path_CT = prefix_path + '/CT/' + f
            NAC.append( _read_PET(Path_NAC))
            AC.append(_read_PET(Path_AC))
            CT.append(_read_CT(Path_CT))
            umap.append(_read_UMAP(NAC, AC, True))
        f = h5py.File("./data/data.hdf5", "w")
        f.create_dataset('NAC', data=np.array(NAC))
        f.create_dataset('AC', data=np.array(AC))
        f.create_dataset('umap', data=np.array(umap))
        f.create_dataset('CT', data=np.array(CT))
        f.close()
        print('Finish H5 file saving')


if __name__ == '__main__':
    dl = Data_Loader()
    dl.save_jpg()






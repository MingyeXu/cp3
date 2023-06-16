import torch
import numpy as np
import torch.utils.data as data
import h5py
import os


class MVP_CP(data.Dataset):
    def __init__(self, prefix="train"):
        if prefix=="train":
            self.file_path = '../../../data/MVP_Train_CP_v2.h5'
            # self.file_path = '../../../data/ablation_data/MVP_Train_InCom_CROP_V5_IOI-I_NUM-1_CropRate-0.9.h5'
            # self.file_path3 = '../../data/MVP_TrainVALTEST_InCom_CROP_V3.h5'
        elif prefix=="val":
            self.file_path = '../../../data/MVP_Test_CP_v2.h5'
        elif prefix=="test":
            self.file_path = '../../../data/MVP_ExtraTest_Shuffled_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])

        # print(self.input_data.shape)

        if prefix is "val":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            self.labels = np.array(input_file['labels'][()])
            print(self.gt_data.shape, self.labels.shape)
        if prefix is "train":
            self.gt_data = np.array(input_file['complete_pcds'][()])
            self.labels = np.array(input_file['labels'][()])
            # print('SS',self.input_data.shape)
            # print('SS',self.gt_data.shape)
            # print('SS',self.labels.shape)


            # input_file2 = h5py.File(self.file_path2, 'r')
            # input_data2 = np.array(input_file2['incomplete_pcds'][()])
            # gt_data2 = np.array(input_file2['complete_pcds'][()])
            # labels2=np.array(np.zeros(gt_data2.shape[0]))

            # input_file3 = h5py.File(self.file_path3, 'r')
            # input_data3 = np.array(input_file3['incomplete_pcds'][()])
            # gt_data3 = np.array(input_file3['complete_pcds'][()])
            # labels3=np.array(np.zeros(gt_data3.shape[0]))   

            # self.input_data = np.concatenate((self.input_data,input_data2))
            # self.gt_data = np.concatenate((self.gt_data,gt_data2))
            # self.labels = np.concatenate((self.labels,labels2))
            
            print(self.input_data.shape)
            print(self.gt_data.shape)
            print(self.labels.shape)
            # exit(0)

        input_file.close()
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))

        if self.prefix is  "val":
            complete = torch.from_numpy((self.gt_data[index]))
            label = (self.labels[index])
            lbs = np.eye(16)[label] 
            return lbs, partial, complete
        if self.prefix is  "train":
            complete = torch.from_numpy((self.gt_data[index]))
            label = (self.labels[index])
            lbs = np.eye(16)[label] 
            return lbs, partial, complete

        else:
            return partial




class ShapeNetH5(data.Dataset):
    def __init__(self, train=True, npoints=2048, novel_input=True, novel_input_only=False):
        if train:
            self.input_path = '../../../data/mvp_org/mvp_train_input.h5'
            self.gt_path = '../../../data/mvp_org/mvp_train_gt_%dpts.h5' % npoints
        else:
            self.input_path = '../../../data/mvp_org/mvp_test_input.h5'
            self.gt_path = '../../../data/mvp_org/mvp_test_gt_%dpts.h5' % npoints
        self.npoints = npoints
        self.train = train

        input_file = h5py.File(self.input_path, 'r')
        self.input_data = np.array((input_file['incomplete_pcds'][()]))
        self.labels = np.array((input_file['labels'][()]))
        self.novel_input_data = np.array((input_file['novel_incomplete_pcds'][()]))
        self.novel_labels = np.array((input_file['novel_labels'][()]))
        input_file.close()

        gt_file = h5py.File(self.gt_path, 'r')
        self.gt_data = np.array((gt_file['complete_pcds'][()]))
        self.novel_gt_data = np.array((gt_file['novel_complete_pcds'][()]))
        gt_file.close()

        if novel_input_only:
            self.input_data = self.novel_input_data
            self.gt_data = self.novel_gt_data
            self.labels = self.novel_labels
        elif novel_input:
            self.input_data = np.concatenate((self.input_data, self.novel_input_data), axis=0)
            self.gt_data = np.concatenate((self.gt_data, self.novel_gt_data), axis=0)
            self.labels = np.concatenate((self.labels, self.novel_labels), axis=0)

        print(self.input_data.shape)
        print(self.gt_data.shape)
        print(self.labels.shape)
        self.len = self.input_data.shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        partial = torch.from_numpy((self.input_data[index]))
        complete = torch.from_numpy((self.gt_data[index // 26]))
        label = (self.labels[index])
        label = label.astype(int)
        # print(label)
        lbs = np.eye(16)[label] 
        return label,lbs, partial, complete

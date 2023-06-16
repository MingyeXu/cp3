import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import sys
from glob import glob
import scipy.io as sio
sys.path.append("../utils")
from mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation



def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)


def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)


class MVP_Reg_ShapeNet_All(data.Dataset):
    def __init__(self, prefix="train", rot_aug=False, data_root='/opt/data/private/pointcloud_completion/data/shapenet_all'):
        self.prefix = prefix
        self.rot_aug = rot_aug  # the loaded partial PCs are aligned, need rot for registration task
        self.data_root = data_root
        self.max_angle = 180
        self.max_trans = 0.5

        category_all = ["airplane", "cabinet", "car", "chair", "lamp",
                        "sofa", "table", "watercraft", "bed", "bench",
                        "bookshelf", "bus", "guitar", "motorbike", "pistol",
                        "skateboard"]
        view_all = ["view_{}".format(i) for i in range(26)]
        self.instance_all = glob(os.path.join(self.data_root, '*', 'view_0', '*.npy'))
        self.len = len(self.instance_all)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        instance_index_view_0 = self.instance_all[index]
        # get two random view
        view_i = np.random.randint(26)
        view_j = np.random.randint(26)
        instance_index_view_i = instance_index_view_0.replace('view_0', 'view_{}'.format(view_i)).replace('view0', 'view{}'.format(view_i))
        instance_index_view_j = instance_index_view_0.replace('view_0', 'view_{}'.format(view_j)).replace('view0', 'view{}'.format(view_j))

        partial_pc_i = np.load(instance_index_view_i)
        partial_pc_j = np.load(instance_index_view_j)

        transform = np.eye(4)
        if self.rot_aug:
            transform = random_pose(self.max_angle, self.max_trans / 2)
            pose1 = random_pose(np.pi, self.max_trans)
            pose2 = transform @ pose1
            partial_pc_i = partial_pc_i @ pose1[:3, :3].T + pose1[:3, 3]
            partial_pc_j = partial_pc_j @ pose2[:3, :3].T + pose2[:3, 3]
        return partial_pc_i, partial_pc_j, transform


class MVP_Com_ShapeNet_All(data.Dataset):
    def __init__(self, prefix="train", rot_aug=False, data_root='/opt/data/private/pointcloud_completion/data/shapenet_all/shapenet_ours'):
        self.prefix = prefix
        self.rot_aug = rot_aug  # the loaded partial PCs are aligned, need rot for registration task
        self.data_root = data_root
        self.max_angle = 180
        self.max_trans = 0.5

        category_all = ["airplane", "cabinet", "car", "chair", "lamp",
                        "sofa", "table", "watercraft", "bed", "bench",
                        "bookshelf", "bus", "guitar", "motorbike", "pistol",
                        "skateboard"]
        view_all = ["view_{}".format(i) for i in range(26)]
        self.instance_all = glob(os.path.join(self.data_root, '*', 'view_0', '*.npy'))
        self.len = len(self.instance_all)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        instance_index_view_0 = self.instance_all[index]
        num_view = 26
        # category car only have 13 views available
        # print(instance_index_view_0)
        if instance_index_view_0.split('/')[-3] == 'car':
            num_view = 13

        save_path = instance_index_view_0.replace('view_0/', '')
        save_path = save_path.replace('view0_', '') 
        save_path = save_path.replace('shapenet_ours', 'shapenet_completed') 
               
        if not os.path.exists(save_path):
        # get one random view and complete view
            instance_index_view_i_all = []
            num_v = num_view
            for view_i in range(num_view):
                file_instance_view_i = instance_index_view_0.replace('view_0', 'view_{}'.format(view_i)).replace('view0', 'view{}'.format(view_i))
                if not os.path.exists(file_instance_view_i):  # skip instances with incomplete views
                    # self.__getitem__(np.random.randint(self.len))
                    num_v = num_v-1
                    
                else:
                    instance_index_view_i_all.append(np.load(file_instance_view_i))
            instance_index_view_i_all = np.array(instance_index_view_i_all)
            # print(num_v)
            partial_pc_i = instance_index_view_i_all[np.random.randint(num_v), :, :]
            complete_pc = instance_index_view_i_all.reshape(-1, 3)
            complete_pc = torch.Tensor(complete_pc).cuda()
            complete_pc = complete_pc.unsqueeze(0).transpose(1, 2).contiguous()
            # print(complete_pc.size())
            idx_fps = furthest_point_sample(complete_pc.transpose(1, 2).contiguous(), 2048)
            complete_pc = gather_points(complete_pc, idx_fps).transpose(1, 2).contiguous().squeeze() # 3, 2048
            
            # print(complete_pc.size())
            complete_pc = complete_pc.cpu().numpy()
            
            # complete_pc = complete_pc[np.random.choice(complete_pc.shape[0], 2048, replace=False)]
            np.save(save_path, complete_pc)
            # print(save_path)
            # print("save .npy done")
            # save_fname_mat = ("test.mat")
            # sio.savemat(save_fname_mat, {
            #             'partial_pc_i': partial_pc_i,
            #             'complete_pc': complete_pc
            #             })
            # print('saved')
            # exit(0)
        else:
            partial_pc_i = np.load(save_path)
            partial_pc_i = partial_pc_i
            # partial_pc_i = np.zeros([2048,3]).astype(np.double)
            complete_pc = partial_pc_i
            # print(complete_pc.dtype)

        return 1


if __name__ == '__main__':
    # ds = MVP_Reg_ShapeNet_All(rot_aug=False)
    # dataloader = torch.utils.data.DataLoader(ds, batch_size=128, num_workers=8)
    # for pc_src, pc_tgt, trans in dataloader:
    #     print("")
    # print(ds.len)

    ds = MVP_Com_ShapeNet_All() #36467
    dataloader = torch.utils.data.DataLoader(ds, batch_size=20, num_workers=0)
    print(ds.len)
    i=0
    for at in dataloader:
        print("processing:",i)
        i=i+1
        # print("pc_src",pc_src.size())
        # print("pc_tgt",pc_tgt.size())
    



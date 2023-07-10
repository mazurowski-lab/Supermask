import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from unet import UNet3D
from .unet_own import Unet

class registerNet(nn.Module):

    def __init__(self, input_channels, encoder_nc,input_size=[128, 128, 128]):
        super(registerNet, self).__init__()
        self.input_size = input_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_features = torch.prod(((((torch.tensor(input_size) - 4) / 2 - 4) / 2) - 4) / 2).long()

        if input_size[0]==128:
            self.conv1 = nn.Sequential(nn.AvgPool3d(2, stride=2),nn.Conv3d(input_channels, 16, kernel_size=5),nn.AvgPool3d(2, stride=2),nn.ReLU(True))
        else:
            self.conv1 = nn.Sequential(nn.Conv3d(input_channels, 16, kernel_size=5),nn.AvgPool3d(2, stride=2),nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv3d(16, 32, kernel_size=5),nn.AvgPool3d(2, stride=2),nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=5),nn.AvgPool3d(2, stride=2),nn.ReLU(True))
        self.fc = nn.Sequential(nn.Linear(64 *4*4*4, 32),nn.ReLU(True))
        self.theta = torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float).view(3, 4)
        #self.grid= torch.tensor((1, 128, 128, 128,3), dtype=torch.float)

        # Regressor for the 3 * 4 affine matrix
        # self.affine_regressor = nn.Linear(32, 3 * 4)

        # initialize the weights/bias with identity transformation
        # self.affine_regressor.weight.data.zero_()
        # self.affine_regressor.bias.data.copy_(torch.tensor([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0], dtype=torch.float))

        # Regressor for individual parameters
        self.translation = nn.Linear(32, 3)
        self.rotation = nn.Linear(32, 3)
        #self.scaling = nn.Linear(32, 3)
        #self.shearing = nn.Linear(32, 3)

        # initialize the weights/bias with identity transformation
        self.translation.weight.data.zero_()
        self.translation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        self.rotation.weight.data.zero_()
        self.rotation.bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        #self.scaling.weight.data.zero_()
        #self.scaling.bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))
        #self.shearing.weight.data.zero_()
        #self.shearing.bias.data.copy_(torch.tensor([0, 0, 0], dtype=torch.float))

    def get_theta(self, i):
        return self.theta[i]

    def forward(self, x):
        xs = self.conv1(x)
        xs = self.conv2(xs)
        xs = self.conv3(xs)
        #print('xs shape:',xs.shape)
        xs = xs.view(xs.size(0), -1)
        xs = self.fc(xs)
        # theta = self.affine_regressor(xs).view(-1, 3, 4)
        self.theta = self.affine_matrix(xs)

        # extract first channel for warping
        mov = x.narrow(dim=1, start=0, length=1)
        mov,grid = self.warp_image(mov)
        #print('The sum of the grid: ',grid.sum())
        # warp image
        return mov,grid

    def gen_3d_mesh_grid(self, d, h, w):
        # move into self to save compute?
        d_s = torch.linspace(-1, 1, d)
        h_s = torch.linspace(-1, 1, h)
        w_s = torch.linspace(-1, 1, w)

        d_s, h_s, w_s = torch.meshgrid([d_s, h_s, w_s])
        one_s = torch.ones_like(w_s)

        mesh_grid = torch.stack([w_s, h_s, d_s, one_s])
        return mesh_grid  # 4 x d x h x w

    def affine_grid(self, theta, size):
        b, c, d, h, w = size
        mesh_grid = self.gen_3d_mesh_grid(d, h, w)
        mesh_grid = mesh_grid.unsqueeze(0)

        mesh_grid = mesh_grid.repeat(b, 1, 1, 1, 1)  # channel dim = 4
        mesh_grid = mesh_grid.view(b, 4, -1)
        mesh_grid = torch.bmm(theta, mesh_grid)  # channel dim = 3
        mesh_grid = mesh_grid.permute(0, 2, 1)  # move channel to last dim
        return mesh_grid.view(b, d, h, w, 3)

    def warp_image(self, img):
        grid = F.affine_grid(self.theta, img.size()).to(img.get_device())
        wrp = F.grid_sample(img, grid)
        return wrp,grid
        
    def repeat_warp(self, img, grid):
        wrp = F.grid_sample(img, grid)
        return wrp

    def affine_matrix(self, x):
        b = x.size(0)

        # self.trans = self.translation(x)
        trans = torch.tanh(self.translation(x)) * 0.5
        translation_matrix = torch.zeros([b, 4, 4], dtype=torch.float)
        translation_matrix[:, 0, 0] = 1.0
        translation_matrix[:, 1, 1] = 1.0
        translation_matrix[:, 2, 2] = 1.0
        translation_matrix[:, 0, 3] = trans[:, 0].view(-1)
        translation_matrix[:, 1, 3] = trans[:, 1].view(-1)
        translation_matrix[:, 2, 3] = trans[:, 2].view(-1)
        translation_matrix[:, 3, 3] = 1.0

        # self.rot = self.rotation(x)
        rot = torch.tanh(self.rotation(x)) * (math.pi / 4.0)
        # rot Z
       
        angle_1 = rot[:, 0].view(-1)
        #print('rotation at z is:', angle_1)
        rotation_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_1[:, 0, 0] = torch.cos(angle_1)
        rotation_matrix_1[:, 0, 1] = -torch.sin(angle_1)
        rotation_matrix_1[:, 1, 0] = torch.sin(angle_1)
        rotation_matrix_1[:, 1, 1] = torch.cos(angle_1)
        rotation_matrix_1[:, 2, 2] = 1.0
        rotation_matrix_1[:, 3, 3] = 1.0
        # rot X
        angle_2 = rot[:, 1].view(-1)
        #print('rotation at x is:', angle_2)
        rotation_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_2[:, 1, 1] = torch.cos(angle_2)
        rotation_matrix_2[:, 1, 2] = -torch.sin(angle_2)
        rotation_matrix_2[:, 2, 1] = torch.sin(angle_2)
        rotation_matrix_2[:, 2, 2] = torch.cos(angle_2)
        rotation_matrix_2[:, 0, 0] = 1.0
        rotation_matrix_2[:, 3, 3] = 1.0
        # rot Y
        angle_3 = rot[:, 2].view(-1)
        #print('rotation at y is:', angle_3)
        rotation_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float)
        rotation_matrix_3[:, 0, 0] = torch.cos(angle_3)
        rotation_matrix_3[:, 2, 0] = -torch.sin(angle_3)
        rotation_matrix_3[:, 0, 2] = torch.sin(angle_3)
        rotation_matrix_3[:, 2, 2] = torch.cos(angle_3)
        rotation_matrix_3[:, 1, 1] = 1.0
        rotation_matrix_3[:, 3, 3] = 1.0

        # rot Z
        # angle_3 = rot[:, 2].view(-1)
        # print('rotation at y is:', angle_3)
        # rotation_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float)
        # rotation_matrix_3[:, 0, 0] = torch.cos(angle_3)
        # rotation_matrix_3[:, 0, 1] = -torch.sin(angle_3)
        # rotation_matrix_3[:, 1, 0] = torch.sin(angle_3)
        # rotation_matrix_3[:, 1, 1] = torch.cos(angle_3)
        # rotation_matrix_3[:, 2, 2] = 1.0
        # rotation_matrix_3[:, 3, 3] = 1.0

        rotation_matrix = torch.bmm(rotation_matrix_1, rotation_matrix_2)
        rotation_matrix = torch.bmm(rotation_matrix, rotation_matrix_3)

        # self.scale = F.softplus(self.scaling(x), beta=np.log(2.0))
        # self.scale = self.scaling(x)
        # scale = torch.tanh(self.scaling(x)) * 0.2
        # scaling_matrix = torch.zeros([b, 4, 4], dtype=torch.float)
        # # scaling_matrix[:, 0, 0] = self.scale[:, 0].view(-1)
        # # scaling_matrix[:, 1, 1] = self.scale[:, 1].view(-1)
        # # scaling_matrix[:, 2, 2] = self.scale[:, 2].view(-1)
        # scaling_matrix[:, 0, 0] = torch.exp(scale[:, 0].view(-1))
        # scaling_matrix[:, 1, 1] = torch.exp(scale[:, 1].view(-1))
        # scaling_matrix[:, 2, 2] = torch.exp(scale[:, 2].view(-1))
        # scaling_matrix[:, 3, 3] = 1.0

        # self.shear = self.shearing(x)
        # shear = torch.tanh(self.shearing(x)) * (math.pi / 4.0)

        # shear_1 = shear[:, 0].view(-1)
        # shearing_matrix_1 = torch.zeros([b, 4, 4], dtype=torch.float)
        # shearing_matrix_1[:, 1, 1] = torch.cos(shear_1)
        # shearing_matrix_1[:, 1, 2] = -torch.sin(shear_1)
        # shearing_matrix_1[:, 2, 1] = torch.sin(shear_1)
        # shearing_matrix_1[:, 2, 2] = torch.cos(shear_1)
        # shearing_matrix_1[:, 0, 0] = 1.0
        # shearing_matrix_1[:, 3, 3] = 1.0

        # shear_2 = shear[:, 1].view(-1)
        # shearing_matrix_2 = torch.zeros([b, 4, 4], dtype=torch.float)
        # shearing_matrix_2[:, 0, 0] = torch.cos(shear_2)
        # shearing_matrix_2[:, 0, 2] = torch.sin(shear_2)
        # shearing_matrix_2[:, 2, 0] = -torch.sin(shear_2)
        # shearing_matrix_2[:, 2, 2] = torch.cos(shear_2)
        # shearing_matrix_2[:, 1, 1] = 1.0
        # shearing_matrix_2[:, 3, 3] = 1.0

        # shear_3 = shear[:, 2].view(-1)
        # shearing_matrix_3 = torch.zeros([b, 4, 4], dtype=torch.float)
        # shearing_matrix_3[:, 0, 0] = torch.cos(shear_3)
        # shearing_matrix_3[:, 0, 1] = -torch.sin(shear_3)
        # shearing_matrix_3[:, 1, 0] = torch.sin(shear_3)
        # shearing_matrix_3[:, 1, 1] = torch.cos(shear_3)
        # shearing_matrix_3[:, 2, 2] = 1.0
        # shearing_matrix_3[:, 3, 3] = 1.0

        #shearing_matrix = torch.bmm(shearing_matrix_1, shearing_matrix_2)
        #shearing_matrix = torch.bmm(shearing_matrix, shearing_matrix_3)

        # Affine transform
        #matrix = torch.bmm(shearing_matrix, scaling_matrix)
        #matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        #matrix = torch.bmm(matrix, rotation_matrix)
        #matrix = torch.bmm(matrix, translation_matrix)

        # matrix = torch.bmm(translation_matrix, rotation_matrix)
        # matrix = torch.bmm(matrix, torch.transpose(shearing_matrix, 1, 2))
        # matrix = torch.bmm(matrix, scaling_matrix)
        # matrix = torch.bmm(matrix, shearing_matrix)

        # No-shear transform
        #matrix = torch.bmm(scaling_matrix, rotation_matrix)
        #matrix = torch.bmm(matrix, translation_matrix)

        # Rigid-body transform
        matrix = torch.bmm(translation_matrix,rotation_matrix)

        # print('shearing')
        # print(shearing_matrix[0, :, :])
        # print('scaling')
        # print(scaling_matrix[0, :, :])
        #print('rotation')
        #print(rotation_matrix[0, :, :])
        #print('translation')
        #print(translation_matrix[0, :, :])
        #print('affine')
        #print(matrix[0, :, :])

        return matrix[:, 0:3, :]

class registerNeter(nn.Module):
    def __init__(self, input_nc, encoder_nc, input_size, gpu_ids=[]):
        super(registerNeter, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = registerNet(input_nc, encoder_nc,input_size)
        if self.gpu_ids:
            self.model = nn.DataParallel(self.model,self.gpu_ids)
        

    def forward(self, input):
        return self.model(input)
        
    def get_theta(self,i):
        return self.model.module.get_theta(i)

    def repeat_warp(self,img,grid):
        return self.model.module.repeat_warp(img,grid)



class segNeter(nn.Module):
    def __init__(self, input_nc, encoder_nc, decoder_nc, gpu_ids=[]):
        super(segNeter, self).__init__()
        self.gpu_ids = gpu_ids
        self.model = UNet3D(
        in_channels=1, 
        out_classes=1,
        num_encoding_blocks=3,
        padding=True,
        normalization='batch'
    )
        #self.model = Unet(input_nc,encoder_nc,decoder_nc)
        if self.gpu_ids:
            self.model = nn.DataParallel(self.model,self.gpu_ids)

    def forward(self, input):
        return self.model(input)


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + 1 + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.9)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

def weights_init_normal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from .base_model import BaseModel
from .stn_networks import *
import torch.nn as nn
import functools
from torch.nn import init
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import os
from .loss import crossCorrelation3D, gradientLoss, DiceSensitivityLoss, MaxoverlapLoss, NormalizedCrossCorrelationLoss, DiceLoss


def define_G(input_nc, encoder_nc, input_size, which_model_netG, init_type='kaiming', gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_netG == 'registnet':
        netG = registerNeter(input_nc, encoder_nc, input_size, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(gpu_ids[0])
    init_weights(netG, init_type=init_type)
    return netG

def define_seg(input_nc, encoder_nc, decoder_nc, which_model_seg, init_type='kaiming',gpu_ids=[]):
    unet= None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert (torch.cuda.is_available())
    if which_model_seg == 'unet':
        segnet = segNeter(input_nc, encoder_nc, decoder_nc, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_seg)
    if len(gpu_ids) > 0:
        segnet.cuda(gpu_ids[0])
    return segnet    

class cycleregister(BaseModel):
    def name(self):
        return 'cycleregister'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.model_type = opt.model_type
        print('model type:',self.model_type)
        nb = opt.batchSize
        size = opt.fineSize[0]
        self.input_A = self.Tensor(nb, 1, size, size)
        self.input_B = self.Tensor(nb, 1, size, size)
        if self.model_type == 'mask' or self.model_type == 'seg_align' or self.model_type=='only_seg':
            self.mask_A = self.Tensor(nb, 1, size, size)
            self.mask_B = self.Tensor(nb, 1, size, size)

        if self.model_type == 'seg_align':
            self.segnet = define_seg(1, opt.encoder_nc, opt.decoder_nc, opt.which_model_netS, opt.init_type, self.gpu_ids)
            self.netG = define_G(opt.input_nc, opt.encoder_nc[0], opt.inputSize, opt.which_model_netG, opt.init_type, self.gpu_ids)
            self.path_G = os.path.join(opt.checkpoints_dir_G,opt.name_G)
            self.path_S = os.path.join(opt.checkpoints_dir_S,opt.name_S)
            # load/define networks
            if opt.continue_train:
                self.load_network(self.netG, 'G', opt.which_epoch_G,self.path_G)
                self.load_network(self.segnet, 'S', opt.which_epoch_S,self.path_S)
            if not self.isTrain:
                self.load_network(self.netG, 'G', opt.which_epoch_G,self.path_G)
                self.load_network(self.segnet, 'S', opt.which_epoch_S,self.path_S)
            if self.isTrain:
                self.old_lr = opt.lr

                # define loss functions
                self.criterionL2 = gradientLoss('l2')
                #self.criterionCC = crossCorrelation3D(1, kernel=(9,9,9))
                self.criterionCC = crossCorrelation3D(1, kernel=(15,15,15))  # image correlation 
                if self.model_type=='mask':
                    self.criterionMk = nn.MSELoss()
                elif self.model_type =='seg_align':
                    #self.criterionMk = nn.MSELoss()
                    #self.criterionMk = crossCorrelation3D(1, kernel=(9,9,9))
                    self.criterionMk = crossCorrelation3D(1, kernel=(15,15,15))
                    self.seg = DiceLoss()

                self.criterionCy = torch.nn.L1Loss()
                self.criterionId = crossCorrelation3D(1, kernel=(9,9,9))
                #self.criterionId = crossCorrelation3D(1, kernel=(21,21,21))

                # initialize optimizers
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_S = torch.optim.Adam(self.segnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers = []
                self.schedulers = []
                self.optimizers.append(self.optimizer_G)
                self.optimizers.append(self.optimizer_S)
                for optimizer in self.optimizers:
                    self.schedulers.append(get_scheduler(optimizer, opt))
        elif self.model_type=='only_seg':
            path = os.path.join(opt.checkpoints_dir,opt.name)
            self.path_S = path
            self.segnet = define_seg(1, opt.encoder_nc, opt.decoder_nc, opt.which_model_netS, opt.init_type, self.gpu_ids)
            self.seg = DiceLoss()
             # load/define networks
            if opt.continue_train:
                which_epoch = opt.which_epoch
                self.load_network(self.segnet, 'S', which_epoch,path)
                #self.load_network(self.netG, 'G_B', which_epoch)
            if not self.isTrain:
                which_epoch = opt.which_epoch
                self.load_network(self.segnet, 'S', which_epoch,path)
            if self.isTrain:
                self.old_lr = opt.lr

                # initialize optimizers
                self.optimizer_S = torch.optim.Adam(self.segnet.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers = []
                self.schedulers = []
                self.optimizers.append(self.optimizer_S)
                for optimizer in self.optimizers:
                    self.schedulers.append(get_scheduler(optimizer, opt))
        else: # only align
            self.path_G = os.path.join(opt.checkpoints_dir_G,opt.name_G)
            self.netG = define_G(opt.input_nc, opt.encoder_nc[0], opt.inputSize, opt.which_model_netG, opt.init_type, self.gpu_ids)
        
            # load/define networks
            if opt.continue_train:
                which_epoch = opt.which_epoch
                self.load_network(self.netG, 'G', which_epoch)
                #self.load_network(self.netG, 'G_B', which_epoch)
            if not self.isTrain:
                which_epoch = opt.which_epoch
                self.load_network(self.netG, 'G', which_epoch)
            if self.isTrain:
                self.old_lr = opt.lr

                # define loss functions
                self.criterionL2 = gradientLoss('l2')
                #self.criterionCC = crossCorrelation3D(1, kernel=(9,9,9))
                self.criterionCC = crossCorrelation3D(1, kernel=(15,15,15))
                if self.model_type=='mask':
                    self.criterionMk = crossCorrelation3D(1, kernel=(15,15,15))

                self.criterionCy = torch.nn.L1Loss()
                #self.criterionId = crossCorrelation3D(1, kernel=(9,9,9))
                self.criterionId = crossCorrelation3D(1, kernel=(15,15,15))

                # initialize optimizers
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers = []
                self.schedulers = []
                self.optimizers.append(self.optimizer_G)
                for optimizer in self.optimizers:
                    self.schedulers.append(get_scheduler(optimizer, opt))

            print('---------- Networks initialized -------------')
            print_network(self.netG)
            print('-----------------------------------------------')

    def set_input(self, input):
        input_A = input['A']
        input_B = input['B']
        mask_A = input['M_A']
        mask_B = input['M_B']
        #plt.imsave('label1.png', mask_A[0,0,30,:,:], cmap='gray')
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        if self.model_type=='mask' or self.model_type=='seg_align' or self.model_type=='only_seg':
            self.mask_A.resize_(mask_A.size()).copy_(mask_A)
            self.mask_B.resize_(mask_B.size()).copy_(mask_B)
        else:
            self.mask_A.resize_(input_A.size()).copy_(input_A)
            self.mask_B.resize_(input_B.size()).copy_(input_A)
        self.image_paths = input['path']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        if self.model_type=='mask' or self.model_type=='seg_align' or self.model_type=='only_seg':
            self.real_mask_A = Variable(self.mask_A)
            self.real_mask_B = Variable(self.mask_B)
        #plt.imsave('label1.png', self.real_mask_A.cpu()[0,0,30,:,:], cmap='gray')

    def test(self):
        if self.model_type=='seg_align':
            with torch.no_grad():
                real_A = Variable(self.input_A)
                real_B = Variable(self.input_B)
                fake_B, flow_A = self.netG(torch.cat((real_A, real_B), dim=1))
                pre_mask_real_A = self.segnet(real_A)
                pre_mask_real_B = self.segnet(real_B)
            self.flow_A = flow_A
            self.fake_B = fake_B
            self.pre_mask_real_A = pre_mask_real_A
            self.pre_mask_real_B = pre_mask_real_B
            self.theta = self.netG.get_theta(0)
        elif self.model_type=='only_seg':
            with torch.no_grad():
                real_A = Variable(self.input_A)
                real_B = Variable(self.input_B)
                pre_mask_real_A = self.segnet(real_A)
                pre_mask_real_B = self.segnet(real_B)
            self.pre_mask_real_A = pre_mask_real_A
            self.pre_mask_real_B = pre_mask_real_B
        else:
            with torch.no_grad():
                real_A = Variable(self.input_A)
                real_B = Variable(self.input_B)
                fake_B, flow_A = self.netG(torch.cat((real_A, real_B), dim=1).to(torch.device("cuda")))
            self.flow_A = flow_A
            self.fake_B = fake_B
            self.theta = self.netG.get_theta(0)

    def get_image_paths(self):
        return self.image_paths

    def backward_G(self,epoch):
        if self.model_type=='seg_align':
            self.netG.train()
            self.segnet.train()
            lambda_ = self.opt.lambda_R # the weight for supervised loss
            alpha = self.opt.lambda_A
            beta = self.opt.lambda_B
            # Registration loss
            #print('real_a, real_b',self.real_A.shape,self.real_B.shape)
            fake_B, flow_A = self.netG(torch.cat([self.real_A, self.real_B], dim=1))
            pre_mask_real_A = self.segnet(self.real_A)
            pre_mask_real_B = self.segnet(self.real_B)
            pre_mask_fake_B = self.segnet(fake_B)
            fake_mask_B= self.netG.repeat_warp(self.real_mask_A,flow_A)
            fake_pre_mask_real_B = self.netG.repeat_warp(pre_mask_real_A,flow_A)

            lossA_RC = self.criterionCC(fake_B, self.real_B)

            fake_A, flow_B = self.netG(torch.cat([self.real_B, self.real_A], dim=1))
            pre_mask_fake_A = self.segnet(fake_A)
            fake_mask_A = self.netG.repeat_warp(self.real_mask_B,flow_B)
            fake_pre_mask_real_A = self.netG.repeat_warp(pre_mask_real_B,flow_B)
            lossB_RC = self.criterionCC(fake_A, self.real_A)

            # Cycle loss
            back_A, bflow_A = self.netG(torch.cat([fake_B, fake_A], dim=1))
            lossA_CY = self.criterionCy(back_A, self.real_A) * alpha
            back_B, bflow_B = self.netG(torch.cat([fake_A, fake_B], dim=1))
            lossB_CY = self.criterionCy(back_B, self.real_B) * alpha


            ## Identity loss
            idt_A, iflow_A = self.netG(torch.cat([self.real_B, self.real_B], dim=1))
            lossA_ID = self.criterionId(idt_A, self.real_B) * beta
            idt_B, iflow_B = self.netG(torch.cat([self.real_A, self.real_A], dim=1))
            lossB_ID = self.criterionId(idt_B, self.real_A) * beta


            lossA_seg = self.seg(pre_mask_real_A,self.real_mask_A) 
            lossB_seg = self.seg(pre_mask_real_B,self.real_mask_B)
            lossA_segfake = self.seg(pre_mask_fake_A,self.real_mask_A)
            lossB_segfake = self.seg(pre_mask_fake_B,self.real_mask_B)


            lossA_mask = self.criterionMk(self.real_mask_B,fake_mask_B) 
            lossB_mask = self.criterionMk(self.real_mask_A,fake_mask_A) 
            lossA_pre_mask = self.criterionMk(pre_mask_real_A,fake_pre_mask_real_A)
            lossB_pre_mask = self.criterionMk(pre_mask_real_B,fake_pre_mask_real_B)

            if epoch<150:
                # train them separately first
                self.optimizer_G.zero_grad()
                self.optimizer_S.zero_grad()
                loss_align = lossA_RC +  lossB_RC + lossA_CY + lossB_CY + lambda_*(epoch/50*1.0)*(lossA_mask + lossB_mask)
                loss_seg = lossA_seg + lossB_seg
                loss_align.backward()
                loss_seg.backward()
                self.optimizer_G.step()
                self.optimizer_S.step()
            else:
                print('update--both')
                self.optimizer_G.zero_grad()
                self.optimizer_S.zero_grad()
                loss_align = lossA_RC  + lossB_RC + lossA_CY + lossB_CY + lambda_*(lossA_mask + lossB_mask) + lossA_ID + lossB_ID
                loss_seg = (lossA_seg+lossB_seg) + 0.05*(lossA_segfake+lossB_segfake) + 0.05*(lossA_pre_mask+lossB_pre_mask)
                if epoch%2 == 0:
                    loss_align.backward()
                    self.optimizer_G.step()
                else:
                    loss_seg.backward()
                    self.optimizer_S.step()

        elif self.model_type=='only_seg':
            pre_mask_real_A = self.segnet(self.real_A)
            pre_mask_real_B = self.segnet(self.real_B)
            lossA_seg = self.seg(pre_mask_real_A,self.real_mask_A) 
            lossB_seg = self.seg(pre_mask_real_B,self.real_mask_B)
            self.optimizer_S.zero_grad()
            loss_seg = lossA_seg + lossB_seg
            loss_seg.backward()
            self.optimizer_S.step()
        else:
            self.netG.train()
            lambda_ = self.opt.lambda_R
            alpha = self.opt.lambda_A
            beta = self.opt.lambda_B
            # Registration loss
            #print('real_a, real_b',self.real_A.shape,self.real_B.shape)
            fake_B, flow_A = self.netG(torch.cat([self.real_A, self.real_B], dim=1))
            fake_mask_B= self.netG.repeat_warp(self.real_mask_A,flow_A)

            lossA_RC = self.criterionCC(fake_B, self.real_B)

            fake_A, flow_B = self.netG(torch.cat([self.real_B, self.real_A], dim=1))
            fake_mask_A = self.netG.repeat_warp(self.real_mask_B,flow_B)
            lossB_RC = self.criterionCC(fake_A, self.real_A)

            # Cycle loss
            back_A, bflow_A = self.netG(torch.cat([fake_B, fake_A], dim=1))
            lossA_CY = self.criterionCy(back_A, self.real_A) * alpha
            back_B, bflow_B = self.netG(torch.cat([fake_A, fake_B], dim=1))
            lossB_CY = self.criterionCy(back_B, self.real_B) * alpha


            ## Identity loss
            idt_A, iflow_A = self.netG(torch.cat([self.real_B, self.real_B], dim=1))
            lossA_ID = self.criterionId(idt_A, self.real_B) * beta
            idt_B, iflow_B = self.netG(torch.cat([self.real_A, self.real_A], dim=1))
            lossB_ID = self.criterionId(idt_B, self.real_A) * beta

            #fake_mask_B, flow_mask_A = self.netG(torch.cat([self.real_mask_A, self.real_mask_B], dim=1))
            lossA_mask = self.criterionMk(self.real_mask_B,fake_mask_B) 
            #fake_mask_A, flow_mask_B = self.netG(torch.cat([self.real_mask_B, self.real_mask_A], dim=1))
            lossB_mask = self.criterionMk(self.real_mask_A,fake_mask_A) 
            self.optimizer_G.zero_grad()
            if self.model_type=='mask':
                loss = lossA_RC +  lossB_RC + lossA_CY + lossB_CY + (epoch/40*1.0)*(lossA_mask + lossB_mask)
                loss.backward()
            else: 
                loss = lossA_RC + lossA_RL + lossB_RC + lossB_RL + lossA_CY + lossB_CY + lossA_ID + lossB_ID
                loss.backward()
            self.optimizer_G.step()



        if self.model_type=='mask':
            self.lossA_RC = lossA_RC.item()
            self.lossB_RC = lossB_RC.item()
            self.lossA_CY = lossA_CY.item()
            self.lossB_CY = lossB_CY.item()
            self.lossA_ID = lossA_ID.item()
            self.lossB_ID = lossB_ID.item()
            self.back_A = back_A.data 
            self.back_B = back_B.data

            self.flow_A  = flow_A.data
            self.flow_B  = flow_B.data
            self.fake_A = fake_A.data
            self.fake_B = fake_B.data
            self.fake_mask_A = fake_mask_A.data 
            self.fake_mask_B = fake_mask_B.data
            self.loss = loss.item()
        elif self.model_type =='seg_align':
            self.lossA_RC = lossA_RC.item()
            self.lossB_RC = lossB_RC.item()
            self.lossA_CY = lossA_CY.item()
            self.lossB_CY = lossB_CY.item()
            self.lossA_ID = lossA_ID.item()
            self.lossB_ID = lossB_ID.item()

            self.flow_A  = flow_A.data
            self.flow_B  = flow_B.data
            self.fake_A = fake_A.data
            self.fake_B = fake_B.data
            self.back_A = back_A.data 
            self.back_B = back_B.data
            self.fake_mask_A = fake_mask_A.data 
            self.fake_mask_B = fake_mask_B.data
            self.pre_mask_real_A = torch.round(torch.sigmoid(pre_mask_real_A.data))
            self.pre_mask_real_B = torch.round(torch.sigmoid(pre_mask_real_B.data))
            self.pre_mask_fake_A = torch.round(torch.sigmoid(pre_mask_fake_A.data))
            self.pre_mask_fake_B = torch.round(torch.sigmoid(pre_mask_fake_B.data))
            self.lossA_seg = lossA_seg.item()
            self.lossB_seg = lossB_seg.item()
            self.lossA_segfake = lossA_segfake.item()
            self.lossB_segfake = lossB_segfake.item()
            self.lossA_pre_mask = lossA_pre_mask.item()
            self.lossB_pre_mask = lossB_pre_mask.item()
            self.loss = loss_seg.item() + loss_align.item()
        elif self.model_type=='only_seg':
            self.lossA_seg = lossA_seg.item()
            self.lossB_seg = lossB_seg.item()
            self.pre_mask_real_A = torch.round(torch.sigmoid(pre_mask_real_A.data))
            self.pre_mask_real_B = torch.round(torch.sigmoid(pre_mask_real_B.data))
            self.loss = loss_seg.item()
        else:
            self.lossA_RC = lossA_RC.item()
            self.lossB_RC = lossB_RC.item()
            self.lossA_CY = lossA_CY.item()
            self.lossB_CY = lossB_CY.item()
            self.lossA_ID = lossA_ID.item()
            self.lossB_ID = lossB_ID.item()

            self.flow_A  = flow_A.data
            self.flow_B  = flow_B.data
            self.fake_A = fake_A.data
            self.fake_B = fake_B.data
            self.back_A = back_A.data 
            self.back_B = back_B.data
            self.fake_mask_A = fake_A.data
            self.fake_mask_B = fake_B.data
            self.loss = loss.item()
        self.loseA_M = 0
        self.loseB_M = 0
        if self.model_type=='mask' or self.model_type=='seg_align':
            self.loseA_M = lossA_mask.item()
            self.loseB_M = lossB_mask.item()
            print('loss registration------',self.lossA_RC,self.lossB_RC)
        if self.model_type =='seg_align' or self.model_type=='only_seg':
            print('loss dice-----',self.lossA_seg,self.lossB_seg)
        
        #print(self.netG.model.feature_extractor1[0].weight.grad)
        #print('the grad for rotation:',self.netG.model.rotation.weight.grad)
        #print(self.netG.model.feature_extractor[0].weight.grad)

    def optimize_parameters(self, epoch):
        # forward
        self.forward()
        self.backward_G(epoch)
        self.update_learning_rate()

    def get_current_errors(self):
        if self.model_type=='seg_align':
            ret_errors = OrderedDict([('A_RC', self.lossA_RC),
                                      ('B_RC', self.lossB_RC),
                                      ('A_CY', self.lossA_CY), ('B_CY', self.lossB_CY),
                                      ('A_ID', self.lossA_ID), ('B_ID', self.lossB_ID),
                                      ('A_M', self.loseA_M), ('B_M', self.loseB_M),
                                      ('A_seg',self.lossA_seg), ('B_seg',self.lossB_seg),
                                      ('A_segfake',self.lossA_segfake), ('B_segfake',self.lossB_segfake),
                                      ('A_premask',self.lossA_pre_mask), ('B_premask',self.lossB_pre_mask),
                                      ('Tot', self.loss)])
        elif self.model_type=='only_seg':
            ret_errors = OrderedDict([
                          ('A_seg',self.lossA_seg), ('B_seg',self.lossB_seg),
                          ('Tot', self.loss)])
        else: 
            ret_errors = OrderedDict([('A_RC', self.lossA_RC), 
                          ('B_RC', self.lossB_RC),
                          ('A_CY', self.lossA_CY), ('B_CY', self.lossB_CY),
                          ('A_ID', self.lossA_ID), ('B_ID', self.lossB_ID),
                          ('A_M', self.loseA_M), ('B_M', self.loseB_M),
                          ('Tot', self.loss)])
        return ret_errors 

    def get_current_visuals(self):
        realSize = self.input_A.shape
        real_A = util.tensor2im(self.input_A[0, 0, int(realSize[2]/2)])
        real_B = util.tensor2im(self.input_B[0, 0, int(realSize[2]/2)])
        if self.model_type=='seg_align':
            flow_A = self.flow_A.permute(0,4,1,2,3)
            flow_B = self.flow_B.permute(0,4,1,2,3)
            real_mask_A = util.tensor2im(self.mask_A[0, 0, int(realSize[2]/2)])
            fake_mask_A = util.tensor2im(self.fake_mask_A[0, 0, int(realSize[2]/2)])
            flow_A = util.tensor2im(flow_A[0, 0, int(realSize[2]/2)])
            fake_B = util.tensor2im(self.fake_B[0, 0, int(realSize[2]/2)])
            back_A = util.tensor2im(self.back_A[0, 0, int(realSize[2] / 2)])


            real_mask_B = util.tensor2im(self.mask_B[0, 0, int(realSize[2]/2)])

            flow_B = util.tensor2im(flow_B[0, 0, int(realSize[2]/2)])
            fake_A = util.tensor2im(self.fake_A[0, 0, int(realSize[2] / 2)])
            fake_mask_B = util.tensor2im(self.fake_mask_B[0, 0, int(realSize[2]/2)])
            back_B = util.tensor2im(self.back_B[0, 0, int(realSize[2] / 2)])
            pre_mask_real_A = util.tensor2im(self.pre_mask_real_A[0, 0, int(realSize[2]/2)])
            pre_mask_fake_A = util.tensor2im(self.pre_mask_fake_A[0, 0, int(realSize[2]/2)])
            pre_mask_real_B = util.tensor2im(self.pre_mask_real_B[0, 0, int(realSize[2]/2)])
            pre_mask_fake_B = util.tensor2im(self.pre_mask_fake_B[0, 0, int(realSize[2]/2)])


            ret_visuals = OrderedDict([('real_A', real_A), ('flow_A', flow_A),
                                       ('fake_B', fake_B), ('back_A', back_A),
                                       ('real_B', real_B), ('flow_B', flow_B),
                                      ('real_mask_A', real_mask_A), ('real_mask_B', real_mask_B),
                                      ('fake_mask_A', fake_mask_A), ('fake_mask_B', fake_mask_B),
                                      ('pre_mask_real_A', pre_mask_real_A), ('pre_mask_real_B', pre_mask_real_B),
                                       ('pre_mask_fake_A', pre_mask_fake_A), ('pre_mask_fake_B', pre_mask_fake_B),
                                       ('fake_A', fake_A), ('back_B', back_B),
                                       ])
        elif self.model_type=='only_seg':
            real_mask_A = util.tensor2im(self.mask_A[0, 0, int(realSize[2]/2)])
            real_mask_B = util.tensor2im(self.mask_B[0, 0, int(realSize[2]/2)])
            pre_mask_real_A = util.tensor2im(self.pre_mask_real_A[0, 0, int(realSize[2]/2)])
            pre_mask_real_B = util.tensor2im(self.pre_mask_real_B[0, 0, int(realSize[2]/2)])
            ret_visuals = OrderedDict([('real_A', real_A), 
                           ('real_B', real_B),
                          ('real_mask_A', real_mask_A), ('real_mask_B', real_mask_B),
                          ('pre_mask_real_A', pre_mask_real_A), ('pre_mask_real_B', pre_mask_real_B),
                           ])

        else:
            flow_A = self.flow_A.permute(0,4,1,2,3)
            flow_B = self.flow_B.permute(0,4,1,2,3)
            fake_B = util.tensor2im(self.fake_B[0, 0, int(realSize[2]/2)])
            fake_A = util.tensor2im(self.fake_A[0, 0, int(realSize[2] / 2)])
            real_mask_A = util.tensor2im(self.mask_A[0, 0, int(realSize[2]/2)])
            real_mask_B = util.tensor2im(self.mask_B[0, 0, int(realSize[2]/2)])
            fake_mask_A = util.tensor2im(self.fake_mask_A[0, 0, int(realSize[2]/2)])
            fake_mask_B = util.tensor2im(self.fake_mask_B[0, 0, int(realSize[2]/2)])
            flow_A = util.tensor2im(flow_A[0, 0, int(realSize[2]/2)])
            flow_B = util.tensor2im(flow_B[0, 0, int(realSize[2]/2)])
            back_A = util.tensor2im(self.back_A[0, 0, int(realSize[2] / 2)])
            back_B = util.tensor2im(self.back_B[0, 0, int(realSize[2] / 2)])
            ret_visuals = OrderedDict([('real_A', real_A), ('flow_A', flow_A),
                           ('fake_B', fake_B), ('back_A', back_A),
                           ('real_B', real_B), ('flow_B', flow_B),
                          ('real_mask_A', real_mask_A), ('real_mask_B', real_mask_B),
                          ('fake_mask_A', fake_mask_A), ('fake_mask_B', fake_mask_B),
                           ('fake_A', fake_A), ('back_B', back_B),
                           ])
        return ret_visuals

    def get_current_data(self):
        if self.model_type=='seg_align':
            ret_visuals = OrderedDict([('flow_A', self.flow_A),('fake_B', self.fake_B),('theta',self.theta),('seg_A',self.pre_mask_real_A),('seg_B',self.pre_mask_real_B)])
        elif self.model_type =='only_seg':
            ret_visuals = OrderedDict([('seg_A',self.pre_mask_real_A),('seg_B',self.pre_mask_real_B)])
        else:
            ret_visuals = OrderedDict([('flow_A', self.flow_A),('fake_B', self.fake_B),('theta',self.theta)])
        return ret_visuals

    def get_test_data(self):
        ret_visuals = OrderedDict([('flow_A', self.flow_A),('fake_B', self.fake_B),('theta',self.theta)])
        return ret_visuals

    def save(self, label):
        if self.model_type=='seg_align':
            self.save_network(self.netG, 'G', label, self.gpu_ids,self.path_G)
            self.save_network(self.segnet,'S', label, self.gpu_ids,self.path_S)
        elif self.model_type=='only_seg':
            self.save_network(self.segnet,'S', label, self.gpu_ids,self.path_S)
        else:
            self.save_network(self.netG, 'G', label, self.gpu_ids,self.path_G)
        #self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
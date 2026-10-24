# -*- coding: utf-8 -*-
import os
import numpy as np
import math
import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable

class Segloss(nn.Module):
    def __init__(self):
        super(Segloss, self).__init__()
        self.CE = nn.BCELoss()
        
    def Dice(self, segment, label, return_dice=False):
        N = segment.shape[0]
        segment_flat = segment.view(N, -1)
        label_flat = label.view(N, -1)
        intersection = segment_flat * label_flat 
        dice = (2 * intersection.sum(1) + 0.001) / (segment_flat.sum(1) + label_flat.sum(1) + 0.001)
        if return_dice:
            return dice.mean()
        loss = 1 - dice.mean()
        return loss
    
    def Gradient(self, segment):
        g_H = (segment[:, :, 1:, :, :]-segment[:, :, :-1, :, :]).abs()
        g_W = (segment[:, :, :, 1:, :]-segment[:, :, :, :-1, :]).abs()
        g_D = (segment[:, :, :, :, 1:]-segment[:, :, :, :, :-1]).abs()
        loss = g_H.mean() + g_W.mean() + g_D.mean()
        return loss

    def forward(self, segments, label_s, weight=[1,1,1]):
        labels = [F.interpolate(label_s, scale_factor=1/8),
                  F.interpolate(label_s, scale_factor=1/4),
                  F.interpolate(label_s, scale_factor=1/2),
                  label_s]
        CEloss = torch.tensor(0.).to(segments[0].device)
        Diceloss = torch.tensor(0.).to(segments[0].device)
        Gradientloss = torch.tensor(0.).to(segments[0].device)
        for segment, label in zip(segments, labels):
            CEloss += self.CE(segment, label)*weight[0]
            Diceloss += self.Dice(segment, label)*weight[1]
            Gradientloss += self.Gradient(segment)*weight[2]
        loss = CEloss + Diceloss + Gradientloss
        return loss, [CEloss.item(), Diceloss.item(), Gradientloss.item()], self.Dice(segments[-1], labels[-1], return_dice=True)

class Regloss(nn.Module):
    def __init__(self):
        super(Regloss, self).__init__()
        self.MSE = nn.MSELoss()
        self.act = nn.ReLU(inplace=True)

    def Warper3d(self, img, flow):
        B, _, D, H, W = img.shape
        # mesh grid
        xx = torch.arange(0, W).view(1,1,-1).repeat(D,H,1).view(1,D,H,W)
        yy = torch.arange(0, H).view(1,-1,1).repeat(D,1,W).view(1,D,H,W)
        zz = torch.arange(0, D).view(-1,1,1).repeat(1,H,W).view(1,D,H,W)
        grid = torch.cat((xx,yy,zz),0).repeat(B,1,1,1,1).float().to(img.device) # [bs, 3, D, H, W]
        vgrid = grid + flow
        # scale grid to [-1,1]
        vgrid[:,0] = 2.0*vgrid[:,0]/(W-1)-1.0 # max(W-1,1)
        vgrid[:,1] = 2.0*vgrid[:,1]/(H-1)-1.0 # max(H-1,1)
        vgrid[:,2] = 2.0*vgrid[:,2]/(D-1)-1.0 # max(D-1,1)
        vgrid = vgrid.permute(0,2,3,4,1) # [bs, D, H, W, 3]        
        output = F.grid_sample(img, vgrid, padding_mode='border')

        return output
        
    def Dice(self, segment, label, return_dice=False):
        N = segment.shape[0]
        segment_flat = segment.view(N, -1)
        label_flat = label.view(N, -1)
        intersection = segment_flat * label_flat 
        dice = (2 * intersection.sum(1) + 0.001) / (segment_flat.sum(1) + label_flat.sum(1) + 0.001)
        if return_dice:
            return dice.mean()
        loss = 1 - dice.mean()
        return loss
    
    def Gradient(self, feild):
        g_H = (feild[:, :, 1:, :, :]-feild[:, :, :-1, :, :]).abs()
        g_W = (feild[:, :, :, 1:, :]-feild[:, :, :, :-1, :]).abs()
        g_D = (feild[:, :, :, :, 1:]-feild[:, :, :, :, :-1]).abs()
        loss = g_H.mean() + g_W.mean() + g_D.mean()
        return loss

    def Jac(self, feild):
        D_x = (feild[:, :, :-1, :-1, 1:]-feild[:, :, :-1, :-1, :-1])
        D_y = (feild[:, :, :-1, 1:, :-1]-feild[:, :, :-1, :-1, :-1])
        D_z = (feild[:, :, 1:, :-1, :-1]-feild[:, :, :-1, :-1, :-1])

        D1 = (D_x[:,0,...]+1)*((D_y[:,1,...]+1)*(D_z[:,2,...]+1) - D_y[:,2,...]*D_z[:,1,...])
        D2 = (D_x[:,1,...])*(D_y[:,2,...]*D_z[:,0,...] - D_y[:,0,...]*(D_z[:,2,...]+1))
        D3 = (D_x[:,2,...])*(D_y[:,0,...]*D_z[:,1,...] - (D_y[:,1,...]+1)*D_z[:,0,...])

        D = D1 + D2 + D3

        return D

    def Jac_loss(self, feild):
        Jac_mat = self.Jac(feild)
        loss = self.act(-Jac_mat).mean()
        return loss

    def forward(self, Rs, S1_d, S2_d, weight=[1,1,0.1]):
        S1_ = [F.interpolate(S1_d, scale_factor=1/8),
               F.interpolate(S1_d, scale_factor=1/4),
               F.interpolate(S1_d, scale_factor=1/2),
               S1_d]
        S2_ = [F.interpolate(S2_d, scale_factor=1/8),
               F.interpolate(S2_d, scale_factor=1/4),
               F.interpolate(S2_d, scale_factor=1/2),
               S2_d]
        Diceloss = torch.tensor(0.).to(Rs[0].device)
        MSEloss = torch.tensor(0.).to(Rs[0].device)
        Gradientloss = torch.tensor(0.).to(Rs[0].device)
        JACloss = torch.tensor(0.).to(Rs[0].device)
        for R, S1, S2, in zip(Rs, S1_, S2_):
            nonrigid_S1 = self.Warper3d(S1, R)
            Diceloss += self.Dice(nonrigid_S1, S2)*weight[0]
            MSEloss += self.MSE(nonrigid_S1, S2)*weight[1]
            Gradientloss += self.Gradient(R)*weight[2]
            JACloss += self.Jac_loss(R)*weight[2]
        loss = Diceloss + MSEloss + Gradientloss + JACloss
        return loss, [Diceloss.item(), MSEloss.item(), Gradientloss.item(), JACloss.item()], self.Dice(self.Warper3d(S1_[-1], Rs[-1]), S2_[-1], return_dice=True)


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"]="1"
    device = torch.device("cuda")
    losser = Segloss()
    x = torch.rand((1,1,64,64,64)).to(device)
    y = torch.zeros((1,1,64,64,64)).to(device)
    z = [torch.zeros((1,1,8,8,8)).to(device),
         torch.zeros((1,1,16,16,16)).to(device),
         torch.zeros((1,1,32,32,32)).to(device),
         torch.zeros((1,1,64,64,64)).to(device)]
    print(losser(x,z,y))


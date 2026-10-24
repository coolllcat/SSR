
import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba

device = torch.device("cuda:1")

bn_mom = 0.1

class Conv3x3(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, no_relu=False, Trans=False):
        super(Conv3x3, self).__init__()
        if Trans:
            self.conv = nn.ConvTranspose3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, output_padding=kernel_size//2, bias=True)
        else:
            self.conv = nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=kernel_size//2, bias=True)
        self.bn = nn.BatchNorm3d(out_planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.no_relu = no_relu

    def forward(self, x):
        out = self.bn(self.conv(x))
        if self.no_relu:
            return out
        else:
            return self.relu(out)

class ResBlock(nn.Module):
    def __init__(self, planes, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = Conv3x3(planes, planes, stride, no_relu=False)
        self.conv2 = Conv3x3(planes, planes, stride, no_relu=True)
        self.conv3 = Conv3x3(planes, planes, stride, no_relu=False)
        self.conv4 = Conv3x3(planes, planes, stride, no_relu=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual1 = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += residual1
        out = self.relu(out)

        residual2 = out
        out = self.conv3(out)
        out = self.conv4(out)
        out += residual2
        out = self.relu(out)

        return out
    
class DAModule(nn.Module):
    def __init__(self, dim, window_size, d_state = 32, d_conv = 4, expand = 1):
        super(DAModule, self).__init__()
        self.window_size = window_size
        self.res_block = ResBlock(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mamba1_1 = Mamba(
            d_model=dim,
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand)
        self.mamba1_2 = Mamba(
            d_model=dim,
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand)
        self.mamba2_1 = Mamba(
            d_model=dim,
            d_state=d_state, 
            d_conv=d_conv, 
            expand=expand)
        self.mamba2_2 = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand)
        
    def window_partition(self, x, window_size):
        B, C, H, W, D = x.shape
        x = x.view(B, C, H // window_size, window_size, W // window_size, window_size, D // window_size, window_size)
        windows = x.permute(0, 1, 2, 4, 6, 3, 5, 7).contiguous().view(B, C, -1, window_size**3)
        return windows # (B, C, windows_num, window_size**3)

    def window_reverse(self, windows, window_size, H, W, D):
        B, C, _, _ = windows.shape
        x = windows.view(B, C, H // window_size, W // window_size, D // window_size, window_size, window_size, window_size)
        x = x.permute(0, 1, 2, 5, 3, 6, 4, 7).contiguous().view(B, C, H, W, D)
        return x

    def forward(self, x):
        B, C, H, W, D = x.shape
        x = self.res_block(x)

        x1 = self.window_partition(x, window_size=self.window_size)
        x1 = x1.permute(0,2,3,1).reshape(-1, self.window_size**3, C)
        x1 = self.norm1(x1)
        x1_out1 = self.mamba1_1(x1)
        x1_rev = torch.flip(x1, [1])
        x1_out2 = self.mamba1_2(x1_rev)
        x1_out2 = torch.flip(x1_out2, [1])
        x1_out = x1_out1 + x1_out2

        x2 = x1_out.reshape(B, -1, self.window_size**3, C).permute(0,2,1,3).reshape(B*self.window_size**3, -1, C)
        x2 = self.norm2(x2)
        x2_out1 = self.mamba2_1(x2)
        x2_rev = torch.flip(x2, [1])
        x2_out2 = self.mamba2_2(x2_rev)
        x2_out2 = torch.flip(x2_out2, [1])
        x2_out = x2_out1 + x2_out2

        y = x2_out.reshape(B, self.window_size**3, -1, C).permute(0,3,2,1)
        y = self.window_reverse(y, self.window_size, H, W, D)
        y = y + x

        return y

class Gate(nn.Module):
    def __init__(self, inp_dim, out_dim):
        super(Gate, self).__init__()
        self.conv = Conv3x3(inp_dim, out_dim, stride=2)
        self.res1 = ResBlock(out_dim)
        self.res2 = ResBlock(out_dim)
        self.res3 = ResBlock(out_dim)
        self.conv_out = Conv3x3(out_dim*3, out_dim, stride=1)

    def forward(self, x, fea1, fea2):
        x1 = self.res1(self.conv(x))
        x2 = self.res2(fea1)
        x3 = self.res3(fea2)
        out = self.conv_out(torch.cat([x1, x2, x3], dim=1))

        return out
    
class Att(nn.Module):
    def __init__(self, dim):
        super(Att, self).__init__()
        self.w_E = Conv3x3(dim, dim//2, stride=1, kernel_size=1, no_relu=True)
        self.w_D = Conv3x3(dim, dim//2, stride=1, kernel_size=1, no_relu=True)
        self.w_W = Conv3x3(dim//2, 1, stride=1, kernel_size=1, no_relu=True)
        self.pro = Conv3x3(dim*2, dim)
        self.relu = nn.ReLU(inplace=True)
        self.acti = nn.Sigmoid()

    def forward(self, E, D):
        w1 = self.w_E(E)
        w2 = self.w_D(D)
        ps = self.relu(w1 + w2)
        pw = self.acti(self.w_W(ps))
        out = self.pro(torch.cat((E*pw, D), dim=1))

        return out

class CAModule(nn.Module):
    def __init__(self, dim, dim_field=3, num_heads=4, patch_size=6):
        super(CAModule, self).__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.scale = (patch_size**3)**-0.5

        self.res = ResBlock(dim*2)
        self.up = Conv3x3(dim*2, dim, stride=2, Trans=True)

        self.norm1 = nn.LayerNorm(patch_size**3)
        self.norm2 = nn.LayerNorm(patch_size**3)
        self.embedding1 = nn.Linear(patch_size**3, patch_size**3)
        self.embedding2_1 = nn.Linear(patch_size**3, patch_size**3)
        self.embedding2_2 = nn.Linear(patch_size**3, patch_size**3)
        self.projection = nn.Linear(patch_size**3, patch_size**3)

        self.norm3 = nn.LayerNorm(dim)
        self.conv1 = nn.Sequential(nn.Conv3d(dim, dim//2, 1),
                                   nn.Conv3d(dim//2, dim_field, 1))
        self.conv2 = nn.Sequential(nn.Conv3d(dim_field, dim//2, 1),
                                   nn.Conv3d(dim//2, dim, 1))

    def Patch(self, x):
        B, C, H, W, D = x.shape
        x = x.reshape(B, C, H//self.patch_size, self.patch_size, W//self.patch_size, self.patch_size, D//self.patch_size, self.patch_size)
        x = x.permute(0, 2, 4, 6, 1, 3, 5, 7) # B*H'*W'*D'*C*k*k*k
        x = x.reshape(B, H*W*D//(self.patch_size**3), C, self.patch_size**3)
        return x

    def unPatch(self, x, B, C, H, W, D):
        x = x.reshape(B, H//self.patch_size, W//self.patch_size, D//self.patch_size, C, self.patch_size, self.patch_size, self.patch_size)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7) # B*C*H'*k*W'*k*D'*k
        x = x.reshape(B, C, H, W, D)
        return x

    def forward(self, x1, x2):
        x1 = self.up(self.res(x1))

        B, C, H, W, D = x1.shape
        x1_ = self.norm1(self.Patch(x1)) # B*N*C*d
        x2_ = self.norm2(self.Patch(x2)) # B*N*C*d
        B1, N1, C1, d1 = x1_.shape
        B2, N2, C2, d2 = x2_.shape
        q = self.embedding1(x1_).reshape(B1, N1, C1, self.num_heads, d1//self.num_heads).permute(0,1,3,2,4) # B*N*h*C*d
        k = self.embedding2_1(x2_).reshape(B2, N2, C2, self.num_heads, d2//self.num_heads).permute(0,1,3,2,4) # B*N*h*C*d
        v = self.embedding2_2(x2_).reshape(B2, N2, C2, self.num_heads, d2//self.num_heads).permute(0,1,3,2,4) # B*N*h*C*d

        attn_score = torch.einsum('bnhld,bnhtd->bnhlt', [q, k]) * self.scale # B*N*h*C*2
        attn = F.softmax(attn_score, dim=-1)
        x_ = torch.einsum('bnhlt,bnhtv->bnhlv', [attn, v]) # B*N*h*C*d
        x = self.projection(x_.permute(0,1,3,2,4).reshape(B1, N1, C1, d1))
        x = self.unPatch(x, B, C, H, W, D)

        y = x + x1
        z = self.norm3(y.permute(0, 2, 3, 4, 1)).permute(0, 4, 1, 2, 3)
        field = self.conv1(z)
        out = self.conv2(field) + y

        return out, field

class DAGNet(nn.Module):
    def __init__(self, dim_base=32, dim_guide=2, dim_field=3):
        super(DAGNet, self).__init__()
        self.dims = [dim_base, dim_base*2, dim_base*4, dim_base*8, dim_base*16]
        self.conv1 = Conv3x3(1, self.dims[0], kernel_size=7)
        self.conv2 = Conv3x3(1, self.dims[0], kernel_size=7)
        self.down1_1 = Conv3x3(self.dims[0], self.dims[1], stride=2)
        self.down1_2 = Conv3x3(self.dims[1], self.dims[2], stride=2)
        self.down1_3 = Conv3x3(self.dims[2], self.dims[3], stride=2)
        self.down1_4 = Conv3x3(self.dims[3], self.dims[4], stride=2)
        self.down2_1 = Conv3x3(self.dims[0], self.dims[1], stride=2)
        self.down2_2 = Conv3x3(self.dims[1], self.dims[2], stride=2)
        self.down2_3 = Conv3x3(self.dims[2], self.dims[3], stride=2)
        self.down2_4 = Conv3x3(self.dims[3], self.dims[4], stride=2)
        self.up1_4 = Conv3x3(self.dims[4], self.dims[3], stride=2, Trans=True)
        self.up1_3 = Conv3x3(self.dims[3], self.dims[2], stride=2, Trans=True)
        self.up1_2 = Conv3x3(self.dims[2], self.dims[1], stride=2, Trans=True)
        self.up1_1 = Conv3x3(self.dims[1], self.dims[0], stride=2, Trans=True)
        self.up2_4 = Conv3x3(self.dims[4], self.dims[3], stride=2, Trans=True)
        self.up2_3 = Conv3x3(self.dims[3], self.dims[2], stride=2, Trans=True)
        self.up2_2 = Conv3x3(self.dims[2], self.dims[1], stride=2, Trans=True)
        self.up2_1 = Conv3x3(self.dims[1], self.dims[0], stride=2, Trans=True)
        self.skip1_4 = Att(self.dims[3])
        self.skip1_3 = Att(self.dims[2])
        self.skip1_2 = Att(self.dims[1])
        self.skip1_1 = Att(self.dims[0])
        self.skip2_4 = Att(self.dims[3])
        self.skip2_3 = Att(self.dims[2])
        self.skip2_2 = Att(self.dims[1])
        self.skip2_1 = Att(self.dims[0])
        self.res1_4 = ResBlock(self.dims[3])
        self.res1_3 = ResBlock(self.dims[2])
        self.res1_2 = ResBlock(self.dims[1])
        self.res1_1 = ResBlock(self.dims[0])
        self.res2_4 = ResBlock(self.dims[3])
        self.res2_3 = ResBlock(self.dims[2])
        self.res2_2 = ResBlock(self.dims[1])
        self.res2_1 = ResBlock(self.dims[0])
        self.conv3_1 = Conv3x3(self.dims[4], self.dims[4])
        self.conv3_2 = Conv3x3(self.dims[4], self.dims[4])
        self.conv3_3 = Conv3x3(self.dims[4], self.dims[4])
        self.gat1 = Gate(2, self.dims[1])
        self.gat2 = Gate(self.dims[1], self.dims[2])
        self.gat3 = Gate(self.dims[2], self.dims[3])
        self.gat4 = Gate(self.dims[3], self.dims[4])
        self.DA1_1 = DAModule(self.dims[1], window_size=8)
        self.DA1_2 = DAModule(self.dims[2], window_size=4)
        self.DA1_3 = DAModule(self.dims[3], window_size=4)
        self.DA1_4 = DAModule(self.dims[4], window_size=2)
        self.DA2_1 = DAModule(self.dims[1], window_size=8)
        self.DA2_2 = DAModule(self.dims[2], window_size=4)
        self.DA2_3 = DAModule(self.dims[3], window_size=4)
        self.DA2_4 = DAModule(self.dims[4], window_size=2)
        self.CA4 = CAModule(self.dims[3])
        self.CA3 = CAModule(self.dims[2])
        self.CA2 = CAModule(self.dims[1])
        self.CA1 = CAModule(self.dims[0])
        self.Seg_head4 = nn.Conv3d(self.dims[3], 1, kernel_size=1)
        self.Seg_head3 = nn.Conv3d(self.dims[2], 1, kernel_size=1)
        self.Seg_head2 = nn.Conv3d(self.dims[1], 1, kernel_size=1)
        self.Seg_head1 = nn.Conv3d(self.dims[0], 1, kernel_size=1)
        self.act = nn.Sigmoid()

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
        output = F.grid_sample(img, vgrid, padding_mode='border', align_corners=True)

        return output

    def forward(self, x1, x2):
        F_f1 = self.conv1(x1)
        F_f2 = self.DA1_1(self.down1_1(F_f1))
        F_f3 = self.DA1_2(self.down1_2(F_f2))
        F_f4 = self.DA1_3(self.down1_3(F_f3))
        F_f5 = self.DA1_4(self.down1_4(F_f4))
        F_m1 = self.conv2(x2)
        F_m2 = self.DA2_1(self.down2_1(F_m1))
        F_m3 = self.DA2_2(self.down2_2(F_m2))
        F_m4 = self.DA2_3(self.down2_3(F_m3))
        F_m5 = self.DA2_4(self.down2_4(F_m4))
        F_e1 = torch.cat([x1, x2], dim=1)
        F_e2 = self.gat1(F_e1, F_f2.detach(), F_m2.detach())
        F_e3 = self.gat2(F_e2, F_f3.detach(), F_m3.detach())
        F_e4 = self.gat3(F_e3, F_f4.detach(), F_m4.detach())
        F_e5 = self.gat4(F_e4, F_f5.detach(), F_m5.detach())

        Fd1_5 = self.conv3_1(F_f5)
        Fd2_5 = self.conv3_2(F_m5)
        Fd3_5 = self.conv3_3(F_e5)

        Fd1_4 = self.res1_4(self.skip1_4(F_f4, self.up1_4(Fd1_5)))
        Fd1_3 = self.res1_3(self.skip1_3(F_f3, self.up1_3(Fd1_4)))
        Fd1_2 = self.res1_2(self.skip1_2(F_f2, self.up1_2(Fd1_3)))
        Fd1_1 = self.res1_1(self.skip1_1(F_f1, self.up1_1(Fd1_2)))
        Fd2_4 = self.res2_4(self.skip2_4(F_m4, self.up2_4(Fd2_5)))
        Fd2_3 = self.res2_3(self.skip2_3(F_m3, self.up2_3(Fd2_4)))
        Fd2_2 = self.res2_2(self.skip2_2(F_m2, self.up2_2(Fd2_3)))
        Fd2_1 = self.res2_1(self.skip2_1(F_m1, self.up2_1(Fd2_2)))
        S1_4 = self.act(self.Seg_head4(Fd1_4))
        S1_3 = self.act(self.Seg_head3(Fd1_3))
        S1_2 = self.act(self.Seg_head2(Fd1_2))
        S1_1 = self.act(self.Seg_head1(Fd1_1))
        S2_4 = self.act(self.Seg_head4(Fd2_4))
        S2_3 = self.act(self.Seg_head3(Fd2_3))
        S2_2 = self.act(self.Seg_head2(Fd2_2))
        S2_1 = self.act(self.Seg_head1(Fd2_1))

        Fd3_4, R4 = self.CA4(Fd3_5, torch.cat([S1_4.detach(), S2_4.detach()], dim=1))
        Fd3_3, R3 = self.CA3(Fd3_4, torch.cat([S1_3.detach(), S2_3.detach()], dim=1))
        Fd3_2, R2 = self.CA2(Fd3_3, torch.cat([S1_2.detach(), S2_2.detach()], dim=1))
        Fd3_1, R1 = self.CA1(Fd3_2, torch.cat([S1_1.detach(), S2_1.detach()], dim=1))
        R4_ = R4
        R3_ = self.Warper3d(F.interpolate(R4_, scale_factor=2, mode='trilinear', align_corners=True), R3) + R3
        R2_ = self.Warper3d(F.interpolate(R3_, scale_factor=2, mode='trilinear', align_corners=True), R2) + R2
        R1_ = self.Warper3d(F.interpolate(R2_, scale_factor=2, mode='trilinear', align_corners=True), R1) + R1

        return [S1_4, S1_3, S1_2, S1_1], [S2_4, S2_3, S2_2, S2_1], [R4_, R3_, R2_, R1_]


if __name__ == '__main__':
    import time
    model = DAGNet().to(device)
    x1 = torch.randn((1,1,96,96,96)).to(device)
    x2 = torch.randn((1,1,96,96,96)).to(device)
    # S1, S2, R1 = model(x1, x2)
    # print(S1[-1].shape, S2[-1].shape, R1[-1].shape)

    tic = time.time()
    S1, S2, R1 = model(x1, x2)
    print(time.time() - tic)
    tic = time.time()
    S1, S2, R1 = model(x1, x2)
    print(time.time() - tic)

    


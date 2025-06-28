
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
from models.EEGViT import EEGViT

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class SelfAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super(SelfAttention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.to_Q = nn.Linear(dim, dim, bias=False)
        self.to_K = nn.Linear(dim, dim, bias=False)
        self.to_V = nn.Linear(dim, dim, bias=False)
        # self.norm = nn.LayerNorm(eeg_dim)

    def forward(self, x):
        b, n_x, d_x, h_x = *x.shape, self.heads
        q = self.to_Q(x).view(b, -1, 1, d_x).transpose(1, 2)
        q = q.repeat(1, self.heads, 1, 1)
        k = self.to_K(x).view(b, -1, self.heads, d_x // self.heads).transpose(1, 2)

        kkt = einsum('b h i d, b h j d -> b h i j', k, k) * self.scale

        dots = einsum('b h i d, b h j d -> b h i j', q.transpose(2, 3), kkt) * self.scale

        attn = F.softplus(dots)
        attn = torch.flatten(attn, 1)
        attn = attn.reshape(40, 32, self.dim)
        return attn

class CrossAttention(nn.Module):
    def __init__(self, face_dim, eeg_dim, heads, dim_head):
        super(CrossAttention, self).__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = face_dim
        self.to_Q = nn.Linear(face_dim, face_dim, bias=False)
        self.to_K = nn.Linear(eeg_dim, eeg_dim, bias=False)
        self.to_V = nn.Linear(eeg_dim, eeg_dim, bias=False)
        # self.norm = nn.LayerNorm(eeg_dim)

    def forward(self, x, y):
        b, n_x, d_x, h_x = *x.shape, self.heads
        b, n_y, d_y, h_y = *y.shape, self.heads

        q = self.to_Q(y).view(b, -1, 1, d_y).transpose(1, 2)
        q = q.repeat(1, self.heads, 1, 1)
        k = self.to_K(x).view(b, -1, self.heads, d_x // self.heads).transpose(1, 2)

        kkt = einsum('b h i d, b h j d -> b h i j', k, k) * self.scale

        dots = einsum('b h i d, b h j d -> b h i j', q.transpose(2, 3), kkt) * self.scale

        attn = F.softplus(dots)
        attn = torch.flatten(attn, 1)
        attn = attn.reshape(40, 32, self.dim)
        return attn


class PriorInfoLayer(nn.Module):
    def __init__(self, face_dim, eeg_dim, heads, dim_head):
        super(PriorInfoLayer, self).__init__()
        self.CrossAttentionFE = CrossAttention(face_dim, eeg_dim, heads, dim_head)
        self.CrossAttentionEF = CrossAttention(eeg_dim, face_dim, heads, dim_head)
        self.Projection_FE = nn.Linear(2, 20)
        self.Projection_EF = nn.Linear(11, 20)
        self.conv = nn.Conv2d(in_channels=64,out_channels=32,kernel_size=(1,1))
        self.softplus = nn.Softplus()

    def forward(self, x, y):

        CsFE = self.CrossAttentionFE(x, y)
        CsEF = self.CrossAttentionEF(y, x)

        CsFE = self.Projection_FE(CsFE)
        CsEF = self.Projection_EF(CsEF)
        CsFE = CsFE.permute(0,2,1).reshape(40, 20, 32, 1)
        CsEF = CsEF.permute(0,2,1).reshape(40, 20, 32, 1)

        out = CsFE + CsEF
        out = self.softplus(out)
        return out


class eeg_expert(nn.Module):
    def __init__(self):
        super(eeg_expert, self).__init__()
        self.Projection_eeg = nn.Linear(11, 20)
        self.classfier = nn.Sequential(
            nn.Linear(640, 2)
        )
    def forward(self,x):
        out = self.Projection_eeg(x)
        out = out.permute(0,2,1)
        out_feature = torch.flatten(out, 1)
        out = self.classfier(out_feature)
        return out_feature,out

class face_expert(nn.Module):
    def __init__(self):
        super(face_expert, self).__init__()
        self.Projection_eeg = nn.Linear(2, 20)
        self.classfier = nn.Sequential(
            nn.Linear(640, 2),
        )
    def forward(self,x):
        out = self.Projection_eeg(x)
        out = out.permute(0, 2, 1)
        out_feature = torch.flatten(out, 1)

        out = self.classfier(out_feature)
        return out_feature,out
class eegfacefusion_expert(nn.Module):
    def __init__(self):
        super(eegfacefusion_expert, self).__init__()
        self.classfier = nn.Sequential(

            nn.Linear(640, 2)
        )

        self.feature = nn.Linear(20480,640)

    def forward(self,x):
        out = torch.flatten(x, 1)
        out_feature = self.feature(out)
        out = self.classfier(out_feature)
        return out_feature,out
class GateNetwork(nn.Module):
    def __init__(self,input_dim,num_experts):
        super(GateNetwork, self).__init__()
        self.fc = nn.Linear(input_dim,num_experts)
    def forward(self,x):
        x = x.unsqueeze(1)
        avgpool = F.adaptive_avg_pool1d(x,1920)
        avgpool = avgpool.squeeze(1)
        weights = F.softmax(self.fc(avgpool),dim=1)
        return weights

class MOE(nn.Module):
    def __init__(self):
        super(MOE, self).__init__()
        self.eeg_expert = eeg_expert()
        self.face_expert = face_expert()
        self.eegfacefusion_expert = eegfacefusion_expert()
        self.gatenetwork = GateNetwork(input_dim=1920,num_experts=3)
        self.eegavg = nn.AvgPool1d(kernel_size=2)

    def forward(self,eeg_feature,face_feature,fusion_feature):
        out_eeg_feature,eeg_out = self.eeg_expert(eeg_feature)
        out_face_feature,face_out = self.face_expert(face_feature)
        out_fusion_feature,fusion_out = self.eegfacefusion_expert(fusion_feature)


        gationg_weight = torch.cat([out_eeg_feature,out_face_feature,out_fusion_feature],dim=1)


        weight = self.gatenetwork(gationg_weight)

        out = weight[:,0].unsqueeze(1)*eeg_out+ weight[:,1].unsqueeze(1)*face_out+ weight[:,2].unsqueeze(1)*fusion_out

        return out_eeg_feature, eeg_out,out_face_feature,face_out,out
class FeedForeard(nn.Module):
    def __init__(self,dim,hideen_dim,dropout):
        super(FeedForeard, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim,hideen_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hideen_dim,dim)
        )
        self.norm = nn.LayerNorm(dim)
    def forward (self,x):
        x=self.net(x)
        x=self.norm(x)
        return x
class net(nn.Module):
    def __init__(self, face_dim, eeg_dim, heads, dim_head, inplanes, planes, num_classes):
        super().__init__()
        self.num_classes = num_classes
        self.linear_face = nn.Linear(580, 100)
        self.PriorInfo = PriorInfoLayer(face_dim, eeg_dim, heads, dim_head)
        self.fc = nn.Linear(320, num_classes)
        self.eeg_encoder = EEGViT(num_chan=20, num_time=160, num_patches=10,num_classes=2)
        self.face_encoder = EEGViT(num_chan=20, num_time=29, num_patches=1,num_classes=2)

        self.moe = MOE()
        self.dropeeg = nn.Dropout(0.5)
        self.dropface = nn.Dropout(0.5)


    def forward(self, dFCN, eeg, face):

        eeg_feature = self.eeg_encoder(eeg)
        face_feature = self.face_encoder(face)

        eeg_feature = eeg_feature.permute(0,2,1)
        face_feature = face_feature.permute(0,2,1)

        PriorInfo = self.PriorInfo(eeg_feature, face_feature)


        dFCN = dFCN * PriorInfo

        outeeg_feature,outeeg,outface_feature,outface,out = self.moe(eeg_feature,face_feature,dFCN)

        outeeg_feature = self.dropeeg(outeeg_feature)
        outface_feature = self.dropface(outface_feature)
        return outeeg_feature,outeeg,outface_feature,outface,out



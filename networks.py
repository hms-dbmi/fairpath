from torch import Tensor
import torch
import torch.nn as nn
import torchvision.models as models

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

class MILAttention(nn.Module):
    def __init__(self, featureLength = 512, featureInside = 256):
        '''
        Parameters:
            featureLength: Length of feature passed in from feature extractor(encoder)
            featureInside: Length of feature in MIL linear
        Output: tensor
            weight of the features
        '''
        super(MILAttention, self).__init__()
        self.featureLength = featureLength
        self.featureInside = featureInside

        self.attetion_V = nn.Sequential(
            nn.Linear(self.featureLength, self.featureInside, bias = True),
            nn.Tanh()
        )
        self.attetion_U = nn.Sequential(
            nn.Linear(self.featureLength, self.featureInside, bias = True),
            nn.Sigmoid()
        )
        self.attetion_weights = nn.Linear(self.featureInside, 1, bias = True)
        self.softmax = nn.Softmax(dim = 1)
        
    def forward(self, x: Tensor) -> Tensor:
        bz, pz, fz = x.shape
        x = x.view(bz*pz, fz)
        att_v = self.attetion_V(x)
        att_u = self.attetion_U(x)
        att = self.attetion_weights(att_u*att_v)
        weight = att.view(bz, pz, 1)
        
        weight = self.softmax(weight)
        weight = weight.view(bz, 1, pz)

        return weight

class MILNet(nn.Module):
    def __init__(self, featureLength = 512, linearLength = 256, pretrained = True):
        '''
        Parameters:
            featureLength: Length of feature from resnet18
            linearLength:  Length of feature for MIL attention
        Forward:
            weight sum of the features
        '''
        super(MILNet, self).__init__()
        model = models.resnet18(pretrained = pretrained)
        flatten = nn.Flatten(start_dim = 1)
        fc = nn.Linear(512, featureLength, bias = False)
        self.featureExteract = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
            model.layer4,
            model.avgpool,
            flatten,
            fc,
            model.relu
        )
        self.attentionlayer = MILAttention(featureLength, linearLength)
        
    def forward(self, x):
        b, p, c, h, w = x.shape              # the input data contain multi-images, b = batch(Batch_size), p = patches in bag, c = channel of images, h,w = shape of image
        x = x.view((b*p, c, h, w))           # Bec. input size of feature_ex..._part1 is b, c, h, w, the shape is needed to reshape into b*p, c, h, w,
                                             #       there, the patches pass the convolution independently
        x = self.featureExteract(x)          # After feature extractor (!!! avgpool is done in this method? bec. after resnet, the size is 512*7*7 and features
                                             #  pass 1*1 avgpool, after that, the size would be 512*1*1, then fc 512 to 256)
        x = x.view(b, p, -1)
        weight = self.attentionlayer(x)
        weight_sum = torch.bmm(weight, x)
        weight_sum = weight_sum.squeeze(1)
        
        return weight_sum

class RepNet(nn.Module):
    def __init__(self, featureLength = 256, MILLength = 128, RepLength = 128, projectTytpe = 'linear', pretrained = True, feature_normalized = True):
        '''
        Parameters:
            featureLength: Length of feature from resnet18
            linearLength:  Length of feature for MIL attention
            RepLength:     Length of feature for representation learning
        Forward:
            weight sum of normalized features
        '''
        super(RepNet, self).__init__()

        if type(pretrained) == bool:
            self.encoder = MILNet(featureLength, MILLength, pretrained)
        elif type(pretrained) == str:
            self.encoder = MILNet(featureLength, MILLength, False)
            print(self.encoder.load_state_dict(torch.load(pretrained), strict=False))
        else:
            raise TypeError(f"The type of pretrained is {type(pretrained)}, only bool and str is allowed.")

        if projectTytpe == 'linear':
            self.projector = nn.Sequential(
                nn.Linear(featureLength, RepLength, bias = False)
            )
        elif projectTytpe == 'mlp':
            self.projector = nn.Sequential(
                nn.Linear(featureLength, RepLength, bias = True),
                nn.ReLU(inplace=True),
                nn.Linear(RepLength, RepLength, bias = True)
            )
        else:
            raise ValueError(f"The str of projectTytpe is {projectTytpe}, only linear and mlp is allowed.")
        self.feature_normalized = feature_normalized

    def forward(self, x):
        representations = self.encoder(x)
        projection = self.projector(representations)
        
        if self.feature_normalized == True:
            return nn.functional.normalize(projection, dim = 1)
        else:
            return projection

class ClfNet(nn.Module):
    def __init__(self, featureLength = 256, MILLength = 128, pretrained = True):
        super(ClfNet, self).__init__()

        if type(pretrained) == bool:
            self.featureExtractor = MILNet(featureLength, MILLength, pretrained)
        elif type(pretrained) == str:
            self.featureExtractor = MILNet(featureLength, MILLength, False)
            print(self.featureExtractor.load_state_dict(torch.load(pretrained), strict=False))
        else:
            raise TypeError(f"The type of pretrained is {type(pretrained)}, only bool and str is allowed.")

        self.classifier = nn.Sequential(
            nn.Linear(256, 256, bias = True),
            nn.ReLU(),
            nn.Linear(256, 128, bias = True),
            nn.ReLU(),
            nn.Linear(128, 1, bias = True)
        )

    def forward(self, x):
        features = self.featureExtractor(x)
        preds = self.classifier(features)
        
        return preds

class ADVClfNet(nn.Module):
    def __init__(self, featureLength = 256, MILLength = 128, pretrained = True):
        super(ADVClfNet, self).__init__()

        if type(pretrained) == bool:
            self.featureExtractor = MILNet(featureLength, MILLength, pretrained)
        elif type(pretrained) == str:
            self.featureExtractor = MILNet(featureLength, MILLength, False)
            print(self.featureExtractor.load_state_dict(torch.load(pretrained), strict=False))
        else:
            raise TypeError(f"The type of pretrained is {type(pretrained)}, only bool and str is allowed.")

        self.classifier = nn.Sequential(
            nn.Linear(256, 256, bias = True),
            nn.ReLU(),
            nn.Linear(256, 128, bias = True),
            nn.ReLU(),
            nn.Linear(128, 1, bias = True)
        )

        self.sensitiveclassifier = nn.Sequential(
            nn.Linear(256, 256, bias = True),
            nn.ReLU(),
            nn.Linear(256, 128, bias = True),
            nn.ReLU(),
            nn.Linear(128, 1, bias = True)
        )

    def forward(self, x):
        features = self.featureExtractor(x)
        preds = self.classifier(features)
        sensitive_preds = self.sensitiveclassifier(grad_reverse(features))
        return preds, sensitive_preds

class MultiRepNet(nn.Module):
    def __init__(self, featureLength = 256, MILLength = 128, RepLength = 128, pretrained = True):
        '''
        Parameters:
            featureLength: Length of feature from resnet18
            linearLength:  Length of feature for MIL attention
            RepLength:     Length of feature for representation learning
        Forward:
            weight sum of normalized features
        '''
        super(MultiRepNet, self).__init__()

        if type(pretrained) == bool:
            self.encoder = MILNet(featureLength, MILLength, pretrained)
        elif type(pretrained) == str:
            self.encoder = MILNet(featureLength, MILLength, False)
            print(self.encoder.load_state_dict(torch.load(pretrained), strict=False))
        else:
            raise TypeError(f"The type of pretrained is {type(pretrained)}, only bool and str is allowed.")

        self.projector = nn.Sequential(
            nn.Linear(featureLength, RepLength, bias = False)
        )

        self.secondprojector = nn.Sequential(
            nn.Linear(featureLength, RepLength, bias = False)
        )

    def forward(self, x):
        representations = self.encoder(x)
        projection = self.projector(representations)
        secondprojection = self.secondprojector(representations)

        return nn.functional.normalize(projection, dim = 1), nn.functional.normalize(secondprojection, dim = 1)

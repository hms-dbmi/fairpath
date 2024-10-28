import torch
from torchstain.base.normalizers.he_normalizer import HENormalizer
from torchstain.torch.utils import cov, percentile

"""
Source code ported from: https://github.com/schaugf/HEnorm_python
Original implementation: https://github.com/mitkovetta/staining-normalization
"""
class TorchMacenkoNormalizer(HENormalizer):
    # modified __convert_rgb2od
    #        ODhat  = OD[torch.mean(OD,dim=1) > beta,:]
    def __init__(self):
        super().__init__()

        self.HERef = torch.tensor([[0.5626, 0.2159],
                                   [0.7201, 0.8012],
                                   [0.4062, 0.5581]])
        self.maxCRef = torch.tensor([1.9705, 1.0308])

    def __convert_rgb2od(self, I, Io, beta):
        I = I.permute(1, 2, 0)
        # calculate optical density
        OD = -torch.log((I.reshape((-1, I.shape[-1])).float() + 1)/Io)

        # remove transparent pixels
        # ODhat = OD[~torch.any(OD < beta, dim=1)]
        ODhat  = OD[torch.mean(OD,dim=1) > beta,:]

        ## for debugging
        # OD3d = -torch.log((I.float() + 1)/Io)
        # ODhat_3d = torch.mean(OD3d,dim=2) > beta
        # import matplotlib.pyplot as plt
        # plt.close()
        # plt.figure(figsize=(15,8))
        # plt.subplot(1,2,1)
        # plt.imshow(I/255)

        # plt.subplot(1,2,2)
        # plt.imshow(ODhat_3d)
        # plt.savefig('test.jpg')

        return OD, ODhat

    def __find_HE(self, ODhat, eigvecs, alpha):
        # project on the plane spanned by the eigenvectors corresponding to the two
        # largest eigenvalues
        That = torch.matmul(ODhat, eigvecs)
        phi = torch.atan2(That[:, 1], That[:, 0])

        minPhi = percentile(phi, alpha)
        maxPhi = percentile(phi, 100 - alpha)

        vMin = torch.matmul(eigvecs, torch.stack((torch.cos(minPhi), torch.sin(minPhi)))).unsqueeze(1)
        vMax = torch.matmul(eigvecs, torch.stack((torch.cos(maxPhi), torch.sin(maxPhi)))).unsqueeze(1)

        # a heuristic to make the vector corresponding to hematoxylin first and the
        # one corresponding to eosin second
        HE = torch.where(vMin[0] > vMax[0], torch.cat((vMin, vMax), dim=1), torch.cat((vMax, vMin), dim=1))

        return HE

    def __find_concentration(self, OD, HE):
        # rows correspond to channels (RGB), columns to OD values
        Y = OD.T

        # determine concentrations of the individual stains
        return torch.lstsq(Y, HE)[0][:2]

    def __compute_matrices(self, I, Io, alpha, beta,HE=None):
        OD, ODhat = self.__convert_rgb2od(I, Io=Io, beta=beta)  
        # import matplotlib.pyplot as plt
        # import cv2
        # I3d = I.permute(1,2,0).int()
        # OD_mask  = ~torch.any(OD < 0.001, dim=1)
        # OD_mask  = torch.mean(OD,dim=1) > 0.03

        # OD_mask = OD_mask.reshape(1460, 1460,1).int()
        # OD_mask = OD_mask * 255

        # cv2.imwrite('I.jpg',I3d.cpu().numpy())
        # cv2.imwrite('OD_mask.jpg',OD_mask.cpu().numpy())


        # compute eigenvectors
        if HE is None:
            _, eigvecs = torch.symeig(cov(ODhat.T), eigenvectors=True)
            eigvecs = eigvecs[:, [1, 2]]

            HE = self.__find_HE(ODhat, eigvecs, alpha)

        C = self.__find_concentration(OD, HE)
        maxC = torch.stack([percentile(C[0, :], 99), percentile(C[1, :], 99)])

        return HE, C, maxC

    def fit(self, I, Io=240, alpha=1, beta=0.15):
        if isinstance(I,list):
            I_list = I
            HEs = []
            maxCs = []
            for I in I_list:
                HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)
                HEs.append(HE)
                maxCs.append(maxC)
            HE = torch.stack(HEs,dim=0)
            maxC = torch.stack(maxCs,dim=0)
            HE_mean = torch.mean(HE,dim=0)
            maxC_mean = torch.mean(maxC,dim=0)
            self.HERef = HE_mean
            self.maxCRef = maxC_mean
        
        else:
            HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)
            self.HERef = HE
            self.maxCRef = maxC
    
    def fit_source(self, I, Io=240, alpha=1, beta=0.15):
        HE, _, maxC = self.__compute_matrices(I, Io, alpha, beta)

        return HE, maxC


    def normalize(self, I, Io=240,Io_out=240, alpha=1, beta=0.15,HE=None,stains=True):
        ''' Normalize staining appearence of H&E stained images

        Example use:
            see test.py

        Input:
            I: RGB input image: tensor of shape [C, H, W] and type uint8
            Io: (optional) transmitted light intensity
            Io_out: output transmitted light intensity
            alpha: percentile
            beta: transparency threshold
            stains: if true, return also H & E components
            HE: HE for source image. If None, estimate it with __compute_matrices

        Output:
            Inorm: normalized image
            H: hematoxylin image
            E: eosin image

        Reference:
            A method for normalizing histology slides for quantitative analysis. M.
            Macenko et al., ISBI 2009
        '''
        c, h, w = I.shape
        HE, C, maxC = self.__compute_matrices(I, Io, alpha, beta,HE)

        # normalize stain concentrations
        C *= (self.maxCRef / maxC).unsqueeze(-1)

        # recreate the image using reference mixing matrix
        Inorm = Io_out * torch.exp(-torch.matmul(self.HERef, C))
        Inorm[Inorm > 255] = 255
        Inorm = Inorm.T.reshape(h, w, c).int()

        H, E = None, None

        if stains:
            H = torch.mul(Io_out, torch.exp(torch.matmul(-self.HERef[:, 0].unsqueeze(-1), C[0, :].unsqueeze(0))))
            H[H > 255] = 255
            H = H.T.reshape(h, w, c).int()

            E = torch.mul(Io_out, torch.exp(torch.matmul(-self.HERef[:, 1].unsqueeze(-1), C[1, :].unsqueeze(0))))
            E[E > 255] = 255
            E = E.T.reshape(h, w, c).int()

        return Inorm, H, E

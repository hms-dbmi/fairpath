import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, labels = None, mask = None, sensitive = None, sim_method = 'dot', method = 'FairCon', device = 'cuda'):

        if features.device != device:
            features = features.to(device)

        if len(features.shape) < 3:
            raise ValueError(f'features is required 3 dimensionsat least. But features is shape of {features.shape}.')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size, contrast_count, featuresLens = features.shape

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.reshape(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        if sim_method == 'dot':
            anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
            if method == 'FairCon':
                anchor_dot_contrast = torch.div(anchor_dot_contrast, torch.sqrt(torch.tensor(anchor_feature.shape[-1])))
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

        elif sim_method == 'cosineSimilarity' or sim_method == 'cosineDistance':
           
            anchor_feature_expand = anchor_feature.repeat(anchor_count*contrast_count, 1, 1)
            contrast_feature_expand = torch.transpose(contrast_feature.repeat(anchor_count*contrast_count, 1, 1), 0, 1)
            cos = nn.CosineSimilarity(dim = 2, eps = 1e-10)
            anchor_cossim_contrast = cos(anchor_feature_expand, contrast_feature_expand)
            if sim_method == 'cosineDistance':
                anchor_cossim_contrast = 1 - anchor_cossim_contrast
            anchor_contrast = anchor_cossim_contrast
            logits = anchor_contrast
        else:
            raise ValueError(f'Similarity Method:{sim_method} can\'t be found.')
   
        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.logical_xor(torch.eye(batch_size*anchor_count, batch_size*anchor_count), torch.tensor(1)).float().to(device)
        mask = mask * logits_mask

        if method == 'SupCon':
            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            # compute mean of log-likelihood over positive
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, batch_size).mean()

        elif method == 'FairCon':
            if sensitive == None:
                raise ValueError('FairCon: no sensitive labels input.')
            sensitive = sensitive.reshape(-1, 1)
            mask = (torch.ne(sensitive, sensitive.T)*torch.eq(labels, labels.T)).float().to(device)
            mask = mask.repeat(anchor_count, contrast_count)
            mask = mask * logits_mask
            numerator = (torch.exp(logits)*mask).sum(dim = 1, keepdim = True)
            # In denominator, excluding different sensi and different label
            denominator_mask = (~(torch.ne(sensitive, sensitive.T)*torch.ne(labels, labels.T))).float().to(device)
            denominator_mask = denominator_mask.repeat(anchor_count, contrast_count)
            denominator_mask = denominator_mask*logits_mask
            denominator = (torch.exp(logits)*denominator_mask).sum(dim = 1, keepdim = True)
            log_prob_pos = torch.div(numerator, denominator)
            mean_log_prob_pos = torch.log(log_prob_pos/mask.sum(dim = 1, keepdim = True))
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.view(anchor_count, batch_size).mean()

        else:
            raise ValueError(f'Method:{method} can\'t be found.')



        return loss


class WeightedBCELoss(nn.Module):
    def __init__(self, pos_weight):
        super(WeightedBCELoss, self).__init__()
        self.pos_weight = pos_weight

    def forward(self, inputs, targets):
        weights = targets * self.pos_weight + (1 - targets)
        bce_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        weighted_bce_loss = weights * bce_loss
        return weighted_bce_loss.mean()

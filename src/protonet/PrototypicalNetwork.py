from typing import Union, Tuple, List, Iterable, Dict
from sentence_transformers import SentenceTransformer
import torch.nn as nn
from torch import Tensor
import torch
from torch.nn import functional as F
import sys
import time

from scl.losses import SupConLoss


class PrototypicalNetwork(nn.Module):
    def __init__(self,
                 model: SentenceTransformer,
                 sentence_embedding_dimension: int,
                 n_support: int,
                 temperature=0.1,
                 loss_method='SimCLR',
                 contrast_mode='all',
                 contrast_weight=0.1,
                 kl_weight=1):
        super(PrototypicalNetwork, self).__init__()

        self.n_support = n_support
        self.sentence_embedding_dimension = sentence_embedding_dimension
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model = model
        self.criterion = SupConLoss(temperature=temperature, contrast_mode=contrast_mode)
        self.loss_method = loss_method
        self.kl_loss = nn.KLDivLoss(size_average=True, reduce=True)
        self.contrast_weight = contrast_weight
        self.kl_weight = kl_weight

    def euclidean_dist(self, x, y):
        """
        Compute euclidean distance between two tensors:
        :param
        x: query_samples
        y: prototypes
        """
        # x: K(query num)*N x N_VIEW x D
        # y: N x N_VIEW x D
        n = x.size(0)   # K(query num)*N
        m = y.size(0)   # N
        n_view = x.size(1)
        d = x.size(2)
        if n_view != y.size(1) and d != y.size(2):
            raise Exception

        x = x.unsqueeze(1).expand(n, m, n_view, d)
        y = y.unsqueeze(0).expand(n, m, n_view, d)

        return torch.pow(x - y, 2).sum(3)

    def get_prototypes(self, reps_aug, labels_aug, classes):
        return torch.stack([reps_aug[labels_aug == c].mean(0) for c in classes])

    def forward(self, sentence_features: Iterable[Dict[str, Tensor]], labels: Tensor, mode='train'):
        """
        Calculating loss of model.

        :param sentence_features:  [view, K*N*2(query/support), max_seq_length]
        :param labels: [K*N, 1]
        :param mode:
        :return: loss / reps, output
        """
        reps_list = []
        n_view = 1

        # if loss-method=None and aug_method!=none use augmented tasks.
        if self.loss_method.find('SimCLR') > -1 \
                or self.loss_method.find('KL') > -1 \
                or (self.loss_method == 'None' and len(sentence_features) != 1):
            n_view = len(sentence_features)

        for i in range(n_view):  # 4 types sentences (3 augmented)
            model_output = self.model(sentence_features[i])
            reps = model_output['sentence_embedding']
            # reps = self.mlp(reps)
            reps_list.append(reps.unsqueeze(1))

        reps = torch.cat(reps_list, dim=1)  # [K*N*2, view, dim]

        def supp_idxs(c):
            return labels.eq(c).nonzero()[:self.n_support].squeeze(1)

        classes = torch.unique(labels)  # N-way: classes in labels
        n_classes = len(classes)  # N

        # assuming n_query, n_target constants
        n_query = labels.eq(classes[0].item()).sum().item() - self.n_support  # num of query samples

        # support_set
        support_idxs = list(map(supp_idxs, classes))  # get support samples' idxs in batch(N*K*2)

        support_idxs = torch.stack(support_idxs).view(-1)
        reps_aug = reps[support_idxs]  # [K*N, view, dim]
        labels_aug = labels[support_idxs]  # [K*N, 1]

        prototypes = self.get_prototypes(reps_aug, labels_aug, classes)  # [N, view, dim]

        # query_set
        query_idxs = torch.stack(list(map(lambda c: labels.eq(c).nonzero()[self.n_support:], classes))).view(-1)
        query_samples = reps[query_idxs]  # [Kq*N, view, dim]
        query_labels = labels[query_idxs]  # [K*N, 1]

        # compute query samples' distance to protos
        dists = self.euclidean_dist(query_samples, prototypes)  # [K*N, N, view]

        log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, dists.size(1), dists.size(2))
        p_y = F.softmax(-dists, dim=1).view(n_classes, n_query, dists.size(1), dists.size(2))  # [N, Kq, N, view]

        target_inds1 = torch.arange(0, n_classes)
        target_inds2 = target_inds1.view(n_classes, 1, 1, 1)
        target_inds = target_inds2.expand(n_classes, n_query, 1, n_view).long().to(self.device)

        # 1. label pred loss
        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        print(f"Proto loss={loss_val}\t", end='')

        # 2. KL Loss
        kl_loss = 0.0
        if 'KL' in self.loss_method:
            for i in range(n_view):
                for j in range(i + 1, n_view):
                    kl_loss += self.kl_loss(log_p_y[:, :, :, i], p_y[:, :, :, j])
            print(f"KL loss={kl_loss}\t", end='')

        # 3. Contrastive Loss
        contrastive_loss = 0.0
        if self.loss_method in ['SimCLR', 'SimCLR+KL']:
            contrastive_loss = self.criterion(reps_aug)
            print(f"SimCLR loss={contrastive_loss}", end='')

        # Total Loss
        loss = loss_val + self.kl_weight * kl_loss + self.contrast_weight * contrastive_loss

        if mode == 'train':
            print(f" ---> Total loss={loss}")
            print(f" ---> TrainingTime={int(round(time.time()*1000))}")
            return loss
        else:
            print('\n')
            _, y_hat = log_p_y.max(2)
            output = y_hat.squeeze().eq(target_inds.squeeze()).view(-1, n_view)[:, 0]
            return reps, output

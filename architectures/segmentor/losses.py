from turtle import forward
from typing import Literal, Optional, Sequence

from einops import rearrange, reduce
from kornia.geometry.transform import resize
from loguru import logger
import torch
from torch import nn
from torch.functional import Tensor


class WeightedPartialCE(nn.Module):

    def __init__(self, num_classes, eps=1e-12, manual: bool = False):
        """Weighted Partial Cross Entropy Loss formulated as in Multiscale-adversarial.

        params:
        num_classes: int        Number of classes
        eps: float              Epsilon correction term.
        manual: bool            Replace WCE from nn.CrossEntropy to manual manipulation.
        """
        super().__init__()
        self.num_classes = num_classes
        self.eps = eps
        self.manual = manual
 
    def forward(self, y_hat: Tensor, ys: Tensor, ignore_bg: bool = False, reduction: Literal['mean', 'sum'] = 'mean', **kwargs) -> Tensor:
        assert y_hat.shape[1] == ys.shape[1], 'Number of class is mismatch.'
        # Masking
        if ignore_bg:
            ys[:, 0] = 0
        if not kwargs.get('full', False):
            y_hat = y_hat * ys

        # Numbers of scribble pixel of each classes
        ni = reduce(ys, 'b c h w -> c', 'sum')
        # Numbers of all scribble pixel
        n_tot = reduce(ni, 'c -> ', 'sum')
        weights = rearrange([n_tot / (ni[c] + 1e-12) for c in range(self.num_classes)], 'c -> c')
        # Processing ys
        if not self.manual:
            ys = ys[:, 1:, :, :]

        y_hat = rearrange(y_hat, 'b c h w -> (b h w) c')
        if not self.manual:
            ys = rearrange(ys, 'b c h w -> (b h w c)').long()
        else:
            ys = rearrange(ys, 'b c h w -> (b h w) c')
        if self.num_classes == 1:
            wce = nn.BCEWithLogitsLoss()(y_hat, ys)
        else:
            if self.manual:
                wce = [ weights[i] * ys[:, i] * torch.log(y_hat[:, i] + 1e-12) for i in range(self.num_classes)]
                wce = rearrange(wce, 'c x -> x c')
                wce = - reduce(wce, 'x c -> x', 'sum')
                wce = reduce(wce, 'x -> ', reduction)
            else:
                try:
                    wce = nn.CrossEntropyLoss()(y_hat, ys)
                except Exception as e:
                    raise e from ValueError(f'{y_hat.dtype} {ys.dtype}')
        return wce


class DiceLoss(nn.Module):

    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor):
        intersect = reduce(input * target, 'b c h w -> b', 'sum')
        cardinal = reduce(input + target, 'b c h w -> b', 'sum')

        return (-(2.0 * intersect / (cardinal + self.eps)) + 1.0).mean()


class ImageMseLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, y_h: Tensor, y: Tensor):
        try:
            return self.loss(y_h.flatten(), y.flatten().float())
        except Exception as e:
            logger.error(f'Error why got shape y_h: {y_h.shape}, y: {y.shape}')
            raise e


class InterlayerDivergence(nn.Module):

    def __init__(
        self,
        mode: Literal['mean', 'sum'] = 'mean',
        eps: float = 1e-12, upscaling_mode: Literal['nn', 'deconv'] = 'nn',
        stop_gradient: bool = False,
        divergence: Literal['KLD', 'JSD'] = 'KLD'):
        """Interlayer Divergence
        mode: Literal           Operation reduce mode.
        eps: float              Epsilon constant.
        stop_gradient: bool     Stop basis gradient.
        divergence: Literal     Divergence calculation method. 'KLD' for basis approximator divergence. 'JSD' for mutual information divergence.
        """
        super().__init__()
        assert mode in ['mean', 'sum'], f'mode {mode} is not exists/implemented.'
        self.mode = mode
        self.eps = eps
        self.stop_gradient = stop_gradient
        self.divergence = divergence

    def forward(self, attentions: Sequence[Tensor], weights: Optional[list] = None) -> Tensor:
        """Calculate divergence of attentions based on highest-level attention.
        """
        basis = attentions[0].detach() if self.stop_gradient else attentions[0]
        log_basis = torch.log(rearrange(basis, 'b c h w -> (b h w) c') + 1e-12)
        height, width = basis.shape[2], basis.shape[3]
        posterior = []
        if weights is None:
            weights = [ 1 for i in range(len(attentions[1:])) ]
        else:
            if len(weights) != len(attentions[1:]):
                # raise ValueError(f'Expect size of weights to be {len(attentions[1:])} got {len(weights)}')  # Deprecated
                weights = weights[:len(attentions)]  # Truncate
        for att, weight in zip(attentions[1:], weights):
            if weight == 0: continue
            posterior.append(resize(att, size=(height, width), interpolation='nearest') * weight)

        if self.divergence == 'KLD':
            # Kullback-Leibler Divergence.
            posterior = rearrange(posterior, 'a b c h w -> a (b h w) c')
            if self.mode == 'mean':
                # p(x) * (log(p(x)) - log(md))
                # m_log_prob = reduce(torch.log(posterior + 1e-12), 'a x c -> x c', 'mean')
                # Weighted Average
                m_log_prob = reduce(torch.log(posterior + 1e-12), 'a x c -> x c', 'sum') / sum(weights)
                try:
                    divergence = rearrange(basis, 'b c h w -> (b h w) c') * (log_basis - m_log_prob)
                    divergence = reduce(divergence, 'x c -> x', 'sum')
                    divergence = reduce(divergence, 'x -> ', 'mean')
                    if torch.any(torch.isnan(divergence)):
                        logger.error(f'Divergence: {divergence}')
                        raise Exception('Divergence is NaN')
                except Exception as e:
                    logger.error(f'Basis: {basis.shape}, Log Basis: {log_basis.shape}, Log Prob: {m_log_prob.shape}, Posterior: {posterior.shape}')
                    logger.error(f'Basis: {torch.any(torch.isnan(basis))}, Log Basis: {torch.any(torch.isnan(log_basis))}, Log Prob: {torch.any(torch.isnan(m_log_prob))}, Posterior: {torch.any(torch.isnan(posterior))}')
                    raise e
                return divergence

            elif self.mode == 'sum':
                raise NotImplementedError('Not implemented yet.')

            else:
                raise Exception('Illegal mode overiden.')
        elif self.divergence == 'JSD':
            # Jessen-Shannon Divergence
            mean_q = rearrange(posterior, 'a b c h w -> a b c h w')
            mean_q = reduce(mean_q, 'a b c h w -> b c h w', 'mean')
            mixture = 0.5 * (basis + mean_q)
            log_mixture = torch.log(rearrange(mixture, 'b c h w -> (b h w) c') + self.eps)
            log_mean_q = reduce(torch.log(mean_q + 1e-12), 'b c h w -> (b h w) c', 'mean')
            # KLD P || M
            kld_p = 0.5 * rearrange(basis, 'b c h w -> (b h w) c') * (log_basis - log_mixture)
            kld_p = reduce(kld_p, 'x c -> x', 'sum')
            kld_p: Tensor = reduce(kld_p, 'x -> ', 'mean')
            # KLD Q || M
            kld_q = 0.5 * rearrange(mean_q, 'b c h w -> (b h w) c') * (log_mean_q - log_mixture)
            kld_q = reduce(kld_q, 'x c -> x', 'sum')
            kld_q: Tensor = reduce(kld_q, 'x -> ', 'mean')
            return kld_p + kld_q
            
        else:
            raise NotImplementedError(f'Invalid divergence type / Not implemented: {self.divergence}')


class CELoss(nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=weight)

    def forward(self, y_pred: Tensor, y_true: Tensor):
        return self.ce(y_pred, torch.argmax(y_true, dim=1))

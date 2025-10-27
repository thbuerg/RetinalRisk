import lifelines
import numba
import numpy as np
import torch
import torchmetrics


class CoxPHLoss(torch.nn.Module):
    def forward(self, logh, durations, events, eps=1e-7):
        """
        Simple approximation of the COX-ph. Log hazard is not computed on risk-sets, but on ranked list instead.
        This approximation is valid for datamodules w/ low percentage of ties.
        Credit to Haavard Kamme/PyCox
        :param logh:
        :param durations:
        :param events:
        :param eps:
        :return:
        """
        # sort:
        idx = durations.sort(descending=True, dim=0)[1]
        events = events[idx].squeeze(-1)
        logh = logh[idx].squeeze(-1)
        # calculate loss:
        gamma = logh.max()
        log_cumsum_h = logh.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
        if events.sum() > 0:
            loss = -logh.sub(log_cumsum_h).mul(events).sum().div(events.sum())
        else:
            loss = -logh.sub(log_cumsum_h).mul(events).sum()
        return loss


@numba.njit(parallel=False, nogil=True)
def cindex(events, event_times, predictions):
    idxs = np.argsort(event_times)

    events = events[idxs]
    event_times = event_times[idxs]
    predictions = predictions[idxs]

    n_concordant = 0
    n_comparable = 0

    for i in numba.prange(len(events)):
        for j in range(i + 1, len(events)):
            if events[i] and events[j]:
                n_comparable += 1
                n_concordant += (event_times[i] > event_times[j]) == (
                    predictions[i] > predictions[j]
                )
            elif events[i]:
                n_comparable += 1
                n_concordant += predictions[i] < predictions[j]

    if n_comparable > 0:
        return n_concordant / n_comparable
    else:
        return np.nan


class CIndex(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=True):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("logits", default=[], dist_reduce_fx="cat")
        self.add_state("events", default=[], dist_reduce_fx="cat")
        self.add_state("times", default=[], dist_reduce_fx="cat")

    def update(self, logits: torch.Tensor, events: torch.Tensor, times: torch.Tensor):
        self.logits.append(logits)
        self.events.append(events)
        self.times.append(times)

    def compute(self):
        # this version is much faster, but doesn't handle ties correctly.
        return torch.Tensor(
            [
                cindex(
                    torch.cat(self.events).cpu().numpy(),
                    torch.cat(self.times).cpu().numpy(),
                    1 - torch.cat(self.logits).cpu().numpy(),
                )
            ]
        )

        if False:
            return torch.Tensor(
                [
                    lifelines.utils.concordance_index(
                        torch.cat(self.times).cpu().numpy(),
                        1 - torch.cat(self.logits).cpu().numpy(),
                        event_observed=torch.cat(self.events).cpu().numpy(),
                    )
                ]
            )

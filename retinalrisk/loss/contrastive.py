import torch


def nt_xent_loss(predictions_to, has_value_to=None, cpc_tau=0.07, cpc_alpha=0.5):
    # Normalized Temperature-scaled Cross Entropy Loss
    # https://arxiv.org/abs/2002.05709v3

    if has_value_to is not None:
        predictions_to = predictions_to[has_value_to]
        labels = torch.where(has_value_to)[0]
        labels_diag = torch.zeros(
            predictions_to.shape[0],
            predictions_to.shape[1],
            device=predictions_to.device,
            dtype=torch.bool,
        )
        labels_diag[torch.arange(predictions_to.shape[0]), labels] = True
    else:
        labels_diag = torch.diag(
            torch.ones(len(predictions_to), device=predictions_to.device)
        ).bool()

    neg = predictions_to[~labels_diag].reshape(len(predictions_to), -1)
    pos = predictions_to[labels_diag]

    neg_and_pos = torch.cat((neg, pos.unsqueeze(1)), dim=1)

    loss_pos = -pos / cpc_tau
    loss_neg = torch.logsumexp(neg_and_pos / cpc_tau, dim=1)

    loss = 2 * (cpc_alpha * loss_pos + (1.0 - cpc_alpha) * loss_neg)

    return loss

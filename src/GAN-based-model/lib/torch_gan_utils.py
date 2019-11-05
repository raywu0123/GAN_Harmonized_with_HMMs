import torch
from torch.autograd import grad


def to_onehot(y: torch.Tensor, class_num: int):
    input_shape = y.shape
    y = y.view(-1)
    n = y.shape[0]
    cat = torch.zeros([n, class_num], device=y.device)
    cat[torch.arange(n), y] = 1
    output_shape = input_shape + (class_num,)
    cat = cat.view(output_shape)
    return cat


def compute_gradient_penalty(score: torch.Tensor, samples: torch.Tensor):
    jac = grad(
        outputs=score,
        inputs=samples,
        grad_outputs=torch.ones_like(score, device=score.device),
        create_graph=True,
        retain_graph=True,
    )[0]
    slopes = torch.sqrt((jac ** 2).sum(-1).sum(-1))
    return torch.mean((slopes - 1) ** 2)

import torch
import torch.nn as nn

import torch.nn.functional as F

logits = torch.tensor([-1,-2,-0.5,-3,-2], dtype=torch.float)
logits.requires_grad=True

tau = torch.tensor([1])

pd1 = torch.distributions.RelaxedOneHotCategorical(
    temperature=tau, logits=logits)

print(pd1.rsample())


pd2 = torch.distributions.Normal(logits, logits)

print(pd2.rsample())
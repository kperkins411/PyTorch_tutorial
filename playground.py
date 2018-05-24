import torch
print(torch.__version__)
x = torch.empty(5, 3)
print(type(x[0][0]))
z = x[0][0]
print(type(z))
a = torch.ones(6)

d=a.view(2,3)
b=a
a[1]=4


# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
import torch

# ==================================================================== #
#                             Initializing Tensor                      #
# ==================================================================== #

# print("Torch version is {0}".format(torch.__version__))

device = "cuda" if torch.cuda.is_available() else "cpu"



my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32, 
                         device=device, requires_grad=True)

print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)

# Other common initialization methods
x = torch.empty(size=(3,3))
x = torch.zeros((3,3))
x = torch.rand(3,3)
x = torch.ones(3,3)
x = torch.eye(3,3)
x = torch.arange(start=0, end=5, step=1)

x = torch.linspace(start=0.1, end=1, steps=10)

x = torch.empty(size=(1,5)).normal_(mean = 0, std=1)
x = torch.empty(size=(1,5)).uniform_(0,1)
print(x)
x = torch.diag(torch.ones(3))
print(x)



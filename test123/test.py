import torch

# ==================================================================== #
#                             Initializing Tensor                      #
# ==================================================================== #

print("Torch version is".format(torch.__version__))

device = "cuda" if torch.cuda.is_available() else "cpu"



my_tensor = torch.tensor([[1,2,3], [4,5,6]], dtype=torch.float32,
                         device=device, requires_grad=True)

print(my_tensor.dtype)
print(my_tensor.device)
print(my_tensor.shape)
print(my_tensor.requires_grad)



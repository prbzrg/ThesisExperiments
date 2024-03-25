import numpy
import torch
import odl
import odl.contrib
import odl.contrib.torch
import dival
import dival.reference_reconstructors
import dival.util
import dival.util.torch_losses

def getmy_ray_trafo():
    return dival.get_standard_dataset('lodopab', impl='skimage').ray_trafo

def getmy_fbp_op():
    return dival.reference_reconstructors.get_reference_reconstructor('fbp', 'lodopab', impl='skimage').fbp_op

def my_fbp(x):
    return my_fbp_op(x)

def my_radon(x):
    return my_ray_trafo_om(x)

def my_first_part(x, y):
    return dival.util.torch_losses.poisson_loss(my_radon(x), y)

def my_fbp_jl(x):
    return my_fbp(torch.tensor(x))

def my_radon_jl(x):
    return my_radon(torch.tensor(x))

def my_first_part_jl(x, y):
    return my_first_part(torch.tensor(x), torch.tensor(y))

def my_first_part_grad_jl(x, y):
    x2 = torch.tensor(x, requires_grad = True)
    my_first_part(x2, torch.tensor(y)).backward()
    return x2.grad

my_ray_trafo = getmy_ray_trafo()
my_ray_trafo_om = odl.contrib.torch.OperatorModule(my_ray_trafo)
my_fbp_op = getmy_fbp_op()

import numpy
import torch
import dival
import odl
import dival.util.torch_losses
import odl.contrib.torch


def get_ray_1()
    ddd = odl.uniform_discr([-0.13, -0.13], [0.13, 0.13], (362, 362), dtype=numpy.float32)
    return odl.tomo.RayTransform(ddd, odl.tomo.parallel_beam_geometry(ddd, num_angles=1000, det_shape=(513,)), impl="skimage")

def get_ray_2()
    return dival.get_standard_dataset('lodopab', impl='skimage').ray_trafo

def fbp_t(x):
    return odl.contrib.torch.OperatorModule(odl.tomo.analytic.filtered_back_projection.fbp_op(
        get_ray_2(), filter_type = 'Hann', frequency_scaling = 0.641025641025641))(x)

def fbp_t_res(x):
    return fbp_t(torch.tensor(x))

def radon_t(x):
    return odl.contrib.torch.OperatorModule(get_ray_2())(x)

def radon_t_res(x):
    return radon_t(torch.tensor(x))

def first_part(x, y_t):
    return dival.util.torch_losses.poisson_loss(radon_t(x), y_t)

def first_part_res(x, y_t):
    return first_part(torch.tensor(x), torch.tensor(y_t))

def grad_first_part(x, y_t):
    x2 = torch.tensor(x, requires_grad = True)
    first_part(x2, torch.tensor(y_t)).backward()
    return x2.grad

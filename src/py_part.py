import numpy
import torch
import dival
import odl
import dival.util.torch_losses
import odl.contrib.torch


def fbp_t(x):
    ddd = odl.uniform_discr([-0.13, -0.13], [0.13, 0.13], (362, 362))
    rt3 = odl.tomo.RayTransform(ddd, odl.tomo.parallel_beam_geometry(ddd, num_angles=1000, det_shape=(513,)))
    fbp = odl.tomo.analytic.filtered_back_projection.fbp_op(rt3,
	                    filter_type = 'Hann', frequency_scaling = 0.641025641025641)
    fbp3 = odl.contrib.torch.OperatorModule(fbp)
    return fbp3(x)

def fbp_t_res(x):
    x2 = torch.tensor(x)
    res2 = fbp_t(x2)
    return res2

def radon_t(x):
    ddd = odl.uniform_discr([-0.13, -0.13], [0.13, 0.13], (362, 362))
    rt3 = odl.tomo.RayTransform(ddd, odl.tomo.parallel_beam_geometry(ddd, num_angles=1000, det_shape=(513,)))
    rt33 = odl.contrib.torch.OperatorModule(rt3)
    return rt33(x)

def radon_t_res(x):
    x2 = torch.tensor(x)
    res2 = radon_t(x2)
    return res2

def first_part(x, y_t):
    return dival.util.torch_losses.poisson_loss(radon_t(x), y_t)

def first_part_res(x, y_t):
    x2 = torch.tensor(x)
    y_t2 = torch.tensor(y_t)
    res2 = first_part(x2, y_t2)
    return res2

def grad_first_part(x, y_t):
    x2 = torch.tensor(x, requires_grad = True)
    y_t2 = torch.tensor(y_t)
    res2 = first_part(x2, y_t2)
    res2.backward()
    dx2 = x2.grad
    return dx2

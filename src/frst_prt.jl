@pyinclude(srcdir("py_part.py"))

function fbp_t(x)
    convert.(Float32, py"fbp_t_res"(x).tolist())
end

function radon_t(x)
    convert.(Float32, py"radon_t_res"(x).tolist())
end

function first_part(x, y)
    convert.(Float32, py"first_part_res"(x, y).tolist())
end

function grad_first_part(x, y)
    convert.(Float32, py"grad_first_part"(x, y).tolist())
end

# r2 = rand(Float32, 1, 1, 362, 362)
# rr3 = rand(Float32, 1, 1, 1000, 513)
# res1 = first_part(r2, rr3)
# res2 = grad_first_part(r2, rr3)
# res3 = radon_t(r2)
# res4 = fbp_t(rr3)

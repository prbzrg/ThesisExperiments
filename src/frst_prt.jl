@pyinclude(srcdir("py_part.py"))

@inline @fastmath function fbp_t(x)
    convert.(Float32, py"fbp_t_res"(x).tolist())
end

@inline @fastmath function radon_t(x)
    convert.(Float32, py"radon_t_res"(x).tolist())
end

@inline @fastmath function first_part(x, y)
    convert.(Float32, py"first_part_res"(x, y).tolist())
end

@inline @fastmath function grad_first_part(x, y)
    convert.(Float32, py"grad_first_part"(x, y).tolist())
end

# r2 = rand(Float32, 1, 1, 362, 362)
# r3 = rand(Float32, 1, 1, 1000, 513)
# res1 = first_part(r2, r3)
# res2 = grad_first_part(r2, r3)
# res3 = radon_t(r2)
# res4 = fbp_t(r3)

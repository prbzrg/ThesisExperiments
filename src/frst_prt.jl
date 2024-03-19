@pyinclude(srcdir("py_part.py"))

@inline function fbp_t(x)
    convert.(Float32, py"my_fbp_jl"(x).tolist())
end

@inline function radon_t(x)
    convert.(Float32, py"my_radon_jl"(x).tolist())
end

@inline function first_part(x, y)
    convert.(Float32, py"my_first_part_jl"(x, y).tolist())
end

@inline function grad_first_part(x, y)
    convert.(Float32, py"my_first_part_grad_jl"(x, y).tolist())
end

# r2 = rand(Float32, 1, 1, 362, 362)
# r3 = rand(Float32, 1, 1, 1000, 513)
# res1 = first_part(r2, r3)
# res2 = grad_first_part(r2, r3)
# res3 = radon_t(r2)
# res4 = fbp_t(r3)

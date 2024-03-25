@pyinclude(srcdir("py_part.py"))

@inline function my_fbp_py(x)
    convert.(Float32, py"my_fbp_jl"(x).asarray())
end

@inline function my_radon_py(x)
    convert.(Float32, py"my_radon_jl"(x).tolist())
end

@inline function my_first_part_py(x, y)
    convert.(Float32, py"my_first_part_jl"(x, y).tolist())
end

@inline function my_first_part_grad_py(x, y)
    convert.(Float32, py"my_first_part_grad_jl"(x, y).tolist())
end

# r2 = rand(Float32, 1, 1, 362, 362)
# r3 = rand(Float32, 1, 1, 1000, 513)
# res1 = my_first_part_py(r2, r3)
# res2 = my_first_part_grad_py(r2, r3)
# res3 = my_radon_py(r2)
# res4 = my_fbp_py(r3[1, 1, :, :])

@inline function main_fbp(x)
    rotr90(my_fbp_py(rotl90(x)))
end

@inline function main_first_part(x, y)
    my_first_part_py(
        reshape(rotl90(x), (1, 1, 362, 362)),
        reshape(rotl90(y), (1, 1, 1000, 513)),
    )
end

@inline function main_first_part_grad(x, y)
    rotr90(
        my_first_part_grad_py(
            reshape(rotl90(x), (1, 1, 362, 362)),
            reshape(rotl90(y), (1, 1, 1000, 513)),
        )[
            1,
            1,
            :,
            :,
        ],
    )
end

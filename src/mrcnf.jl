const tp_t = Dict(Float16 => Float32, Float32 => Float64, Float64 => BigFloat)

struct MTrans
    tp::Any
    c::Any
    a::Any
    @inline @fastmath function MTrans(
        tp = Float32,
        c = convert(tp_t[tp], 2)^(convert(tp_t[tp], 2) / convert(tp_t[tp], 3)),
        a = convert(tp_t[tp], 4),
    )
        new(tp, c, a)
    end
end

# return [y₁, y₂, y₃, x̄]
@inline @fastmath function forward_trans(mt::MTrans, x)
    c = mt.c
    a = mt.a
    forward_t_m = [
        c^-1 c^-1 -c^-1 -c^-1
        c^-1 -c^-1 c^-1 -c^-1
        c^-1 -c^-1 -c^-1 c^-1
        a^-1 a^-1 a^-1 a^-1
    ]
    forward_t_m * x
end

# return [x₁, x₂, x₃, x₄]
@inline @fastmath function backward_trans(mt::MTrans, y)
    c = mt.c
    a = mt.a
    backward_t_m = (1 / a) * [
        c c c a
        c -c -c a
        -c c -c a
        -c -c c a
    ]
    convert.(mt.tp, backward_t_m * y)
end

mutable struct MRCNF
    nf_arr::Any
end

@inline @fastmath function MRCNF(s)
    nf_arr = []
    push!(nf_arr, NF(1))
    for i in 0:(log2(s) - 1)
        push!(nf_arr, NF(4^i * 3))
    end
end

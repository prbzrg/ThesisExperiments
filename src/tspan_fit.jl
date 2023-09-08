struct FluxTspanLayer{L, RE, I} <: LuxCore.AbstractExplicitLayer
    layer::L
    re::RE
    init_parameters::I
end

@inline @fastmath function FluxTspanLayer(l)
    p, re = Optimisers.destructure(l)
    p_ = copy(p)
    return FluxTspanLayer(l, re, () -> p_)
end

@inline @fastmath function Lux.initialparameters(::AbstractRNG, l::FluxTspanLayer)
    vcat(one(Float32), l.init_parameters())
end

@inline @fastmath function (l::FluxTspanLayer)(x, ps, st)
    l.re(@view ps[(begin + 1):end])(x), st
end

@inline @fastmath function Base.show(io::IO, l::FluxTspanLayer)
    print(io, "FluxTspanLayer($(l.layer))")
end

@inline @fastmath function myloss_tspan(icnf, mode, xs, ps, st)
    first(ps) + loss(icnf, mode, xs, ps, st; tspan = (zero(Float32), first(ps)))
end

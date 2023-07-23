struct FluxTspanLayer{L, RE, I} <: LuxCore.AbstractExplicitLayer
    layer::L
    re::RE
    init_parameters::I
end

function FluxTspanLayer(l)
    p, re = Optimisers.destructure(l)
    p_ = copy(p)
    return FluxTspanLayer(l, re, () -> p_)
end

function Lux.initialparameters(::AbstractRNG, l::FluxTspanLayer)
    vcat(one(Float32), l.init_parameters())
end

(l::FluxTspanLayer)(x, ps, st) = l.re(@view ps[(begin + 1):end])(x), st

Base.show(io::IO, l::FluxTspanLayer) = print(io, "FluxTspanLayer($(l.layer))")

function myloss_tspan(icnf, mode, xs, ps, st)
    first(ps) + loss(icnf, mode, xs, ps, st; tspan = (zero(Float32), first(ps)))
end

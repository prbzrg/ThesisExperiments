# beta
# beta - lux
nn = Lux.Dense(nvars => nvars, tanh)
nn = Lux.Chain(Lux.Dense(nvars => n_hidden, tanh), Lux.Dense(n_hidden => nvars, tanh))
# beta - flux
nn = FluxCompatLayer(Flux.Dense(nvars => nvars, tanh))
nn = FluxCompatLayer(
    Flux.Chain(Flux.Dense(nvars => n_hidden, tanh), Flux.Dense(n_hidden => nvars, tanh)),
)

# mnist
# mnist - lux
nn = Lux.Dense(28 * 28 => 28 * 28, tanh)
nn = Lux.Chain(
    rs_f,
    Lux.Parallel(
        +,
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 1, pad = Lux.SamePad()),
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 2, pad = Lux.SamePad()),
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 3, pad = Lux.SamePad()),
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 4, pad = Lux.SamePad()),
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 5, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 6, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 7, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 8, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 9, pad = Lux.SamePad()),
    ),
    Lux.Conv((3, 3), 3 => 1, tanh; pad = Lux.SamePad()),
    Lux.FlattenLayer(),
)
nn = Lux.Chain(
    rs_f,
    Lux.Parallel(
        agg_f,
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 1, pad = Lux.SamePad()),
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 2, pad = Lux.SamePad()),
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 3, pad = Lux.SamePad()),
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 4, pad = Lux.SamePad()),
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 5, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 6, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 7, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 8, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 9, pad = Lux.SamePad()),
    ),
    Lux.Conv((3, 3), 3 * 5 => 1, tanh; pad = Lux.SamePad()),
    Lux.FlattenLayer(),
)
# mnist - flux
nn = FluxCompatLayer(Flux.gpu(f32(Flux.Dense(28 * 28 => 28 * 28, tanh))))
nn = FluxCompatLayer(
    Flux.gpu(
        f32(
            Flux.Chain(
                rs_f,
                Flux.Parallel(
                    +,
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 1, pad = Flux.SamePad()),
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 2, pad = Flux.SamePad()),
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 3, pad = Flux.SamePad()),
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 4, pad = Flux.SamePad()),
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 5, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 6, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 7, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 8, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 9, pad = Flux.SamePad()),
                ),
                Flux.Conv((3, 3), 3 => 1, tanh; pad = Flux.SamePad()),
                MLUtils.flatten,
            ),
        ),
    ),
)
nn = FluxCompatLayer(
    Flux.gpu(
        f32(
            Flux.Chain(
                rs_f,
                Flux.Parallel(
                    agg_f,
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 1, pad = Flux.SamePad()),
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 2, pad = Flux.SamePad()),
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 3, pad = Flux.SamePad()),
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 4, pad = Flux.SamePad()),
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 5, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 6, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 7, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 8, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 9, pad = Flux.SamePad()),
                ),
                Flux.Conv((3, 3), 3 * 5 => 1, tanh; pad = Flux.SamePad()),
                MLUtils.flatten,
            ),
        ),
    ),
)

# patchnr
# patchnr - lux
nn = Lux.Chain(Lux.Dense(nvars => nvars * 4, tanh), Lux.Dense(nvars * 4 => nvars, tanh))
nn = Lux.Chain(
    rs_f,
    Lux.Parallel(
        +,
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 1, pad = Lux.SamePad()),
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 2, pad = Lux.SamePad()),
        Lux.Conv((3, 3), 1 => 3, tanh; dilation = 3, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 4, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 5, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 6, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 7, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 8, pad = Lux.SamePad()),
        # Lux.Conv((3, 3), 1 => 3, tanh; dilation = 9, pad = Lux.SamePad()),
    ),
    Lux.Conv((3, 3), 3 => 1, tanh; pad = Lux.SamePad()),
    Lux.FlattenLayer(),
)
# patchnr - flux
nn = FluxCompatLayer(
    Flux.gpu(
        f32(
            Flux.Chain(
                Flux.Dense(nvars => nvars * 4, tanh),
                Flux.Dense(nvars * 4 => nvars, tanh),
            ),
        ),
    ),
)
nn = FluxCompatLayer(
    Flux.gpu(
        f32(
            Flux.Chain(
                rs_f,
                Flux.Parallel(
                    +,
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 1, pad = Flux.SamePad()),
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 2, pad = Flux.SamePad()),
                    Flux.Conv((3, 3), 1 => 3, tanh; dilation = 3, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 4, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 5, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 6, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 7, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 8, pad = Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation = 9, pad = Flux.SamePad()),
                ),
                Flux.Conv((3, 3), 3 => 1, tanh; pad = Flux.SamePad()),
                MLUtils.flatten,
            ),
        ),
    ),
)

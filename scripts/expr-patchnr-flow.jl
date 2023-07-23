using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

const allparams = Dict(
    # test
    "n_iter_rec" => 300,
    # "n_iter_rec" => [4, 8, 16, 128, 256, 300],
    "sel_a" => ["min", "max"],

    # train
    "n_t_imgs" => 6,
    # "p_s" => 8,
    "p_s" => 6,
    # "p_s" => [4, 6, 8],

    # nn
    "n_hidden_rate" => 2,
    # "arch" => "Dense-ML",
    "arch" => "Dense",
    # "back" => "Lux",
    "back" => "Flux",

    # construct
    "tspan_end" => 1,
    # "tspan_end" => [1, 4, 8, 32],

    # ICNFModel
    "n_epochs" => 50,
    # "n_epochs" => 2,
    "batch_size" => 2^12,
    # "batch_size" => 32,
)
const dicts = convert.(Dict{String, Any}, dict_list(allparams))

const gt_train_fn = datadir("lodoct", "ground_truth_train_000.hdf5")
const gt_test_fn = datadir("lodoct", "ground_truth_test_000.hdf5")
const obs_train_fn = datadir("lodoct", "observation_train_000.hdf5")
const obs_test_fn = datadir("lodoct", "observation_test_000.hdf5")

d = first(dicts)

@unpack p_s,
n_epochs,
batch_size,
n_iter_rec,
tspan_end,
arch,
back,
n_t_imgs,
sel_a,
n_hidden_rate = d
d2 = Dict{String, Any}("p_s" => p_s)
d3 = Dict{String, Any}(
    # train
    "n_t_imgs" => n_t_imgs,
    "p_s" => p_s,

    # nn
    "n_hidden_rate" => n_hidden_rate,
    "arch" => arch,
    "back" => back,

    # construct
    "tspan_end" => tspan_end,

    # ICNFModel
    "n_epochs" => n_epochs,
    "batch_size" => batch_size,
)
fulld = copy(d)

tspan = convert.(Float32, (0, tspan_end))
fulld["tspan"] = tspan

data, fn = produce_or_load(x -> error(), d2, datadir("gen-ld-patch"))
ptchs = data["ptchs"]
n_pts = size(ptchs, 4)
fulld["n_pts"] = n_pts

data, fn = produce_or_load(x -> error(), d3, datadir("ld-ct-sims"))
@unpack ps, st = data
if use_gpu_nn_test
    ps = gdev(ps)
    st = gdev(st)
end

nvars = p_s * p_s
n_hidden = n_hidden_rate * nvars
fulld["nvars"] = nvars
fulld["n_hidden"] = n_hidden
rs_f(x) = reshape(x, (p_s, p_s, 1, :))

if back == "Lux"
    if arch == "Dense"
        nn = Lux.Dense(nvars => nvars, tanh)
    elseif arch == "Dense-ML"
        nn = Lux.Chain(
            Lux.Dense(nvars => n_hidden, tanh),
            Lux.Dense(n_hidden => nvars, tanh),
        )
    else
        error("Not Imp")
    end
elseif back == "Flux"
    if use_gpu_nn_test
        if arch == "Dense"
            nn = FluxCompatLayer(Flux.gpu(Flux.f32(Flux.Dense(nvars => nvars, tanh))))
        elseif arch == "Dense-ML"
            nn = FluxCompatLayer(
                Flux.gpu(
                    Flux.f32(
                        Flux.Chain(
                            Flux.Dense(nvars => n_hidden, tanh),
                            Flux.Dense(n_hidden => nvars, tanh),
                        ),
                    ),
                ),
            )
        else
            error("Not Imp")
        end
    else
        if arch == "Dense"
            nn = FluxCompatLayer(Flux.f32(Flux.Dense(nvars => nvars, tanh)))
        elseif arch == "Dense-ML"
            nn = FluxCompatLayer(
                Flux.f32(
                    Flux.Chain(
                        Flux.Dense(nvars => n_hidden, tanh),
                        Flux.Dense(n_hidden => nvars, tanh),
                    ),
                ),
            )
        else
            error("Not Imp")
        end
    end
else
    error("Not Imp")
end
if use_gpu_nn_test
    icnf = construct(
        FFJORD,
        nn,
        nvars;
        tspan,
        compute_mode = ZygoteMatrixMode,
        array_type = CuArray,
        sol_kwargs,
    )
else
    icnf = construct(FFJORD, nn, nvars; tspan, compute_mode = ZygoteMatrixMode, sol_kwargs)
end

smp = ptchs[:, :, 1, 10_000:10_000]
smp_f = MLUtils.flatten(smp)
prob = ContinuousNormalizingFlows.inference_prob(icnf, TrainMode(), smp_f, ps, st)
sl = solve(prob)
display(sl.stats)
plt = plot(
    sl[1:(end - (ContinuousNormalizingFlows.n_augment(icnf, TrainMode()) + 1)), 1, :]',
)
savefig(plt, plotsdir("plot-lines", "plt_new.png"))

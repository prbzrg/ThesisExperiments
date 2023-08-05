using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

const allparams = Dict(
    # test
    "n_iter_rec" => 300,
    # "n_iter_rec" => [4, 8, 16, 128, 256, 300],
    "sel_a" => vcat(["min", "max"], 1:32),

    # train
    # "sel_pol" => nothing,
    "sel_pol" => "equ_d",
    # "sel_pol" => "min_max",
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
    "tspan_end" => 10,
    # "tspan_end" => [1, 4, 8, 32],

    # ICNFModel
    "n_epochs" => 48,
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

@unpack n_iter_rec,
sel_a,
sel_pol,
n_t_imgs,
p_s,
n_hidden_rate,
arch,
back,
tspan_end,
n_epochs,
batch_size = d
d2 = Dict{String, Any}("p_s" => p_s)
d3 = Dict{String, Any}(
    # train
    "sel_pol" => sel_pol,
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

@inline function rs_f(x)
    reshape(x, (p_s, p_s, 1, :))
end

if back == "Lux"
    if arch == "Dense"
        nn = Lux.Dense(nvars * 2 => nvars * 2, tanh)
    elseif arch == "Dense-ML"
        nn = Lux.Chain(
            Lux.Dense(nvars * 2 => n_hidden * 2, tanh),
            Lux.Dense(n_hidden * 2 => nvars * 2, tanh),
        )
    else
        error("Not Imp")
    end
elseif back == "Flux"
    if use_gpu_nn_test
        if arch == "Dense"
            nn = FluxCompatLayer(
                Flux.gpu(Flux.f32(Flux.Dense(nvars * 2 => nvars * 2, tanh))),
            )
        elseif arch == "Dense-ML"
            nn = FluxCompatLayer(
                Flux.gpu(
                    Flux.f32(
                        Flux.Chain(
                            Flux.Dense(nvars * 2 => n_hidden * 2, tanh),
                            Flux.Dense(n_hidden * 2 => nvars * 2, tanh),
                        ),
                    ),
                ),
            )
        else
            error("Not Imp")
        end
    else
        if arch == "Dense"
            nn = FluxCompatLayer(Flux.f32(Flux.Dense(nvars * 2 => nvars * 2, tanh)))
        elseif arch == "Dense-ML"
            nn = FluxCompatLayer(
                Flux.f32(
                    Flux.Chain(
                        Flux.Dense(nvars * 2 => n_hidden * 2, tanh),
                        Flux.Dense(n_hidden * 2 => nvars * 2, tanh),
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
    icnf =
        construct(FFJORD, nn, nvars, nvars; tspan, resource = CUDALibs(), augmented = true)
else
    icnf = construct(FFJORD, nn, nvars, nvars; tspan, augmented = true)
end

# way 4
# smp_f = rand(Float32, 36)

# way 3
# smp_f = ones(Float32, 36)
smp_f = zeros(Float32, 36)

# way 2
# ptchs2 = reshape(ptchs, 36, size(ptchs, 4) * 128);
# high_std = argmin(std.(eachcol(ptchs2)))
# smp_f = ptchs2[:, high_std]

# way 1
# smp = ptchs[:, :, 1, 10_000, 10]
# smp_f = vec(smp)
# smp_f = MLUtils.flatten(smp)

prob = ContinuousNormalizingFlows.inference_prob(icnf, TestMode(), smp_f, ps, st)
sl = solve(prob, icnf.sol_args...; icnf.sol_kwargs...)
display(sl.stats)
f = Figure()
ax = Makie.Axis(f[1, 1]; title = "Flow")
broadcast(
    x -> lines!(ax, sl.t, x),
    eachrow(sl[1:(end - (ContinuousNormalizingFlows.n_augment(icnf, TestMode()) + 1)), :]),
)
# ax2 = Makie.Axis(f[1, 2]; title = "Log")
# lines!(ax2, sl.t, sl[end - (ContinuousNormalizingFlows.n_augment(icnf, TestMode())), :])
save(plotsdir("plot-lines", "flow_new.svg"), f)
save(plotsdir("plot-lines", "flow_new.png"), f)

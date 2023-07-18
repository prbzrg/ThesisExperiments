using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

allparams = Dict(
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
    "tspan_end" => 8,
    # "tspan_end" => [1, 4, 8, 32],

    # ICNFModel
    "n_epochs" => 40,
    # "n_epochs" => 2,
    "batch_size" => 2^12,
    # "batch_size" => 32,
)
dicts = dict_list(allparams)
dicts = convert.(Dict{String, Any}, dicts)

gt_train_fn = datadir("lodoct", "ground_truth_train_000.hdf5")
gt_test_fn = datadir("lodoct", "ground_truth_test_000.hdf5")
obs_train_fn = datadir("lodoct", "observation_train_000.hdf5")
obs_test_fn = datadir("lodoct", "observation_test_000.hdf5")

d = first(dicts)
@unpack p_s, n_epochs, batch_size, tspan_end, arch, back, n_t_imgs, n_hidden_rate = d
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
# sel_pc = argmax(vec(std(reshape(ptchs, (:, 128)); dims = 1)))
# sp = sample(1:128, 6)
sp_std = vec(std(reshape(ptchs, (:, 128)); dims = 1))
n_t_imgs_h = n_t_imgs ÷ 2
sp1 =
    broadcast(x -> x[1], sort(collect(enumerate(sp_std)); rev = true, by = (x -> x[2])))[1:n_t_imgs_h]
sp2 = broadcast(x -> x[1], sort(collect(enumerate(sp_std)); by = (x -> x[2])))[1:n_t_imgs_h]
sp = vcat(sp1, sp2)
# fulld["sp"] = [sel_pc]
fulld["sp"] = sp
# ptchs = ptchs[:, :, :, :, sel_pc]
ptchs = reshape(ptchs[:, :, :, :, sp], (p_s, p_s, 1, :))

x = MLUtils.flatten(ptchs)
df = DataFrame(transpose(x), :auto)

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
    if use_gpu_nn_train
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
myloss(icnf, mode, xs, ps, st) = loss(icnf, mode, xs, ps, st, 1.0f-1, 1.0f-1)
if use_gpu_nn_train
    icnf = construct(
        RNODE,
        nn,
        nvars;
        tspan,
        compute_mode = ZygoteMatrixMode,
        array_type = CuArray,
        sol_kwargs,
    )
    model = ICNFModel(
        icnf,
        myloss;
        optimizers,
        n_epochs,
        batch_size,
        # adtype = AutoForwardDiff(),
        resource = CUDALibs(),
    )
else
    icnf = construct(RNODE, nn, nvars; tspan, compute_mode = ZygoteMatrixMode, sol_kwargs)
    model = ICNFModel(
        icnf,
        myloss;
        optimizers,
        n_epochs,
        batch_size,
        # adtype = AutoForwardDiff(),
    )
end

data, fn = produce_or_load(x -> error(), d3, datadir("ld-ct-sims"))
@unpack ps, st = data
if use_gpu_nn_test
    ps = gpu(ps)
    st = gpu(st)
end

smp = ptchs[:, :, 1, 10_000:10_000]
smp_f = MLUtils.flatten(smp)
prob = ContinuousNormalizingFlows.inference_prob(icnf, TrainMode(), smp_f, ps, st)
sl = solve(prob)
display(sl.stats)
plt = plot(sl[1:(end - 3), 1, :]')
savefig(plt, "plt_new.png")

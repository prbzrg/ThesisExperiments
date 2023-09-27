using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

const allparams = Dict(
    # test
    "n_iter_rec" => 300,
    # "n_iter_rec" => [4, 8, 16, 128, 256, 300],
    # "sel_a" => "min",
    "sel_a" => vcat(["min", "max"], 1:128),

    # train
    # "sel_pol" => nothing,
    # "sel_pol" => "total",
    # "sel_pol" => "random",
    # "sel_pol" => "one_min",
    # "sel_pol" => "one_max",
    # "sel_pol" => "equ_d",
    "sel_pol" => "min_max",
    # "n_t_imgs" => 0,
    "n_t_imgs" => 6,
    "p_s" => 6,
    # "p_s" => [4, 6, 8],
    "naug_rate" => 1 + (1 / 36),
    "rnode_reg" => eps_sq[4],
    "steer_reg" => eps_sq[5],

    # nn
    "n_hidden_rate" => 0,
    # "arch" => "Dense-ML",
    "arch" => "Dense",
    # "back" => "Lux",
    "back" => "Flux",
    "have_bias" => nothing,
    # "have_bias" => true,

    # construct
    "tspan_end" => 12,

    # ICNFModel
    # "n_epochs" => 3,
    "n_epochs" => 100,
    # "batch_size" => 2^5,
    "batch_size" => 2^12,
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
naug_rate,
rnode_reg,
steer_reg,
n_hidden_rate,
arch,
back,
have_bias,
tspan_end,
n_epochs,
batch_size = d

if isnothing(have_bias)
    have_bias = true
end

d2 = Dict{String, Any}("p_s" => p_s)
d3 = Dict{String, Any}(
    # train
    "sel_pol" => sel_pol,
    "n_t_imgs" => n_t_imgs,
    "p_s" => p_s,
    "naug_rate" => naug_rate,
    "rnode_reg" => rnode_reg,
    "steer_reg" => steer_reg,

    # nn
    "n_hidden_rate" => n_hidden_rate,
    "arch" => arch,
    "back" => back,
    "have_bias" => have_bias,

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
@unpack nvars, naug_vl, n_in_out, n_hidden = data
@unpack ps, st = data
if use_gpu_nn_test
    ps = gdev(ps)
    st = gdev(st)
end

@inline function rs_f(x)
    reshape(x, (p_s, p_s, 1, :))
end

if back == "Lux"
    if arch == "Dense"
        nn = Lux.Dense(n_in_out => n_in_out, tanh; use_bias = have_bias)
    elseif arch == "Dense-ML"
        nn = Lux.Chain(
            Lux.Dense(n_in_out => n_hidden, tanh; use_bias = have_bias),
            Lux.Dense(n_hidden => n_in_out, tanh; use_bias = have_bias),
        )
    else
        error("Not Imp")
    end
elseif back == "Flux"
    if use_gpu_nn_test
        if arch == "Dense"
            nn = FluxCompatLayer(
                Flux.gpu(
                    Flux.f32(Flux.Dense(n_in_out => n_in_out, tanh; bias = have_bias)),
                ),
            )
        elseif arch == "Dense-ML"
            nn = FluxCompatLayer(
                Flux.gpu(
                    Flux.f32(
                        Flux.Chain(
                            Flux.Dense(n_in_out => n_hidden, tanh; bias = have_bias),
                            Flux.Dense(n_hidden => n_in_out, tanh; bias = have_bias),
                        ),
                    ),
                ),
            )
        else
            error("Not Imp")
        end
    else
        if arch == "Dense"
            nn = FluxCompatLayer(
                Flux.f32(Flux.Dense(n_in_out => n_in_out, tanh; bias = have_bias)),
            )
        elseif arch == "Dense-ML"
            nn = FluxCompatLayer(
                Flux.f32(
                    Flux.Chain(
                        Flux.Dense(n_in_out => n_hidden, tanh; bias = have_bias),
                        Flux.Dense(n_hidden => n_in_out, tanh; bias = have_bias),
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
        nvars,
        naug_vl;
        tspan,
        compute_mode = ZygoteMatrixMode,
        resource = CUDALibs(),
        # sol_kwargs,
    )
else
    icnf = construct(
        FFJORD,
        nn,
        nvars,
        naug_vl;
        tspan,
        compute_mode = ZygoteMatrixMode,
        # sol_kwargs,
    )
end

# way 4
# smp_f = rand(Float32, 36)

# way 3
# smp_f = ones(Float32, 36)
smp_f = zeros(Float32, 36, 1)

# way 2
# ptchs2 = reshape(ptchs, 36, size(ptchs, 4) * 128);
# high_std = argmin(std.(eachcol(ptchs2)))
# smp_f = ptchs2[:, high_std]

# way 1
# smp = ptchs[:, :, 1, 10_000, 10]
# smp_f = vec(smp)
# smp_f = MLUtils.flatten(smp)

@inline function diff_loss(x)
    loss(icnf, TrainMode(), x, ps, st)
end

l_bench = @benchmark diff_loss(smp_f)
@info "diff_loss"
display(l_bench)
# l_bench_ad11 = @benchmark Zygote.gradient(diff_loss, smp_f)
# @info "Zygote.gradient"
# display(l_bench_ad11)
# l_bench_ad12 = @benchmark Zygote.jacobian(diff_loss, smp_f)
# @info "Zygote.jacobian"
# display(l_bench_ad12)
# l_bench_ad21 = @benchmark Zygote.diaghessian(diff_loss, smp_f)
# @info "Zygote.diaghessian"
# display(l_bench_ad21)
# l_bench_ad22 = @benchmark Zygote.hessian(diff_loss, smp_f)
# @info "Zygote.hessian"
# display(l_bench_ad22)
# l_bench_ad23 = @benchmark Zygote.hessian_reverse(diff_loss, smp_f)
# @info "Zygote.hessian_reverse"
# display(l_bench_ad23)

@info "non-stiff"
slvs = Any[VCABM]
# slvs = Any[VCAB3, VCAB4, VCAB5, VCABM3, VCABM4, VCABM5, VCABM, AN5]
# JVODE_Adams

for slv in slvs
    sl_kw = deepcopy(sol_kwargs)
    sl_kw[:alg] = slv()
    icnf = construct(
        FFJORD,
        nn,
        nvars,
        naug_vl;
        tspan,
        compute_mode = ZygoteMatrixMode,
        sol_kwargs = sl_kw,
    )
    display(icnf.sol_kwargs)
    l_bench = @benchmark loss(icnf, TrainMode(), smp_f, ps, st)
    @info "diff_loss: $slv"
    display(l_bench)
end

@info "stiff"
slvs2 = Any[QNDF]
# slvs2 = Any[QNDF1, QBDF1, ABDF2, QNDF2, QBDF2, QNDF, QBDF, FBDF]
# MEBDF2

for slv2 in slvs2
    sl_kw = deepcopy(sol_kwargs)
    sl_kw[:alg] = slv2()
    sl_kw[:alg_hints] = [:stiff, :memorybound]

    icnf = construct(
        FFJORD,
        nn,
        nvars,
        naug_vl;
        tspan,
        compute_mode = ZygoteMatrixMode,
        sol_kwargs = sl_kw,
    )
    display(icnf.sol_kwargs)
    l_bench = @benchmark loss(icnf, TrainMode(), smp_f, ps, st)
    @info "diff_loss: $slv2"
    display(l_bench)
end

using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

const allparams = Dict(
    # test
    "n_iter_rec" => 300,
    # "sel_a" => "min",
    "sel_a" => vcat(["min", "max"], 1:24),

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
    # "p_s" => [4, 6, 8, 10],
    "naug_rate" => 1,
    # "naug_rate" => 1 + (1 / 36),
    "rnode_reg" => eps_sq[3],
    "steer_reg" => eps_sq[4],
    "ode_reltol" => eps_sq[3],
    "tspan_end" => 13,

    # nn
    "n_hidden_rate" => 0,
    # "arch" => "Dense-ML",
    "arch" => "Dense",
    "back" => "Lux",
    # "back" => "Flux",
    # "have_bias" => nothing,
    "have_bias" => false,
    # "have_bias" => true,

    # ICNFModel
    "n_epochs" => 50,
    # "n_epochs" => 9,
    # "n_epochs" => 50,
    # "batch_size" => 2^10,
    "batch_size" => 2^12,
)
const dicts = convert.(Dict{String, Any}, dict_list(allparams))
d = first(dicts)

const gt_train_fn = datadir("lodoct", "ground_truth_train_000.hdf5")
const gt_test_fn = datadir("lodoct", "ground_truth_test_000.hdf5")
const obs_train_fn = datadir("lodoct", "observation_train_000.hdf5")
const obs_test_fn = datadir("lodoct", "observation_test_000.hdf5")

@unpack n_iter_rec,
sel_a,
sel_pol,
n_t_imgs,
p_s,
naug_rate,
rnode_reg,
steer_reg,
ode_reltol,
tspan_end,
n_hidden_rate,
arch,
back,
have_bias,
n_epochs,
batch_size = d

d3 = Dict{String, Any}(
    # train
    "sel_pol" => sel_pol,
    "n_t_imgs" => n_t_imgs,
    "p_s" => p_s,
    "naug_rate" => naug_rate,
    "rnode_reg" => rnode_reg,
    "steer_reg" => steer_reg,
    "ode_reltol" => ode_reltol,
    "tspan_end" => tspan_end,

    # nn
    "n_hidden_rate" => n_hidden_rate,
    "arch" => arch,
    "back" => back,
    "have_bias" => have_bias,

    # ICNFModel
    "n_epochs" => n_epochs,
    "batch_size" => batch_size,
)

if isnothing(have_bias)
    have_bias = true
end

tspan = convert.(Float32, (0, tspan_end))

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
    # elseif back == "Flux"
    #     if use_gpu_nn_test
    #         if arch == "Dense"
    #             nn = FluxCompatLayer(
    #                 Flux.gpu(
    #                     Flux.f32(Flux.Dense(n_in_out => n_in_out, tanh; bias = have_bias)),
    #                 ),
    #             )
    #         elseif arch == "Dense-ML"
    #             nn = FluxCompatLayer(
    #                 Flux.gpu(
    #                     Flux.f32(
    #                         Flux.Chain(
    #                             Flux.Dense(n_in_out => n_hidden, tanh; bias = have_bias),
    #                             Flux.Dense(n_hidden => n_in_out, tanh; bias = have_bias),
    #                         ),
    #                     ),
    #                 ),
    #             )
    #         else
    #             error("Not Imp")
    #         end
    #     else
    #         if arch == "Dense"
    #             nn = FluxCompatLayer(
    #                 Flux.f32(Flux.Dense(n_in_out => n_in_out, tanh; bias = have_bias)),
    #             )
    #         elseif arch == "Dense-ML"
    #             nn = FluxCompatLayer(
    #                 Flux.f32(
    #                     Flux.Chain(
    #                         Flux.Dense(n_in_out => n_hidden, tanh; bias = have_bias),
    #                         Flux.Dense(n_hidden => n_in_out, tanh; bias = have_bias),
    #                     ),
    #                 ),
    #             )
    #         else
    #             error("Not Imp")
    #         end
    #     end
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
        sol_kwargs = merge(sol_kwargs_base, (reltol = ode_reltol,)),
        # sol_kwargs,
        # inplace = true,
    )
else
    icnf = construct(
        FFJORD,
        nn,
        nvars,
        naug_vl;
        tspan,
        compute_mode = ZygoteMatrixMode,
        sol_kwargs = merge(sol_kwargs_base, (reltol = ode_reltol,)),
        # sol_kwargs,
        # inplace = true,
    )
end

# imgs_1 = load(gt_train_fn)["data"]
# imgs_1 = reshape(imgs_1, (362, 362, 1, n_data_b))
# ptchs_1 = extract_patch(imgs_1, p_s, p_s)
# ptchs_1_n = Any[]
# @showprogress for i in vcat([25, 115], 1:24)
#     push!(
#         ptchs_1_n,
#         loss(icnf, TrainMode(), MLUtils.flatten(selectdim(ptchs_1, 5, i)), ps, st),
#     )
# end
# ptchs_1_n = [
#     loss(icnf, TrainMode(), MLUtils.flatten(selectdim(ptchs_1, 5, i)), ps, st) for
#     i in 1:n_data_b
# ]
# @show mean(ptchs_1_n)
# x_pts_1 = MLUtils.flatten(ptchs_1)
# arr_l_1 = [for i in 1:n_data_b]
# @show loss(icnf, TrainMode(), x_pts_1, ps, st)

imgs_2 = load(gt_test_fn)["data"]
imgs_2 = reshape(imgs_2, (362, 362, 1, n_data_b))
ptchs_2 = extract_patch(imgs_2, p_s, p_s)
ptchs_2_n = Any[]
@showprogress for i in vcat([25, 115], 1:24)
    push!(
        ptchs_2_n,
        loss(icnf, TrainMode(), MLUtils.flatten(selectdim(ptchs_2, 5, i)), ps, st),
    )
end
# ptchs_2_n = [
#     loss(icnf, TrainMode(), MLUtils.flatten(selectdim(ptchs_2, 5, i)), ps, st) for
#     i in 1:n_data_b
# ]
@show mean(ptchs_2_n)
# x_pts_2 = MLUtils.flatten(ptchs_2)

# @show loss(icnf, TrainMode(), x_pts_2, ps, st)

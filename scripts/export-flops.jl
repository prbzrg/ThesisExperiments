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

d2 = Dict{String, Any}("p_s" => p_s)
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

data, fn = produce_or_load(x -> error(), d2, datadir("gen-ld-patch"))
ptchs = data["ptchs"]
n_pts = size(ptchs, 4)

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
    )
end

ptchnr = PatchNR(; icnf_f = let icnf = icnf, md = TrainMode(), ps = ps, st = st
    x -> loss(icnf, md, x, ps, st)
end, n_pts, p_s)
gt_x = load(gt_test_fn)["data"]
if sel_a == "just-train"
    error()
elseif sel_a == "min"
    sel_t_img = argmin(vec(std(MLUtils.flatten(gt_x); dims = 1)))
elseif sel_a == "max"
    sel_t_img = argmax(vec(std(MLUtils.flatten(gt_x); dims = 1)))
else
    sel_t_img = sel_a
end
gt_x = gt_x[:, :, sel_t_img]
obs_y = load(obs_test_fn)["data"]
obs_y = obs_y[:, :, sel_t_img]

s_point = main_fbp(obs_y)
s_point_c = copy(s_point)
u_init = vec(s_point_c)
if use_gpu_nn_test
    u_init = gdev(u_init)
end

opt = only(optimizers)

my_cpu = 467.5 * 1.0e9

@belapsed _loss(u_init, ptchnr, obs_y)
t_direct = @belapsed _loss(u_init, ptchnr, obs_y)
@show t_direct
t_direct_r = round((t_direct * my_cpu * 300) / 1.0e15; digits = 2)
@show t_direct_r

@belapsed _loss_gd_o(u_init, ptchnr, obs_y)
t_direct_g = @belapsed _loss_gd_o(u_init, ptchnr, obs_y)
@show t_direct_g
t_direct_g_r = round((t_direct_g * my_cpu * 300) / 1.0e15; digits = 2)
@show t_direct_g_r

# recn_loss_pt2(ptchnr, u_init, obs_y)
# res1 = @count_ops recn_loss_pt2(ptchnr, u_init, obs_y)
# @show res1
# res2 = @gflops recn_loss_pt2(ptchnr, u_init, obs_y)
# @show res2

# recn_loss_pt2_grad(ptchnr, u_init, obs_y)
# res3 = @count_ops recn_loss_pt2_grad(ptchnr, u_init, obs_y)
# @show res3
# res4 = @gflops recn_loss_pt2_grad(ptchnr, u_init, obs_y)
# @show res4

# @belapsed main_fbp(obs_y)
# t_direct = @belapsed main_fbp(obs_y)
# @show t_direct
# t_direct_r = round((t_direct * my_cpu * 300) / 1.0e15; digits = 2)
# @show t_direct_r

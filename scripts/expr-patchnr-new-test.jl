using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

const sol_kwargs_new = (
    save_everystep = false,
    # alg = BS3(; thread = OrdinaryDiffEq.True()),
    alg = VCABM(),
    reltol = 1.0f-2,
    sensealg = BacksolveAdjoint(;
        autodiff = true,
        autojacvec = ZygoteVJP(),
        checkpointing = true,
    ),
)

const allparams = Dict(
    # test
    :n_iter_rec => 300,
    # :sel_a => :min,
    :sel_a => vcat([:min, :max], 1:24),

    # train
    # :sel_pol => nothing,
    # :sel_pol => :total,
    # :sel_pol => :random,
    # :sel_pol => :one_min,
    # :sel_pol => :one_max,
    # :sel_pol => :equ_d,
    :sel_pol => :min_max,
    # :n_t_imgs => 0,
    :n_t_imgs => 6,
    :p_s => 6,
    # :p_s => [4, 6, 8, 10],
    :naug_rate => 1,
    # :naug_rate => 1 + (1 / 36),
    :rnode_reg => 1.0f-2,
    :steer_reg => 1.0f-1,
    :tspan_end => [1, 13],

    # nn
    :n_hidden_rate => 0,
    # :arch => :Dense_ML,
    :arch => :Dense,
    # :back => :SimpleChains,
    :back => :Lux,
    :have_bias => false,
    # :have_bias => true,

    # ICNFModel
    :n_epochs => 50,
    # :n_epochs => 9,
    # :n_epochs => 50,
    # :batch_size => 2^10,
    :batch_size => 2^12,
)
const dicts = convert.(Dict{Symbol, Any}, dict_list(allparams))

d = first(dicts)
@unpack n_iter_rec,
sel_a,
sel_pol,
n_t_imgs,
p_s,
naug_rate,
rnode_reg,
steer_reg,
tspan_end,
n_hidden_rate,
arch,
back,
have_bias,
n_epochs,
batch_size = d

d2 = Dict{Symbol, Any}(:p_s => p_s)
d3 = Dict{Symbol, Any}(
    # train
    :sel_pol => sel_pol,
    :n_t_imgs => n_t_imgs,
    :p_s => p_s,
    :naug_rate => naug_rate,
    :rnode_reg => rnode_reg,
    :steer_reg => steer_reg,
    :tspan_end => tspan_end,

    # nn
    :n_hidden_rate => n_hidden_rate,
    :arch => arch,
    :back => back,
    :have_bias => have_bias,

    # ICNFModel
    :n_epochs => n_epochs,
    :batch_size => batch_size,
)
fulld = copy(d)

tspan = convert.(Float32, (0, tspan_end))
fulld[:tspan] = tspan

data, fn = produce_or_load(x -> error(), d2, datadir("gen-ld-patch"); filename = hash)
@unpack ptchs = data
n_pts = size(ptchs, 4)
fulld[:n_pts] = n_pts

data, fn = produce_or_load(x -> error(), d3, datadir("ld-ct-sims"); filename = hash)
@unpack nvars, naug_vl, n_in_out, n_hidden = data
@unpack ps, st = data
if use_gpu_nn_test
    ps = gdev(ps)
    st = gdev(st)
end

@inline function rs_f(x)
    reshape(x, (p_s, p_s, 1, :))
end

nn = if back == :Lux
    if arch == :Dense
        Lux.Dense(n_in_out => n_in_out, tanh; use_bias = have_bias)
    elseif arch == :Dense_ML
        Lux.Chain(
            Lux.Dense(n_in_out => n_hidden, tanh; use_bias = have_bias),
            Lux.Dense(n_hidden => n_in_out, tanh; use_bias = have_bias),
        )
    else
        error("Not Imp")
    end
elseif back == :SimpleChains
    if arch == :Dense
        Lux.SimpleChainsLayer(
            SimpleChains.SimpleChain(
                static(n_in_out),
                SimpleChains.TurboDense{have_bias}(tanh, n_in_out),
            ),
        )
    elseif arch == :Dense_ML
        Lux.SimpleChainsLayer(
            SimpleChains.SimpleChain(
                static(n_in_out),
                SimpleChains.TurboDense{have_bias}(tanh, n_hidden),
                SimpleChains.TurboDense{have_bias}(tanh, n_in_out),
            ),
        )
    else
        error("Not Imp")
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
        sol_kwargs = sol_kwargs_new,
    )
else
    icnf = construct(
        FFJORD,
        nn,
        nvars,
        naug_vl;
        tspan,
        compute_mode = ZygoteMatrixMode,
        sol_kwargs = sol_kwargs_new,
    )
end

ptchnr = PatchNR(; icnf_f = let icnf = icnf, md = TrainMode(), ps = ps, st = st
    x -> loss(icnf, md, x, ps, st)
end, n_pts, p_s)
gt_x = load(gt_test_fn)["data"]
if sel_a == :just_train
    return fulld
elseif sel_a == :min
    sel_t_img = argmin(vec(std(MLUtils.flatten(gt_x); dims = 1)))
elseif sel_a == :max
    sel_t_img = argmax(vec(std(MLUtils.flatten(gt_x); dims = 1)))
else
    sel_t_img = sel_a
end
fulld[:sel_t_img] = sel_t_img
gt_x = gt_x[:, :, sel_t_img]
obs_y = load(obs_test_fn)["data"]
obs_y = obs_y[:, :, sel_t_img]

fulld[:gt_x] = gt_x

s_point = main_fbp(obs_y)

fulld[:fbp_img] = s_point
fulld[:fbp_a_psnr] = assess_psnr(s_point, gt_x)
fulld[:fbp_a_ssim] = assess_ssim(s_point, gt_x)
fulld[:fbp_a_msssim] = assess_msssim(s_point, gt_x)

# u_init = vec(cstm_fbp(obs_y))
# u_init = standardize(UnitRangeTransform, u_init)
s_point_c = copy(s_point)
u_init = vec(s_point_c)
if use_gpu_nn_test
    u_init = gdev(u_init)
end

new_ps, tst_one = train_loop(u_init, ptchnr, obs_y, Lion(), n_iter_rec)
new_img = reshape(new_ps, (362, 362))
fulld[:res_img] = new_img
fulld[:a_psnr] = assess_psnr(new_img, gt_x)
fulld[:a_ssim] = assess_ssim(new_img, gt_x)
fulld[:a_msssim] = assess_msssim(new_img, gt_x)
fulld[:time_obj] = tst_one

fulld

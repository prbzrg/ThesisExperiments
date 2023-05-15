using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

allparams = Dict(
    "p_s" => 8,
    # "p_s" => [4, 6, 8],
    "n_epochs" => 2,
    # "batch_size" => 128,
    "n_iter_rec" => 16,
)
dicts = dict_list(allparams)
dicts = convert.(Dict{String, Any}, dicts)

gt_train_fn = datadir("lodoct", "ground_truth_train_000.hdf5")
gt_test_fn = datadir("lodoct", "ground_truth_test_000.hdf5")
obs_train_fn = datadir("lodoct", "observation_train_000.hdf5")
obs_test_fn = datadir("lodoct", "observation_test_000.hdf5")

function makesim_gendata(d::Dict)
    @unpack p_s, = d
    fulld = copy(d)

    fulld["p_w"] = p_s
    fulld["p_h"] = p_s

    imgs = load(gt_train_fn)["data"]
    imgs = reshape(imgs, (362, 362, 1, 128))

    ptchs = extract_patch(imgs, p_s, p_s)
    fulld["ptchs"] = ptchs

    fulld
end

function makesim_genflows(d::Dict)
    @unpack p_s, n_epochs = d
    d2 = Dict{String, Any}("p_s" => p_s)
    fulld = copy(d)

    # tspan = convert.(Float32, (0, tspan_end))
    # fulld["tspan"] = tspan

    data, fn = produce_or_load(makesim_gendata, d2, datadir("gen-ld-patch"))
    ptchs = data["ptchs"]
    sel_pc = argmax(vec(std(reshape(ptchs, (:, 128)); dims = 1)))
    # sp = sample(1:128, 6)
    fulld["sp"] = [sel_pc]
    # fulld["sp"] = sp
    ptchs = ptchs[:, :, :, :, sel_pc]
    # ptchs = reshape(ptchs[:, :, :, :, sp], (p_s, p_s, 1, :))

    x = MLUtils.flatten(ptchs)
    df = DataFrame(transpose(x), :auto)

    nvars = p_s * p_s
    fulld["nvars"] = nvars

    rs_f(x) = reshape(x, (p_s, p_s, 1, :))

    nn = FluxCompatLayer(
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
    )

    icnf = construct(RNODE, nn, nvars; compute_mode = ZygoteMatrixMode, sol_kwargs)

    model = ICNFModel(icnf; optimizers, n_epochs)

    mach = machine(model, df)
    fit!(mach)
    ps, st = fitted_params(mach)
    fulld["ps"] = Lux.cpu(ps)
    fulld["st"] = Lux.cpu(st)

    rpt = report(mach)
    fulld["fit_stats"] = rpt.stats

    fulld
end


d = first(dicts)
@unpack p_s, n_epochs, n_iter_rec = d
d2 = Dict{String, Any}("p_s" => p_s)
d3 = Dict{String, Any}("p_s" => p_s, "n_epochs" => n_epochs)
fulld = copy(d)

# tspan = convert.(Float32, (0, tspan_end))
# fulld["tspan"] = tspan

data, fn = produce_or_load(makesim_gendata, d2, datadir("gen-ld-patch"))
ptchs = data["ptchs"]
n_pts = size(ptchs, 4)
fulld["n_pts"] = n_pts

data, fn = produce_or_load(makesim_genflows, d3, datadir("ld-ct-sims"))
@unpack ps, st = data

nvars = p_s * p_s
fulld["nvars"] = nvars
rs_f(x) = reshape(x, (p_s, p_s, 1, :))

nn = FluxCompatLayer(
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
)
icnf = construct(FFJORD, nn, nvars; compute_mode = ZygoteMatrixMode, sol_kwargs)

icnf_f(x) = loss(icnf, x, ps, st)
ptchnr = PatchNR(; icnf_f, n_pts, p_s)
gt_x = load(gt_test_fn)["data"]
sel_t_img = argmax(vec(std(reshape(gt_x, (:, 128)); dims = 1)))
fulld["sel_t_img"] = sel_t_img
gt_x = gt_x[:, :, sel_t_img]
obs_y = load(obs_test_fn)["data"]
obs_y = obs_y[:, :, sel_t_img]

fulld["gt_x"] = gt_x

s_point = cstm_fbp_2(obs_y)

fulld["fbp_img"] = s_point
fulld["fbp_a_psnr"] = assess_psnr(s_point, gt_x)
fulld["fbp_a_ssim"] = assess_ssim(s_point, gt_x)
fulld["fbp_a_msssim"] = assess_msssim(s_point, gt_x)

# u_init = vec(cstm_fbp(obs_y))
# u_init = standardize(UnitRangeTransform, u_init)
u_init = vec(s_point)
# u_init = rand(Float32, 362*362)

opt = Optimisers.Lion()

# tst_one = @timed new_ps = train_loop(u_init, ptchnr, obs_y, opt, n_iter_rec)
# new_img = reshape(new_ps, (362, 362))
# fulld["res_img"] = new_img
# fulld["a_psnr"] = assess_psnr(new_img, gt_x)
# fulld["a_ssim"] = assess_ssim(new_img, gt_x)
# fulld["a_msssim"] = assess_msssim(new_img, gt_x)
# fulld["time_obj"] = tst_one

fulld

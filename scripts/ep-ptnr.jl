using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

allparams = Dict(
    "p_s" => 6,
    # "p_s" => [4, 6, 8],
    "n_epochs" => 2,
    "batch_size" => 32,
    "n_iter_rec" => 300,
    # "n_iter_rec" => [4, 16, 128, 256, 100],
    "tspan_end" => 8,
    "arch" => "Dense",
    "n_t_imgs" => 6,
    "reg_la" => 2,
    "sel_a" => "max",
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
    @unpack p_s, n_epochs, batch_size, tspan_end, arch, n_t_imgs = d
    d2 = Dict{String, Any}("p_s" => p_s)
    fulld = copy(d)

    tspan = convert.(Float32, (0, tspan_end))
    fulld["tspan"] = tspan

    data, fn = produce_or_load(makesim_gendata, d2, datadir("gen-ld-patch"))
    ptchs = data["ptchs"]
    # sel_pc = argmax(vec(std(reshape(ptchs, (:, 128)); dims = 1)))
    # sp = sample(1:128, 6)
    sp_std = vec(std(reshape(ptchs, (:, 128)); dims = 1))
    sp = broadcast(
        x -> x[1],
        sort(collect(enumerate(sp_std)); rev = true, by = (x -> x[2])),
    )[1:n_t_imgs]
    # fulld["sp"] = [sel_pc]
    fulld["sp"] = sp
    # ptchs = ptchs[:, :, :, :, sel_pc]
    ptchs = reshape(ptchs[:, :, :, :, sp], (p_s, p_s, 1, :))

    x = MLUtils.flatten(ptchs)
    df = DataFrame(transpose(x), :auto)

    nvars = p_s * p_s
    fulld["nvars"] = nvars

    rs_f(x) = reshape(x, (p_s, p_s, 1, :))

    nn = FluxCompatLayer(f32(Flux.Dense(nvars => nvars, tanh)))
    icnf = construct(RNODE, nn, nvars; tspan, compute_mode = ZygoteMatrixMode, sol_kwargs)
    model = ICNFModel(icnf; optimizers, n_epochs, batch_size)

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
@unpack p_s, n_epochs, batch_size, n_iter_rec, tspan_end, arch, n_t_imgs, sel_a = d
d2 = Dict{String, Any}("p_s" => p_s)
d3 = Dict{String, Any}(
    "p_s" => p_s,
    "n_epochs" => n_epochs,
    "batch_size" => batch_size,
    "tspan_end" => tspan_end,
    "arch" => arch,
    "n_t_imgs" => n_t_imgs,
)
fulld = copy(d)

tspan = convert.(Float32, (0, tspan_end))
fulld["tspan"] = tspan

data, fn = produce_or_load(makesim_gendata, d2, datadir("gen-ld-patch"))
ptchs = data["ptchs"]
n_pts = size(ptchs, 4)
fulld["n_pts"] = n_pts

data, fn = produce_or_load(makesim_genflows, d3, datadir("ld-ct-sims"))
@unpack ps, st = data

nvars = p_s * p_s
fulld["nvars"] = nvars
rs_f(x) = reshape(x, (p_s, p_s, 1, :))

nn = FluxCompatLayer(f32(Flux.Dense(nvars => nvars, tanh)))
icnf = construct(FFJORD, nn, nvars; tspan, compute_mode = ZygoteMatrixMode, sol_kwargs)

icnf_f(x) = loss(icnf, x, ps, st)
ptchnr = PatchNR(; icnf_f, n_pts, p_s)
gt_x = load(gt_test_fn)["data"]
if sel_a == "min"
    sel_t_img = argmin(vec(std(reshape(gt_x, (:, 128)); dims = 1)))
elseif sel_a == "max"
    sel_t_img = argmax(vec(std(reshape(gt_x, (:, 128)); dims = 1)))
end
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
s_point_c = copy(s_point)
u_init = vec(s_point_c)
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

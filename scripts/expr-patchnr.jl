using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

allparams = Dict(
    "p_s" => [8],
    # "p_s" => [4, 6, 8],
    "n_epochs" => 2,
    # "batch_size" => 128,
)
dicts = dict_list(allparams)
dicts = convert.(Dict{String, Any}, dicts)

gt_fn = datadir("lodoct", "ground_truth_train_000.hdf5")
obs_fn = datadir("lodoct", "observation_test_000.hdf5")

function makesim_gendata(d::Dict)
    @unpack p_s, n_epochs = d
    fulld = copy(d)

    fulld["p_w"] = p_s
    fulld["p_h"] = p_s

    imgs = load(gt_fn)["data"]
    imgs = reshape(imgs, (362, 362, 1, 128))

    ptchs = extract_patch(imgs, p_s, p_s)
    fulld["ptchs"] = ptchs

    fulld
end

function makesim_genflows(d::Dict)
    @unpack p_s, n_epochs = d
    d2 = copy(d)
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
    nn2 = FluxCompatLayer(
        Flux.gpu(
            f32(
                Flux.Chain(
                    Flux.Dense(nvars => nvars * 4, tanh),
                    Flux.Dense(nvars * 4 => nvars, tanh),
                ),
            ),
        ),
    )

    rs_f(x) = reshape(x, (p_s, p_s, 1, :))

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
                        # Flux.Conv((3, 3), 1 => 3, tanh; dilation=4, pad=Flux.SamePad()),
                        # Flux.Conv((3, 3), 1 => 3, tanh; dilation=5, pad=Flux.SamePad()),
                        # Flux.Conv((3, 3), 1 => 3, tanh; dilation=6, pad=Flux.SamePad()),
                        # Flux.Conv((3, 3), 1 => 3, tanh; dilation=7, pad=Flux.SamePad()),
                        # Flux.Conv((3, 3), 1 => 3, tanh; dilation=8, pad=Flux.SamePad()),
                        # Flux.Conv((3, 3), 1 => 3, tanh; dilation=9, pad=Flux.SamePad()),
                    ),
                    Flux.Conv((3, 3), 3 => 1, tanh; pad = Flux.SamePad()),
                    MLUtils.flatten,
                ),
            ),
        ),
    )

    icnf = construct(
        RNODE,
        nn,
        nvars;
        compute_mode = ZygoteMatrixMode,
        array_type = CuArray,
        sol_kwargs,
    )

    model = ICNFModel(icnf; resource = CUDALibs(), optimizers, n_epochs)

    mach = machine(model, df)
    fit!(mach)
    ps, st = fitted_params(mach)
    fulld["ps"] = Lux.cpu(ps)
    fulld["st"] = Lux.cpu(st)

    rpt = report(mach)
    fulld["fit_stats"] = rpt.stats

    fulld
end

function makesim(d::Dict)
    @unpack p_s, n_epochs = d
    d2 = copy(d)
    d3 = copy(d)
    fulld = copy(d)

    # tspan = convert.(Float32, (0, tspan_end))
    # fulld["tspan"] = tspan

    data, fn = produce_or_load(makesim_gendata, d2, datadir("gen-ld-patch"))
    ptchs = data["ptchs"]
    n_pts = size(ptchs, 4)
    fulld["n_pts"] = n_pts

    data, fn = produce_or_load(makesim_genflows, d3, datadir("ld-ct-sims"))
    @unpack ps, st = data
    # ps = Lux.gpu(ps)
    # st = Lux.gpu(st)

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
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation=4, pad=Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation=5, pad=Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation=6, pad=Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation=7, pad=Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation=8, pad=Flux.SamePad()),
                    # Flux.Conv((3, 3), 1 => 3, tanh; dilation=9, pad=Flux.SamePad()),
                ),
                Flux.Conv((3, 3), 3 => 1, tanh; pad = Flux.SamePad()),
                MLUtils.flatten,
                # vec,
            ),
        ),
    )
    # ) |> f32 |> Flux.gpu |> FluxCompatLayer
    icnf = construct(
        FFJORD,
        nn,
        nvars;
        compute_mode = ZygoteMatrixMode,
        # array_type = CuArray,
        sol_kwargs,
    )

    icnf_f(x) = loss(icnf, x, ps, st)
    ptchnr = PatchNR(; icnf_f, n_pts, p_s)
    obs_y = load(obs_fn)["data"]
    obs_y = obs_y[:, :, 1]
    gt_x = load(gt_fn)["data"]
    gt_x = gt_x[:, :, 1]

    opt = Optimisers.Lion()
    n_iter = 10

    u_init = vec(cstm_fbp(obs_y))
    u_init = standardize(UnitRangeTransform, u_init)
    # u_init = rand(Float32, 362*362)

    tst_one = @timed new_ps = train_loop(u_init, ptchnr, obs_y, opt, n_iter)
    new_img = reshape(new_ps, (362, 362))
    fulld["res_img"] = new_img
    fulld["a_psnr"] = assess_psnr(new_img, gt_x)
    fulld["a_ssim"] = assess_ssim(new_img, gt_x)
    fulld["a_msssim"] = assess_msssim(new_img, gt_x)
    fulld["time_obj"] = tst_one

    fulld
end

for (i, d) in enumerate(dicts)
    CUDA.allowscalar() do
        produce_or_load(makesim, d, datadir("patchnr-sims"))
    end
end

df = collect_results(datadir("patchnr-sims"))

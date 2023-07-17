using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

allparams = Dict(
    # test
    "n_iter_rec" => 40,
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
    @unpack p_s, n_epochs, batch_size, tspan_end, arch, back, n_t_imgs, n_hidden_rate = d
    d2 = Dict{String, Any}("p_s" => p_s)
    fulld = copy(d)

    tspan = convert.(Float32, (0, tspan_end))
    fulld["tspan"] = tspan

    data, fn = produce_or_load(makesim_gendata, d2, datadir("gen-ld-patch"))
    ptchs = data["ptchs"]
    # sel_pc = argmax(vec(std(reshape(ptchs, (:, 128)); dims = 1)))
    # sp = sample(1:128, 6)
    sp_std = vec(std(reshape(ptchs, (:, 128)); dims = 1))
    n_t_imgs_h = n_t_imgs ÷ 2
    sp1 = broadcast(
        x -> x[1],
        sort(collect(enumerate(sp_std)); rev = true, by = (x -> x[2])),
    )[1:n_t_imgs_h]
    sp2 =
        broadcast(x -> x[1], sort(collect(enumerate(sp_std)); by = (x -> x[2])))[1:n_t_imgs_h]
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
    # myloss(icnf, mode, xs, ps, st) = loss(icnf, mode, xs, ps, st, 1.0f-1, 1.0f-1)
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
            # myloss,
            ;
            optimizers,
            n_epochs,
            batch_size,
            # adtype = AutoForwardDiff(),
            resource = CUDALibs(),
        )
    else
        icnf =
            construct(RNODE, nn, nvars; tspan, compute_mode = ZygoteMatrixMode, sol_kwargs)
        model = ICNFModel(
            icnf,
            # myloss,
            ;
            optimizers,
            n_epochs,
            batch_size,
            # adtype = AutoForwardDiff(),
        )
    end

    mach = machine(model, df)
    fit!(mach)
    ps, st = fitted_params(mach)
    fulld["ps"] = Lux.cpu(ps)
    fulld["st"] = Lux.cpu(st)

    rpt = report(mach)
    fulld["fit_stats"] = rpt.stats

    fulld
end

function makesim_expr(d::Dict)
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

    data, fn = produce_or_load(makesim_gendata, d2, datadir("gen-ld-patch"))
    ptchs = data["ptchs"]
    n_pts = size(ptchs, 4)
    fulld["n_pts"] = n_pts

    data, fn = produce_or_load(makesim_genflows, d3, datadir("ld-ct-sims"))
    @unpack ps, st = data
    if use_gpu_nn_test
        ps = Lux.gpu(ps)
        st = Lux.gpu(st)
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
        icnf =
            construct(FFJORD, nn, nvars; tspan, compute_mode = ZygoteMatrixMode, sol_kwargs)
    end

    icnf_f(x) = loss(icnf, TrainMode(), x, ps, st)
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

    tst_one = @timed new_ps = train_loop_optpkg(u_init, ptchnr, obs_y, opt, n_iter_rec)
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
        produce_or_load(makesim_expr, d, datadir("patchnr-sims"))
    end
end

df = collect_results(datadir("patchnr-sims"))

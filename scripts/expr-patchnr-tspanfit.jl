using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))
include(srcdir("tspan_fit.jl"))

const allparams = Dict(
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

    # ICNFModel
    "n_epochs" => 50,
    # "n_epochs" => 2,
    "batch_size" => 2^12,
    # "batch_size" => 32,
)
const dicts = convert.(Dict{String, Any}, dict_list(allparams))

const gt_train_fn = datadir("lodoct", "ground_truth_train_000.hdf5")
const gt_test_fn = datadir("lodoct", "ground_truth_test_000.hdf5")
const obs_train_fn = datadir("lodoct", "observation_train_000.hdf5")
const obs_test_fn = datadir("lodoct", "observation_test_000.hdf5")

@inline function makesim_gendata(d::Dict)
    @unpack p_s, = d
    fulld = copy(d)

    fulld["p_w"] = p_s
    fulld["p_h"] = p_s

    imgs = load(gt_train_fn)["data"]
    imgs = reshape(imgs, (362, 362, 1, n_data_b))

    ptchs = extract_patch(imgs, p_s, p_s)
    fulld["ptchs"] = ptchs

    fulld
end

@inline function makesim_genflows(d::Dict)
    @unpack p_s, n_epochs, batch_size, arch, back, n_t_imgs, n_hidden_rate = d
    d2 = Dict{String, Any}("p_s" => p_s)
    fulld = copy(d)

    data, fn = produce_or_load(makesim_gendata, d2, datadir("gen-ld-patch"))
    ptchs = data["ptchs"]
    # sel_pc = argmax(vec(std(reshape(ptchs, (:, n_data_b)); dims = 1)))
    # sp = sample(1:n_data_b, 6)
    sp_std = vec(std(reshape(ptchs, (:, n_data_b)); dims = 1))
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

    @inline function rs_f(x)
        reshape(x, (p_s, p_s, 1, :))
    end

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
                nn = FluxTspanLayer(Flux.gpu(Flux.f32(Flux.Dense(nvars => nvars, tanh))))
            elseif arch == "Dense-ML"
                nn = FluxTspanLayer(
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
                nn = FluxTspanLayer(Flux.f32(Flux.Dense(nvars => nvars, tanh)))
            elseif arch == "Dense-ML"
                nn = FluxTspanLayer(
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
    if use_gpu_nn_train
        icnf = construct(
            RNODE,
            nn,
            nvars;
            compute_mode = ZygoteMatrixMode,
            resource = CUDALibs(),
            sol_kwargs,
            λ₁ = 1.0f-1,
            λ₂ = 1.0f-1,
        )
        model = ICNFModel(
            icnf,
            myloss_tspan;
            optimizers,
            n_epochs,
            batch_size,
            # adtype = AutoForwardDiff(),
        )
    else
        icnf = construct(
            RNODE,
            nn,
            nvars;
            compute_mode = ZygoteMatrixMode,
            sol_kwargs,
            λ₁ = 1.0f-1,
            λ₂ = 1.0f-1,
        )
        model = ICNFModel(
            icnf,
            myloss_tspan;
            optimizers,
            n_epochs,
            batch_size,
            # adtype = AutoForwardDiff(),
        )
    end

    mach = machine(model, df)
    fit!(mach)
    ps, st = fitted_params(mach)
    fulld["ps"] = cdev(ps)
    fulld["st"] = cdev(st)

    rpt = report(mach)
    fulld["fit_stats"] = rpt.stats

    fulld
end

@inline function makesim_expr(d::Dict)
    @unpack p_s,
    n_epochs,
    batch_size,
    n_iter_rec,
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

        # ICNFModel
        "n_epochs" => n_epochs,
        "batch_size" => batch_size,
    )
    fulld = copy(d)

    data, fn = produce_or_load(makesim_gendata, d2, datadir("gen-ld-patch"))
    ptchs = data["ptchs"]
    n_pts = size(ptchs, 4)
    fulld["n_pts"] = n_pts

    data, fn = produce_or_load(makesim_genflows, d3, datadir("ld-ct-sims-tspanfit"))
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
                nn = FluxTspanLayer(Flux.gpu(Flux.f32(Flux.Dense(nvars => nvars, tanh))))
            elseif arch == "Dense-ML"
                nn = FluxTspanLayer(
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
                nn = FluxTspanLayer(Flux.f32(Flux.Dense(nvars => nvars, tanh)))
            elseif arch == "Dense-ML"
                nn = FluxTspanLayer(
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
            compute_mode = ZygoteMatrixMode,
            resource = CUDALibs(),
            sol_kwargs,
        )
    else
        icnf = construct(FFJORD, nn, nvars; compute_mode = ZygoteMatrixMode, sol_kwargs)
    end

    @inline function icnf_f(x)
        loss(icnf, TrainMode(), x, ps, st)
    end

    ptchnr = PatchNR(; icnf_f, n_pts, p_s)
    gt_x = load(gt_test_fn)["data"]
    if sel_a == "min"
        sel_t_img = argmin(vec(std(reshape(gt_x, (:, n_data_b)); dims = 1)))
    elseif sel_a == "max"
        sel_t_img = argmax(vec(std(reshape(gt_x, (:, n_data_b)); dims = 1)))
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
    if use_gpu_nn_train || use_gpu_nn_test
        CUDA.allowscalar() do
            produce_or_load(makesim_expr, d, datadir("patchnr-sims-tspanfit"))
        end
    else
        produce_or_load(makesim_expr, d, datadir("patchnr-sims-tspanfit"))
    end
end

df = collect_results(datadir("patchnr-sims-tspanfit"))

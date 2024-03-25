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
    "rnode_reg" => eps_sq[4],
    "steer_reg" => eps_sq[5],
    "ode_reltol" => eps_sq[3],
    "tspan_end" => 1,

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
    @unpack sel_pol,
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
    fulld = copy(d)

    if isnothing(have_bias)
        have_bias = true
    end

    tspan = convert.(Float32, (0, tspan_end))
    fulld["tspan"] = tspan

    data, fn = produce_or_load(makesim_gendata, d2, datadir("gen-ld-patch"))
    ptchs = data["ptchs"]
    sp_std = vec(std(MLUtils.flatten(ptchs); dims = 1))
    if sel_pol == "min_max"
        n_t_imgs_h = n_t_imgs ÷ 2
        sp1 = broadcast(
            x -> x[1],
            sort(collect(enumerate(sp_std)); rev = true, by = (x -> x[2])),
        )[1:n_t_imgs_h]
        sp2 =
            broadcast(x -> x[1], sort(collect(enumerate(sp_std)); by = (x -> x[2])))[1:n_t_imgs_h]
        sp = vcat(sp1, sp2)
    elseif sel_pol == "equ_d"
        sp_tp = broadcast(x -> x[1], sort(collect(enumerate(sp_std)); by = (x -> x[2])))
        sp_imd = [1, n_data_b]
        n_cut = n_t_imgs - 1
        stp = n_data_b / n_cut
        for i in 1:(n_cut - 1)
            push!(sp_imd, round(Int, i * stp))
        end
        sp = sp_tp[sp_imd]
    elseif sel_pol == "one_max"
        sp = [argmax(vec(std(MLUtils.flatten(ptchs); dims = 1)))]
    elseif sel_pol == "one_min"
        sp = [argmin(vec(std(MLUtils.flatten(ptchs); dims = 1)))]
    elseif sel_pol == "random"
        sp = sample(1:n_data_b, n_t_imgs)
    elseif sel_pol == "total"
        sp = collect(1:n_data_b)
    else
        error("Not Imp")
    end
    sort!(sp)
    @info sp
    fulld["sp"] = sp
    ptchs = reshape(ptchs[:, :, :, :, sp], (p_s, p_s, 1, :))

    x = MLUtils.flatten(ptchs)
    df = DataFrame(transpose(x), :auto)

    nvars = p_s * p_s
    naug_vl = convert(Int, naug_rate * nvars)
    n_in_out = nvars + naug_vl
    n_hidden = convert(Int, n_hidden_rate * n_in_out)
    fulld["nvars"] = nvars
    fulld["naug_vl"] = naug_vl
    fulld["n_in_out"] = n_in_out
    fulld["n_hidden"] = n_hidden

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
        if use_gpu_nn_train
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
    if use_gpu_nn_train
        icnf = construct(
            RNODE,
            nn,
            nvars,
            naug_vl;
            tspan,
            compute_mode = ZygoteMatrixMode,
            resource = CUDALibs(),
            steer_rate = steer_reg,
            sol_kwargs = merge(sol_kwargs_base, (reltol = ode_reltol,)),
            λ₁ = rnode_reg,
            λ₂ = rnode_reg,
        )
    else
        icnf = construct(
            RNODE,
            nn,
            nvars,
            naug_vl;
            tspan,
            compute_mode = ZygoteMatrixMode,
            steer_rate = steer_reg,
            sol_kwargs = merge(sol_kwargs_base, (reltol = ode_reltol,)),
            λ₁ = rnode_reg,
            λ₂ = rnode_reg,
        )
    end

    model = ICNFModel(icnf; optimizers, n_epochs, batch_size)
    mach = machine(model, df)
    fit!(mach)
    ps, st = fitted_params(mach)
    fulld["ps"] = cdev(ps)
    fulld["st"] = cdev(st)

    rpt = report(mach)
    fulld["fit_stats"] = rpt.stats

    Zygote.refresh()

    fulld
end

@inline function makesim_expr(d::Dict)
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
    fulld = copy(d)

    if isnothing(have_bias)
        have_bias = true
    end

    tspan = convert.(Float32, (0, tspan_end))
    fulld["tspan"] = tspan

    data, fn = produce_or_load(makesim_gendata, d2, datadir("gen-ld-patch"))
    ptchs = data["ptchs"]
    n_pts = size(ptchs, 4)
    fulld["n_pts"] = n_pts

    data, fn = produce_or_load(makesim_genflows, d3, datadir("ld-ct-sims"))
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
        return fulld
    elseif sel_a == "min"
        sel_t_img = argmin(vec(std(MLUtils.flatten(gt_x); dims = 1)))
    elseif sel_a == "max"
        sel_t_img = argmax(vec(std(MLUtils.flatten(gt_x); dims = 1)))
    else
        sel_t_img = sel_a
    end
    fulld["sel_t_img"] = sel_t_img
    gt_x = gt_x[:, :, sel_t_img]
    obs_y = load(obs_test_fn)["data"]
    obs_y = obs_y[:, :, sel_t_img]

    fulld["gt_x"] = gt_x

    s_point = main_fbp(obs_y)

    fulld["fbp_img"] = s_point
    fulld["fbp_a_psnr"] = assess_psnr(s_point, gt_x)
    fulld["fbp_a_ssim"] = assess_ssim(s_point, gt_x)
    fulld["fbp_a_msssim"] = assess_msssim(s_point, gt_x)

    # u_init = vec(cstm_fbp(obs_y))
    # u_init = standardize(UnitRangeTransform, u_init)
    s_point_c = copy(s_point)
    u_init = vec(s_point_c)
    # u_init = rand(Float32, 362*362)

    opt = only(optimizers)

    tst_one = @timed new_ps = train_loop_optpkg(u_init, ptchnr, obs_y, opt, n_iter_rec)
    new_img = reshape(new_ps, (362, 362))
    fulld["res_img"] = new_img
    fulld["a_psnr"] = assess_psnr(new_img, gt_x)
    fulld["a_ssim"] = assess_ssim(new_img, gt_x)
    fulld["a_msssim"] = assess_msssim(new_img, gt_x)
    fulld["time_obj"] = tst_one

    fulld
end

@inline function makesim_findpostp(d::Dict)
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

    d2 = Dict{String, Any}(
        # test
        "n_iter_rec" => n_iter_rec,
        "sel_a" => sel_a,

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
    fulld = copy(d)

    data2, fn2 = produce_or_load(makesim_expr, d2, datadir("patchnr-sims"))
    gx = data2["gt_x"]
    ri = data2["res_img"]
    new_ri = imfilter(ri, Kernel.Laplacian())

    @inline function fopt(x)
        -assess_ssim((only(x) * new_ri) + ri, gx)
    end

    # @inline function fopt(x)
    #     -50 * assess_ssim((only(x) * new_ri) + ri, gx) -
    #     assess_psnr((only(x) * new_ri) + ri, gx)
    # end

    opt = NewtonTrustRegion()
    # opt = only(optimizers)
    optfunc = OptimizationFunction((ps, θ) -> fopt(ps), AutoForwardDiff())
    optprob = OptimizationProblem(optfunc, ones(Float32, 1))
    res = solve(optprob, opt; maxiters = 10000)

    opt2 = Newton()
    optfunc = OptimizationFunction((ps, θ) -> fopt(ps), AutoForwardDiff())
    optprob = OptimizationProblem(optfunc, res.u)
    res2 = solve(optprob, opt2; maxiters = 10000)

    fulld["scl_v"] = only(res2.u)

    fulld
end

@inline function makesim_postp(d::Dict)
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

    d2 = Dict{String, Any}(
        # test
        "n_iter_rec" => n_iter_rec,
        "sel_a" => sel_a,

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
    d3 = Dict{String, Any}(
        # test
        "n_iter_rec" => n_iter_rec,
        "sel_a" => "min",

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
    fulld = copy(d)

    data2, fn2 = produce_or_load(makesim_expr, d2, datadir("patchnr-sims"))
    merge!(fulld, data2)
    gx = data2["gt_x"]
    ri = data2["res_img"]
    new_ri = imfilter(ri, Kernel.Laplacian())

    data3, fn3 = produce_or_load(makesim_findpostp, d3, datadir("findpostp-sims"))
    scl_v = data3["scl_v"]
    fulld["scl_v"] = scl_v

    postp_img = (scl_v * new_ri) + ri
    fulld["postp_img"] = postp_img
    fulld["pp_psnr"] = assess_psnr(postp_img, gx)
    fulld["pp_ssim"] = assess_ssim(postp_img, gx)
    fulld["pp_msssim"] = assess_msssim(postp_img, gx)

    fulld
end

if use_thrds
    @sync for d in dicts
        @spawn if use_gpu_nn_train || use_gpu_nn_test
            CUDA.allowscalar() do
                produce_or_load(makesim_postp, d, datadir("postp-sims"))
            end
        else
            produce_or_load(makesim_postp, d, datadir("postp-sims"))
        end
    end
else
    for d in dicts
        if use_gpu_nn_train || use_gpu_nn_test
            CUDA.allowscalar() do
                produce_or_load(makesim_postp, d, datadir("postp-sims"))
            end
        else
            produce_or_load(makesim_postp, d, datadir("postp-sims"))
        end
    end
end

df = collect_results(datadir("postp-sims"))

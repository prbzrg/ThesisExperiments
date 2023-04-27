using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

allparams = Dict(
	"p_s" => 8,
	# "p_s" => [4, 6, 8],
    "tspan_end" => 1,
    # "tspan_end" => [1, 2, 4, 8],
    "alg" => BS3(; thread = OrdinaryDiffEq.True()),
    "n_epochs" => 1,
	"batch_size" => 128,
)
dicts = dict_list(allparams)
dicts = convert.(Dict{String, Any}, dicts)

gt_fn = datadir("lodoct", "ground_truth_train_000.hdf5")
obs_fn = datadir("lodoct", "observation_test_000.hdf5")

function makesim_gendata(d::Dict)
    @unpack p_s, = d

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
	@unpack p_s, tspan_end, alg, n_epochs, batch_size = d

    tspan = convert.(Float32, (0, tspan_end))
	fulld = copy(d)
	fulld["tspan"] = tspan

    config = Dict(
        "p_s" => p_s,
    )
    data, fn = produce_or_load(makesim_gendata, config, datadir("gen-ld-patch"))
    ptchs = data["ptchs"]
    sel_pc = argmax(vec(std(reshape(ptchs, (:, 128)); dims=1)))
    @show sel_pc
    # sp = sample(1:128, 6)
    fulld["sp"] = [sel_pc]
    # fulld["sp"] = sp
    ptchs = ptchs[:, :, :, :, sel_pc]
    # ptchs = reshape(ptchs[:, :, :, :, sp], (p_s, p_s, 1, :))

    x = MLUtils.flatten(ptchs)
    df = DataFrame(transpose(x), :auto)

    nvars = p_s * p_s
    # nn = Flux.Chain(
    #     Flux.Dense(nvars => nvars*4, tanh),
    #     Flux.Dense(nvars*4 => nvars, tanh),
    # ) |> f32 |> Flux.gpu |> FluxCompatLayer

    rs_f(x) = reshape(x, (p_s, p_s, 1, :))

    nn = Flux.Chain(
		rs_f,
		Flux.Parallel(+,
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=1, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=2, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=3, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=4, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=5, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=6, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=7, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=8, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=9, pad=Flux.SamePad()),
		),
		Flux.Conv((3, 3), 3 => 1, tanh; pad=Flux.SamePad()),
		MLUtils.flatten,
	) |> f32 |> Flux.gpu |> FluxCompatLayer

    icnf = construct(RNODE, nn, nvars;
            tspan, compute_mode = ZygoteMatrixMode,
            array_type = CuArray,
            sol_kwargs = Dict(
                :alg => alg,
                :sensealg => QuadratureAdjoint(; autodiff = true, autojacvec = ZygoteVJP())))

    optimizers = Any[
		Optimisers.AMSGrad(),
	]
    # optimizers = Any[
    #     Optim.NelderMead(),
    #     Optimisers.OptimiserChain(Optimisers.AMSGrad(), Optimisers.WeightDecay()),
    #     # Optim.Newton(),
    # ]

    model = ICNFModel(icnf;
        resource = CUDALibs(),
        optimizers, n_epochs, batch_size)

    mach = machine(model, df)
	@show Dates.now()
	fit!(mach)
	ps, st = fitted_params(mach)
	fulld["ps"] = Lux.cpu(ps)
	fulld["st"] = Lux.cpu(st)

    rpt = report(mach)
    fulld["fit_stats"] = rpt.stats

    fulld
end

function makesim(d::Dict)
	@unpack p_s, tspan_end, alg, n_epochs, batch_size = d
    d2 = copy(d)

    tspan = convert.(Float32, (0, tspan_end))
	fulld = copy(d)
	fulld["tspan"] = tspan

    config = Dict(
        "p_s" => p_s,
    )
    data, fn = produce_or_load(makesim_gendata, config, datadir("gen-ld-patch"))
    ptchs = data["ptchs"]
    n_pts = size(ptchs, 4)
    fulld["n_pts"] = n_pts

    data, fn = produce_or_load(makesim_genflows, d2, datadir("ld-ct-sims"))
    # data = load(datadir("ld-ct-sims", "batch_size=128_n_epochs=1_p_s=8_tspan_end=1.jld2"))
    @unpack ps, st = data
    # ps = Lux.gpu(ps)
    # st = Lux.gpu(st)

    nvars = p_s * p_s
    fulld["nvars"] = nvars
    rs_f(x) = reshape(x, (p_s, p_s, 1, :))

    function sh_v(x)
        @show("sh_v", size(x), typeof(x))
        x
    end
    function sh_v_2(x)
        @show("sh_v_2", size(x), typeof(x))
        x
    end

    nn = Flux.Chain(
        # sh_v,
		rs_f,
		Flux.Parallel(+,
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=1, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=2, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=3, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=4, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=5, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=6, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=7, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=8, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=9, pad=Flux.SamePad()),
		),
		Flux.Conv((3, 3), 3 => 1, tanh; pad=Flux.SamePad()),
		MLUtils.flatten,
        # vec,
        # sh_v_2,
	) |> f32 |> FluxCompatLayer
	# ) |> f32 |> Flux.gpu |> FluxCompatLayer
    icnf = construct(FFJORD, nn, nvars;
            # tspan,
            compute_mode = ZygoteMatrixMode,
            # differentiation_backend = AbstractDifferentiation.ForwardDiffBackend(),
            # array_type = CuArray,
            sol_kwargs = Dict(
                :alg => alg,
                # :sensealg => ForwardDiffSensitivity(),
                :sensealg => QuadratureAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
            ),
        )

    icnf_f(x) = loss(icnf, x, ps, st)
    # icnf_f(x) = -first(inference(icnf, TrainMode(), x, ps, st))
    ptchnr = PatchNR(; icnf_f, n_pts, p_s)
    obs_y = load(obs_fn)["data"]
    obs_y = obs_y[:, :, 1]
    gt_x = load(gt_fn)["data"]
    gt_x = gt_x[:, :, 1]

    opt = Optimisers.AMSGrad()
    # opt = Optimisers.Adam()
    # opt = Optimisers.OptimiserChain(Optimisers.ClipGrad(), Optimisers.Adam())
    # opt = Optim.NelderMead()
    # opt = Optim.ConjugateGradient()
    # opt = Optim.LBFGS()

    # _loss(ps, θ) = recn_loss(ptchnr, ps, obs_y)
    # function _loss(ps, θ)
    #     pt1 = recn_loss_pt1(ptchnr, ps, obs_y)
    #     pt2 = recn_loss_pt2(ptchnr, ps, obs_y)
    #     pt1 + pt2
    # end
    # function _loss_gd(ps_i, ps, θ)
    #     @show ps_i[1]
    #     # pt1 = ReverseDiff.gradient(x -> recn_loss_pt1(ptchnr, x, obs_y), ps)
    #     # pt1 = ForwardDiff.gradient(x -> recn_loss_pt1(ptchnr, x, obs_y), ps)
    #     # pt2 = ForwardDiff.gradient(x -> recn_loss_pt2(ptchnr, x, obs_y), ps)
    #     # pt2 = only(Zygote.gradient(x -> recn_loss_pt2(ptchnr, x, obs_y), ps))
    #     # ps_i .= pt1 + pt2
    #     ps_i .= rand(Float32, 362*362)
    #     ps_i
    # end

    n_iter = 10
    # prgr = Progress(n_iter; dt = eps(), desc = "Min for CT: ", showspeed = true)
    # function _callback(ps, l)
    #     ProgressMeter.next!(
    #         prgr;
    #         showvalues = [
    #             (:loss_value, l),
    #             (:last_update, Dates.now()),
    #         ],
    #     )
    #     false
    # end

    # u_init = vec(cstm_fbp(obs_y))
    # u_init = standardize(UnitRangeTransform, u_init)
    u_init = rand(Float32, 362*362)

    # optfunc = OptimizationFunction(_loss, Optimization.AutoForwardDiff())
    # optfunc = OptimizationFunction(_loss, Optimization.AutoReverseDiff())
    # optfunc = OptimizationFunction(_loss, Optimization.AutoTracker())
    # optfunc = OptimizationFunction(_loss, Optimization.AutoZygote())
    # optfunc = OptimizationFunction(_loss, Optimization.AutoFiniteDiff())
    # optfunc = OptimizationFunction{true}(_loss, grad=_loss_gd)
    # optfunc = OptimizationFunction(_loss)
    # optprob = OptimizationProblem(optfunc, u_init)
    # tst_one = @timed res = solve(optprob, opt; callback = _callback, maxiters=n_iter)
    # ProgressMeter.finish!(prgr)

    tst_one = @timed new_ps = train_loop(u_init, ptchnr, obs_y, opt, n_iter)
    new_img = reshape(new_ps, (362, 362))
    fulld["res_img"] = new_img
    fulld["a_psnr"] = assess_psnr(new_img, gt_x)
    fulld["a_ssim"] = assess_ssim(new_img, gt_x)
    fulld["a_msssim"] = assess_msssim(new_img, gt_x)
    # fulld["res_img"] = res.u
    fulld["time_obj"] = tst_one

    fulld
end

for (i, d) in enumerate(dicts)
    # f = makesim(d)
    CUDA.allowscalar() do
        produce_or_load(makesim, d, datadir("patchnr-sims"))
    end
    # @tagsave(datadir("simulations", savename(d, "jld2")), f)
end

df = collect_results(datadir("patchnr-sims"))

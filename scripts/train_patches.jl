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

function makesim(d::Dict)
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

for (i, d) in enumerate(dicts)
    # f = makesim(d)
    CUDA.allowscalar() do
        produce_or_load(makesim, d, datadir("ld-ct-sims"))
    end
    # @tagsave(datadir("simulations", savename(d, "jld2")), f)
end

df = collect_results(datadir("ld-ct-sims"))

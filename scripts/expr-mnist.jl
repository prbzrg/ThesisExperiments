using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

debuglogger = Logging.ConsoleLogger(Logging.Error)
Logging.global_logger(debuglogger)


rs_f(x) = reshape(x, (28, 28, 1, :))
agg_f(x...) = cat(x...; dims=3)

lux_mdls = [
	Lux.Dense(28*28 => 28*28, tanh),
	Lux.Chain(
		rs_f,
		Lux.Parallel(+,
			Lux.Conv((3, 3), 1 => 3, tanh; dilation=1, pad=Lux.SamePad()),
			Lux.Conv((3, 3), 1 => 3, tanh; dilation=2, pad=Lux.SamePad()),
			Lux.Conv((3, 3), 1 => 3, tanh; dilation=3, pad=Lux.SamePad()),
			Lux.Conv((3, 3), 1 => 3, tanh; dilation=4, pad=Lux.SamePad()),
			Lux.Conv((3, 3), 1 => 3, tanh; dilation=5, pad=Lux.SamePad()),
			# Lux.Conv((3, 3), 1 => 3, tanh; dilation=6, pad=Lux.SamePad()),
			# Lux.Conv((3, 3), 1 => 3, tanh; dilation=7, pad=Lux.SamePad()),
			# Lux.Conv((3, 3), 1 => 3, tanh; dilation=8, pad=Lux.SamePad()),
			# Lux.Conv((3, 3), 1 => 3, tanh; dilation=9, pad=Lux.SamePad()),
		),
		Lux.Conv((3, 3), 3 => 1, tanh; pad=Lux.SamePad()),
		Lux.FlattenLayer(),
	),
	Lux.Chain(
		rs_f,
		Lux.Parallel(agg_f,
			Lux.Conv((3, 3), 1 => 3, tanh; dilation=1, pad=Lux.SamePad()),
			Lux.Conv((3, 3), 1 => 3, tanh; dilation=2, pad=Lux.SamePad()),
			Lux.Conv((3, 3), 1 => 3, tanh; dilation=3, pad=Lux.SamePad()),
			Lux.Conv((3, 3), 1 => 3, tanh; dilation=4, pad=Lux.SamePad()),
			Lux.Conv((3, 3), 1 => 3, tanh; dilation=5, pad=Lux.SamePad()),
			# Lux.Conv((3, 3), 1 => 3, tanh; dilation=6, pad=Lux.SamePad()),
			# Lux.Conv((3, 3), 1 => 3, tanh; dilation=7, pad=Lux.SamePad()),
			# Lux.Conv((3, 3), 1 => 3, tanh; dilation=8, pad=Lux.SamePad()),
			# Lux.Conv((3, 3), 1 => 3, tanh; dilation=9, pad=Lux.SamePad()),
		),
		Lux.Conv((3, 3), 3*5 => 1, tanh; pad=Lux.SamePad()),
		Lux.FlattenLayer(),
	),
]
flux_mdls = [
	Flux.Dense(28*28 => 28*28, tanh) |> FluxCompatLayer,
	Flux.Chain(
		rs_f,
		Flux.Parallel(+,
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=1, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=2, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=3, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=4, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=5, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=6, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=7, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=8, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=9, pad=Flux.SamePad()),
		),
		Flux.Conv((3, 3), 3 => 1, tanh; pad=Flux.SamePad()),
		MLUtils.flatten,
	) |> FluxCompatLayer,
	Flux.Chain(
		rs_f,
		Flux.Parallel(agg_f,
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=1, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=2, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=3, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=4, pad=Flux.SamePad()),
			Flux.Conv((3, 3), 1 => 3, tanh; dilation=5, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=6, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=7, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=8, pad=Flux.SamePad()),
			# Flux.Conv((3, 3), 1 => 3, tanh; dilation=9, pad=Flux.SamePad()),
		),
		Flux.Conv((3, 3), 3*5 => 1, tanh; pad=Flux.SamePad()),
		MLUtils.flatten,
	) |> FluxCompatLayer,
]

allparams = Dict(
    "tspan_end" => 1,
    "alg" => BS3(; thread = OrdinaryDiffEq.True()),
	"autojacvec" => ZygoteVJP(),
	"n_epochs" => 4,
	"batch_size" => 125,
	"n_newdata" => 4,
)
dicts = dict_list(allparams)

# data_train = MNIST(Float32, :train).features
# data_test = MNIST(Float32, :test).features
# data_all = cat(data_train, data_test; dims=3)
# x = MLUtils.flatten(data_all)
x = MLUtils.flatten(cat(MNIST(Float32, :train).features, MNIST(Float32, :test).features; dims=3))
df = DataFrame(transpose(x), :auto)

function makesim(d::Dict)
	@unpack tspan_end, alg, autojacvec, n_epochs, batch_size, n_newdata = d

	tspan = convert.(Float32, (0, tspan_end))
	fulld = copy(d)
	fulld["tspan"] = tspan

	nn = flux_mdls[2]

	# QuadratureAdjoint(; autodiff = true, autojacvec)
	# ForwardDiffSensitivity()
	icnf = construct(RNODE, nn, 28*28;
						tspan,
						compute_mode = ZygoteMatrixMode,
						array_type = CuArray,
						sol_kwargs = Dict(:alg => alg, :sensealg => QuadratureAdjoint(; autodiff = true, autojacvec)))

	# optimizers = Any[
	# 	Optimisers.AMSGrad(),
	# ]
	optimizers = Any[
		Optim.NelderMead(),
		Optimisers.OptimiserChain(Optimisers.AMSGrad(), Optimisers.WeightDecay()),
		# Optim.Newton(),
	]

	model = ICNFModel(icnf;
						resource = CUDALibs(),
						# adtype = Optimization.AutoForwardDiff(),
						have_callback = false,
						optimizers, n_epochs, batch_size)
	mach = machine(model, df)
	@show Dates.now()
	fit!(mach)
	ps, st = fitted_params(mach)
	fulld["ps"] = ps
	fulld["st"] = st

	dist = ICNFDist(icnf, ps, st)
	new_data = rand(dist, n_newdata)
	new_data2 = reshape(new_data, (28, 28, :))
	new_data3 = Gray.(new_data2)
	imshow(new_data3)

	fulld
end


for (i, d) in enumerate(dicts)
    # f = makesim(d)
	CUDA.allowscalar() do
	    produce_or_load(makesim, d, datadir("mnist-sims-res"))
	end
    # @tagsave(datadir("simulations", savename(d, "jld2")), f)
end

df = collect_results(datadir("mnist-sims-res"))

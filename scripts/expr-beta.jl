using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

autojacvec_l = [ZygoteVJP(), EnzymeVJP(), ReverseDiffVJP(true), ReverseDiffVJP(false), true]
optimizers_l = [
    [
        # Optim.NelderMead(),
        Optimisers.OptimiserChain(Optimisers.AMSGrad(), Optimisers.WeightDecay()),
        # Optim.Newton(),
    ],
    [
        Optimisers.Lion(),
    ],
    [
        Optimisers.AMSGrad(),
    ],
    [
        Optimisers.Adam(),
    ],
    [
        Optim.NelderMead(),
        Optimisers.OptimiserChain(Optimisers.AMSGrad(), Optimisers.WeightDecay()),
        # Optim.Newton(),
    ],
    [
        Optim.SimulatedAnnealing(),
        Optimisers.OptimiserChain(Optimisers.AMSGrad(), Optimisers.WeightDecay()),
        # Optim.Newton(),
    ],
    [
        Optim.SimulatedAnnealing(),
        Optim.NelderMead(),
        Optimisers.OptimiserChain(Optimisers.AMSGrad(), Optimisers.WeightDecay()),
        # Optim.Newton(),
    ],
]

allparams = Dict(
    "nvars" => 1,
    # "nvars" => 2 .^ (0:3),
    "n" => 2^10,
    # "n" => 2 .^ (7, 10, 13),
    "data_dist" => Beta(2, 4),
    "n_hidden_rate" => 4,
    # "n_hidden_rate" => 2 .^ (0:3),
    "tspan_end" => 8,
    # "tspan_end" => 2 .^ (0:4),
    "alg" => BS3(; thread=OrdinaryDiffEq.True()),
    # "alg" => [BS3(), Tsit5(), TsitPap8(), Vern9(), Feagin14()],
    "autojacvec_i" => 1,
    # "autojacvec_i" => collect(1:length(autojacvec_l)),
    "optimizers_i" => 1,
    # "optimizers_i" => collect(1:length(optimizers_l)),
)
dicts = dict_list(allparams)


function gen_data(nvars, n, data_dist=Beta(2, 4))
    rand(data_dist, nvars, n)
end

function makesim_gendata(d::Dict)
    @unpack nvars, n, data_dist = d
    r = gen_data(nvars, n, data_dist)
    fulld = copy(d)
    fulld["r"] = r
    fulld
end

function makesim(d::Dict)
    @unpack nvars, n, data_dist, n_hidden_rate, tspan_end, alg, autojacvec_i, optimizers_i = d
    autojacvec = autojacvec_l[autojacvec_i]
    optimizers = optimizers_l[optimizers_i]

    n_hidden = n_hidden_rate * nvars
    tspan = convert.(Float32, (0, tspan_end))
    fulld = copy(d)
    fulld["n_hidden"] = n_hidden
    fulld["tspan"] = tspan

    config = Dict(
        "nvars" => nvars,
        "n" => n,
        "data_dist" => data_dist,
    )
    data, fn = produce_or_load(makesim_gendata, config, datadir("synthetic-gendata"))
    r = data["r"]

    # Model
    nn = Lux.Chain(Lux.Dense(nvars => n_hidden, tanh), Lux.Dense(n_hidden => nvars, tanh))
    # nn = Lux.Dense(nvars => nvars, tanh)
    icnf = construct(RNODE, nn, nvars; tspan, compute_mode = ZygoteMatrixMode, sol_kwargs = Dict(:alg => alg, :sensealg => QuadratureAdjoint(; autodiff = true, autojacvec)))

    # Training
    df = DataFrame(transpose(r), :auto)
    model = ICNFModel(icnf; optimizers, adtype = Optimization.AutoForwardDiff())
    mach = machine(model, df)
    fit!(mach)
    ps, st = fitted_params(mach)
    rpt = report(mach)
    fulld["elapsed_time"] = rpt.stats.time

    # Use It
    dist = ICNFDist(icnf, ps, st)
    actual_pdf = pdf.(data_dist, vec(r))
    estimated_pdf = pdf(dist, r)
    # new_data = rand(dist, n)

    # Evaluation
    mad_ = meanad(estimated_pdf, actual_pdf)
    msd_ = msd(estimated_pdf, actual_pdf)
    tv_dis = totalvariation(estimated_pdf, actual_pdf) / n
    fulld["meanad"] = mad_
    fulld["msd"] = msd_
    fulld["totalvariation"] = tv_dis

    # Plot
    # hist_data = collect((0:0.001:1)')
    # hist_actual_pdf = pdf.(data_dist, vec(hist_data))
    # hist_estimated_pdf = pdf(dist, hist_data)

    # p = plot(data_dist, label="actual")
    # p = plot(hist_data', hist_actual_pdf, label="actual")
    # p = plot(Base.Fix1(pdf, data_dist), 0, 1, label="actual")
    p = plot(x -> pdf(data_dist, x), 0, 1, label="actual")
    # p = plot!(p, hist_data', hist_estimated_pdf, label="estimated")
    # p = plot!(p, Base.Fix1(pdf, dist), 0, 1, label="estimated")
    p = plot!(p, x -> pdf(dist, vcat(x)), 0, 1, label="estimated")
    # display(p)
    savefig(p, plotsdir("synthetic-sims", savename(d, "png")))
    fulld
end

for (i, d) in enumerate(dicts)
    # f = makesim(d)
    produce_or_load(makesim, d, datadir("synthetic-sims-res"))
    # @tagsave(datadir("simulations", savename(d, "jld2")), f)
end

df = collect_results(datadir("synthetic-sims-res"))

# safesave(datadir("ana", "linear.jld2"), @strdict analysis)

using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

allparams = Dict(
    "nvars" => 2^0,
    # "nvars" => 2 .^ (0:3),
    "n" => 2^10,
    # "n" => 2 .^ (7, 10, 13),
    "data_dist" => Beta{Float32}(2.0f0, 4.0f0),
    "n_hidden_rate" => 2 .^ (0:3),
    "tspan_end" => 2^0,
    # "tspan_end" => 2 .^ (0:4),
)
dicts = dict_list(allparams)
dicts = convert.(Dict{String, Any}, dicts)

function gen_data(nvars, n, data_dist = Beta{Float32}(2.0f0, 4.0f0))
    convert.(Float32, rand(data_dist, nvars, n))
end

function makesim_gendata(d::Dict)
    @unpack nvars, n, data_dist = d
    fulld = copy(d)

    r = gen_data(nvars, n, data_dist)
    fulld["r"] = r

    fulld
end

function makesim_expr(d::Dict)
    @unpack nvars, n, data_dist, n_hidden_rate, tspan_end = d
    fulld = copy(d)

    n_hidden = n_hidden_rate * nvars
    tspan = convert.(Float32, (0, tspan_end))
    fulld["n_hidden"] = n_hidden
    fulld["tspan"] = tspan

    config = Dict("nvars" => nvars, "n" => n, "data_dist" => data_dist)
    data, fn = produce_or_load(makesim_gendata, config, datadir("synthetic-gendata"))
    r = data["r"]

    # Model
    nn = FluxCompatLayer(
        f32(
            Flux.Chain(
                Flux.Dense(nvars => n_hidden, tanh),
                Flux.Dense(n_hidden => nvars, tanh),
            ),
        ),
    )
    icnf = construct(RNODE, nn, nvars; tspan, compute_mode = ZygoteMatrixMode, sol_kwargs)

    # Training
    df = DataFrame(transpose(r), :auto)
    model = ICNFModel(icnf; optimizers)
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
    mad_ = Distances.meanad(estimated_pdf, actual_pdf)
    msd_ = Distances.msd(estimated_pdf, actual_pdf)
    tv_dis = Distances.totalvariation(estimated_pdf, actual_pdf) / n
    fulld["meanad"] = mad_
    fulld["msd"] = msd_
    fulld["totalvariation"] = tv_dis

    # Plot
    # hist_data = collect((0:0.001:1)')
    # hist_actual_pdf = pdf.(data_dist, vec(hist_data))
    # hist_estimated_pdf = pdf(dist, hist_data)

    # p = plot(data_dist; label = "actual")
    # p = plot(hist_data', hist_actual_pdf; label = "actual")
    # p = plot(Base.Fix1(pdf, data_dist), 0, 1; label = "actual")
    p = plot(x -> pdf(data_dist, x), 0, 1; label = "actual")
    # p = plot!(p, hist_data', hist_estimated_pdf; label = "estimated")
    # p = plot!(p, Base.Fix1(pdf, dist), 0, 1; label = "estimated")
    p = plot!(p, x -> pdf(dist, convert.(Float32, vcat(x))), 0, 1; label = "estimated")
    savefig(p, plotsdir("synthetic-sims", savename(d, "png")))

    fulld
end

for (i, d) in enumerate(dicts)
    CUDA.allowscalar() do
        produce_or_load(makesim_expr, d, datadir("synthetic-sims-res"))
    end
end

df = collect_results(datadir("synthetic-sims-res"))

using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

const allparams = Dict(
    # data
    "nvars" => 2^0,
    # "nvars" => 2 .^ (0:3),
    "n" => 2^12,
    # "n" => 2 .^ (7, 10, 13),
    "data_dist" => Beta{Float32}(2.0f0, 4.0f0),

    # nn
    "n_hidden_rate" => 2^2,
    # "n_hidden_rate" => 2 .^ (0:3),
    "arch" => ["Dense", "Dense-ML"],

    # construct
    "tspan_end" => [2^0, 2^5],
    # "tspan_end" => 2 .^ (0:4),

    # ICNFModel
    "n_epochs" => 2^10 * 2^3,
    "batch_size" => 2^12,
)
const dicts = convert.(Dict{String, Any}, dict_list(allparams))

@inline function gen_data(nvars, n, data_dist)
    convert.(Float32, rand(data_dist, nvars, n))
end

@inline function makesim_gendata(d::Dict)
    @unpack nvars, n, data_dist = d
    fulld = copy(d)

    r = gen_data(nvars, n, data_dist)
    fulld["r"] = r

    fulld
end

@inline function makesim_expr(d::Dict)
    @unpack nvars, n, data_dist, n_hidden_rate, tspan_end, arch, batch_size, n_epochs = d
    fulld = copy(d)

    n_hidden = n_hidden_rate * nvars
    tspan = convert.(Float32, (0, tspan_end))
    fulld["n_hidden"] = n_hidden
    fulld["tspan"] = tspan

    config = Dict("nvars" => nvars, "n" => n, "data_dist" => data_dist)
    data, fn = produce_or_load(makesim_gendata, config, datadir("synthetic-gendata"))
    r = data["r"]

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
    icnf = construct(
        RNODE,
        nn,
        nvars;
        tspan,
        compute_mode = ZygoteMatrixMode,
        sol_kwargs,
        λ₁ = 1.0f-1,
        λ₂ = 1.0f-1,
    )

    df = DataFrame(transpose(r), :auto)
    model = ICNFModel(icnf; optimizers, batch_size, n_epochs)
    mach = machine(model, df)
    fit!(mach)
    ps, st = fitted_params(mach)
    rpt = report(mach)
    fulld["elapsed_time"] = rpt.stats.time

    dist = ICNFDist(icnf, ps, st)
    actual_pdf = pdf.(data_dist, vec(r))
    estimated_pdf = pdf(dist, r)

    mad_ = Distances.meanad(estimated_pdf, actual_pdf)
    msd_ = Distances.msd(estimated_pdf, actual_pdf)
    tv_dis = Distances.totalvariation(estimated_pdf, actual_pdf) / n
    fulld["meanad"] = mad_
    fulld["msd"] = msd_
    fulld["totalvariation"] = tv_dis

    f = Figure()
    ax = Makie.Axis(f[1, 1]; title = "Result")
    lines!(ax, 0.0f0 .. 1.0f0, x -> pdf(data_dist, x); label = "actual")
    lines!(ax, 0.0f0 .. 1.0f0, x -> pdf(dist, vcat(x)); label = "estimated")
    axislegend(ax)
    save(plotsdir("synthetic-sims", savename(d, "svg")), f)
    save(plotsdir("synthetic-sims", savename(d, "png")), f)

    fulld
end

for (i, d) in enumerate(dicts)
    if use_gpu_nn_train || use_gpu_nn_test
        CUDA.allowscalar() do
            produce_or_load(makesim_expr, d, datadir("synthetic-sims-res"))
        end
    else
        produce_or_load(makesim_expr, d, datadir("synthetic-sims-res"))
    end
end

df = collect_results(datadir("synthetic-sims-res"))

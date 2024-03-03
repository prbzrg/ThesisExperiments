using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

const allparams = Dict(
    # data
    "nvars" => 1,
    # "nvars" => 2 .^ (0:3),
    "n" => 2^12,
    # "n" => 2 .^ (7, 10, 13),
    "data_dist" => Beta{Float32}(2.0f0, 4.0f0),

    # train
    "rnode_reg" => eps_sq[3],
    "steer_reg" => eps_sq[4],
    "nvld" => collect(1:3),

    # nn
    "n_hidden_rate" => 0,
    # "n_hidden_rate" => 2 .^ (0:3),
    "arch" => "Dense",
    # "arch" => ["Dense", "Dense-ML"],
    "have_bias" => false,

    # construct
    "tspan_end" => 13,
    # "tspan_end" => 2 .^ (0:4),

    # ICNFModel
    "n_epochs" => 2^10,
    "batch_size" => 2^7,
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
    @unpack nvars,
    n,
    data_dist,
    rnode_reg,
    steer_reg,
    n_hidden_rate,
    arch,
    have_bias,
    tspan_end,
    batch_size,
    n_epochs = d
    fulld = copy(d)

    n_hidden = n_hidden_rate * nvars
    tspan = convert.(Float32, (0, tspan_end))
    fulld["n_hidden"] = n_hidden
    fulld["tspan"] = tspan

    config = Dict("nvars" => nvars, "n" => n, "data_dist" => data_dist)
    data, fn = produce_or_load(makesim_gendata, config, datadir("synthetic-gendata"))
    r = data["r"]

    if arch == "Dense"
        nn = Lux.Dense(nvars * 2 => nvars * 2, tanh; use_bias = have_bias)
    elseif arch == "Dense-ML"
        nn = Lux.Chain(
            Lux.Dense(nvars * 2 => n_hidden * 2, tanh; use_bias = have_bias),
            Lux.Dense(n_hidden * 2 => nvars * 2, tanh; use_bias = have_bias),
        )
    else
        error("Not Imp")
    end
    icnf = construct(
        RNODE,
        nn,
        nvars,
        nvars;
        tspan,
        compute_mode = ZygoteMatrixMode,
        steer_rate = steer_reg,
        sol_kwargs = sol_kwargs_base,
        # sol_kwargs,
        λ₁ = rnode_reg,
        λ₂ = rnode_reg,
    )

    df = DataFrame(transpose(r), :auto)
    model = ICNFModel(icnf; optimizers, batch_size, n_epochs)
    mach = machine(model, df)
    fit!(mach)
    ps, st = fitted_params(mach)
    rpt = report(mach)
    fulld["elapsed_time"] = rpt.stats.time

    dist = ICNFDist(icnf, TestMode(), ps, st)
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

using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

const allparams = Dict(
    # test
    "n_iter_rec" => 300,
    "sel_a" => vcat(["min", "max"], 1:128),

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
    "steer_reg" => eps_sq[4],
    "ode_reltol" => eps_sq[3],
    "tspan_end" => 1,

    # nn
    "n_hidden_rate" => 0,
    # "arch" => "Dense-ML",
    "arch" => "Dense",
    # "back" => "Lux",
    "back" => "Flux",
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

d = first(dicts)

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

data, fn = produce_or_load(x -> error(), d2, datadir("gen-ld-patch"))
ptchs = data["ptchs"]
n_pts = size(ptchs, 4)
fulld["n_pts"] = n_pts

data, fn = produce_or_load(x -> error(), d3, datadir("ld-ct-sims"))
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
        compute_mode = ZygoteVectorMode,
        resource = CUDALibs(),
    )
else
    icnf = construct(FFJORD, nn, nvars, naug_vl; tspan, compute_mode = ZygoteVectorMode)
end

# way 4
# smp_f = rand(Float32, 36)

# way 3
# smp_f = ones(Float32, 36)
smp_f = zeros(Float32, 36)

# way 2
# ptchs2 = reshape(ptchs, 36, size(ptchs, 4) * 128);
# high_std = argmin(std.(eachcol(ptchs2)))
# smp_f = ptchs2[:, high_std]

# way 1
# smp = ptchs[:, :, 1, 10_000, 10]
# smp_f = vec(smp)
# smp_f = MLUtils.flatten(smp)

prob = ContinuousNormalizingFlows.inference_prob(icnf, TestMode(), smp_f, ps, st)
n_aug = ContinuousNormalizingFlows.n_augment(icnf, TestMode())
sol = solve(prob; icnf.sol_kwargs...)
display(sol.stats)
f = Figure()
ax = Makie.Axis(f[1, 1]; title = "Flow")
broadcast(x -> lines!(ax, sol.t, x), eachrow(sol[begin:(end - n_aug - 1), :]))
# ax2 = Makie.Axis(f[1, 2]; title = "Log")
# lines!(ax2, sol.t, sol[(end - n_aug), :])
save(plotsdir("plot-lines", "flow_new.svg"), f)
save(plotsdir("plot-lines", "flow_new.png"), f)

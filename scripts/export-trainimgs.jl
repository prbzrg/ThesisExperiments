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
    "rnode_reg" => eps_sq[3],
    "steer_reg" => eps_sq[4],
    "ode_reltol" => eps_sq[3],
    "tspan_end" => 13,

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

if isnothing(have_bias)
    have_bias = true
end

tspan = convert.(Float32, (0, tspan_end))

data, fn = produce_or_load(x -> error(), d3, datadir("ld-ct-sims"))
@unpack sp = data

imgs = load(gt_train_fn)["data"]

# reconstruction

resl_h, resl_w = size(imgs)
resl = (resl_w * 2 * length(sp) ÷ 2, resl_h * 2 * 2)
f = Figure(; size = resl)
# for (i, idx) in enumerate(sp)
#     image!(Makie.Axis(f[1, i]), rotr90(imgs[:, :, idx]))
# end

image!(Makie.Axis(f[1, 1]), rotr90(imgs[:, :, sp[1]]))
image!(Makie.Axis(f[1, 2]), rotr90(imgs[:, :, sp[2]]))
image!(Makie.Axis(f[1, 3]), rotr90(imgs[:, :, sp[4]]))
image!(Makie.Axis(f[2, 1]), rotr90(imgs[:, :, sp[3]]))
image!(Makie.Axis(f[2, 2]), rotr90(imgs[:, :, sp[5]]))
image!(Makie.Axis(f[2, 3]), rotr90(imgs[:, :, sp[6]]))

save(plotsdir("trainimgs2.svg"), f)
save(plotsdir("trainimgs2.png"), f)

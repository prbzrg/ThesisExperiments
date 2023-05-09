using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

gt_fn = datadir("lodoct", "ground_truth_train_000.hdf5")
obs_fn = datadir("lodoct", "observation_test_000.hdf5")

allparams = Dict(
    "p_s" => 8,
    # "p_s" => [4, 6, 8],
    "n_epochs" => 2,
)
dicts = dict_list(allparams)
dicts = convert.(Dict{String, Any}, dicts)
fdt = dicts[1]

@unpack p_s, n_epochs = fdt

# tspan = convert.(Float32, (0, tspan_end))

config = Dict("p_s" => p_s)
data = load(datadir("gen-ld-patch", savename(config, "jld2")))
ptchs = data["ptchs"]
n_pts = size(ptchs, 4)

data = load(datadir("ld-ct-sims", savename(fdt, "jld2")))
@unpack ps, st = data

nvars = p_s * p_s
rs_f(x) = reshape(x, (p_s, p_s, 1, :))

nn = FluxCompatLayer(
    f32(
        Flux.Chain(
            rs_f,
            Flux.Parallel(
                +,
                Flux.Conv((3, 3), 1 => 3, tanh; dilation = 1, pad = Flux.SamePad()),
                Flux.Conv((3, 3), 1 => 3, tanh; dilation = 2, pad = Flux.SamePad()),
                Flux.Conv((3, 3), 1 => 3, tanh; dilation = 3, pad = Flux.SamePad()),
                # Flux.Conv((3, 3), 1 => 3, tanh; dilation=4, pad=Flux.SamePad()),
                # Flux.Conv((3, 3), 1 => 3, tanh; dilation=5, pad=Flux.SamePad()),
                # Flux.Conv((3, 3), 1 => 3, tanh; dilation=6, pad=Flux.SamePad()),
                # Flux.Conv((3, 3), 1 => 3, tanh; dilation=7, pad=Flux.SamePad()),
                # Flux.Conv((3, 3), 1 => 3, tanh; dilation=8, pad=Flux.SamePad()),
                # Flux.Conv((3, 3), 1 => 3, tanh; dilation=9, pad=Flux.SamePad()),
            ),
            Flux.Conv((3, 3), 3 => 1, tanh; pad = Flux.SamePad()),
            MLUtils.flatten,
            # vec,
        ),
    ),
)

icnf = construct(
    FFJORD,
    nn,
    nvars;
    compute_mode = ZygoteMatrixMode,
    # array_type = CuArray,
    sol_kwargs,
)

icnf_f(x) = loss(icnf, x, ps, st)
ptchnr = PatchNR(; icnf_f, n_pts, p_s)
obs_y = load(obs_fn)["data"]
obs_y = obs_y[:, :, 1]

# _loss(ps, θ) = recn_loss(ptchnr, ps, obs_y)
_loss(ps, θ, args...) = recn_loss_pt1(ptchnr, ps, obs_y) + recn_loss_pt2(ptchnr, ps, obs_y)
function _loss_gd(ps, θ, args...)
    # pt1 = ReverseDiff.gradient(x -> recn_loss_pt1(ptchnr, x, obs_y), ps)
    pt1 = ForwardDiff.gradient(x -> recn_loss_pt1(ptchnr, x, obs_y), ps)

    # pt2 = ForwardDiff.gradient(x -> recn_loss_pt2(ptchnr, x, obs_y), ps)
    pt2 = only(Zygote.gradient(x -> recn_loss_pt2(ptchnr, x, obs_y), ps))
    pt1 + pt2
end

fp = vec(cstm_fbp(obs_y))
app_icnf = ptchnr
# f_tp(x) = sum(app_icnf.forward_op(reshape(x, (app_icnf.w_d, app_icnf.w_d))))
l_tp(x) = _loss(x, nothing)

# @timev f_tp(fp)
# @timev l_tp(fp)
# @timev ReverseDiff.gradient(f_tp, fp)
# @timev ReverseDiff.gradient(l_tp, fp)
# @timev ForwardDiff.gradient(l_tp, fp)
# @timev Zygote.gradient(l_tp, fp)

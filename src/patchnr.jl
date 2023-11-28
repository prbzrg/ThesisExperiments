# i = 1
# i = 3
# i = 9 -> 112
# i = 27 -> 38
# i = 37
# i = 111
# i = 333
# i = 999

const MU_WATER = 20
const MU_AIR = 0.02
const MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER

const lmbd_ps = Dict(4 => 1600.0f0, 6 => 700.0f0, 8 => 400.0f0, 10 => 250.0f0)

@inline function cstm_radon(x)
    # radon_transform_new(prep_img_radon(x), range(0, 1; length = 1000), -256:256)
    # radon_transform_new(prep_img_radon(x), range(0, π; length = 1000), -256:256)
    radon_transform_new(prep_img_radon(x), range(0, π; length = 38), -256:256)
end

@concrete struct PatchNR
    icnf_f::Function
    n_pts::Integer
    p_s::Integer
    w_d::Integer
    s::Integer
    d::Integer
    reduce_rate::Integer
    n_skp::Integer
    Nₚ::Integer
    N₀::Integer
    μ::AbstractFloat
    λ::AbstractFloat
    forward_op::Function
    sel_pts::AbstractArray

    @inline function PatchNR(;
        icnf_f::Function,
        n_pts::Integer,
        p_s::Integer,
        w_d::Integer = 362,
        s::Integer = p_s * p_s,
        d::Integer = w_d * w_d,
        # reduce_rate::Integer = 100,
        reduce_rate::Integer = 4,
        n_skp::Integer = 27,
        # Nₚ::Integer = n_pts,
        Nₚ::Integer = 40000,
        # Nₚ::Integer = n_pts ÷ reduce_rate,
        N₀::Integer = 4096,
        μ::AbstractFloat = MU_MAX,
        λ::AbstractFloat = lmbd_ps[p_s],
        # λ::AbstractFloat = convert(Float32, lmbd_ps[p_s] * (s / Nₚ)),
        forward_op::Function = cstm_radon,
    )
        sel_pts = sample(1:n_pts, Nₚ)
        new(
            icnf_f,
            n_pts,
            p_s,
            w_d,
            s,
            d,
            reduce_rate,
            n_skp,
            Nₚ,
            N₀,
            μ,
            λ,
            forward_op,
            sel_pts,
        )
    end
end

# @inline function recn_loss(app_icnf::PatchNR, x, y)
#     y = y[:, 1:(app_icnf.n_skp):1000]
#     N₀ = app_icnf.N₀
#     forw = app_icnf.forward_op(reshape(x, (app_icnf.w_d, app_icnf.w_d)))
#     pt_1st = sum(exp.(-forw) * N₀ + exp.(-y) * N₀ .* (forw .- log(N₀)))
#     pt_2ed = app_icnf.icnf_f(nr_patchs(app_icnf, x))
#     # pt_2ed = extract_patch_one(app_icnf.icnf_f, reshape(x, (app_icnf.w_d, app_icnf.w_d, 1, :)), app_icnf.p_s, app_icnf.p_s)
#     # pt_2ed = mean(app_icnf.icnf_f.(eachcol(nr_patchs(app_icnf, x))))
#     pt_1st + app_icnf.λ * pt_2ed
# end

# @inline function recn_loss_pt1(app_icnf::PatchNR, x, y)
#     y = y[:, 1:(app_icnf.n_skp):1000]
#     N₀ = app_icnf.N₀
#     forw = app_icnf.forward_op(reshape(x, (app_icnf.w_d, app_icnf.w_d)))
#     pt_1st = sum(exp.(-forw) * N₀ + exp.(-y) * N₀ .* (forw .- log(N₀)))
#     pt_1st
# end

# main
# @inline function recn_loss_pt1(app_icnf::PatchNR, x, y)
#     y = y[:, 1:(app_icnf.n_skp):1000]
#     N₀ = app_icnf.N₀
#     μ = app_icnf.μ
#     forw = app_icnf.forward_op(reshape(x, (app_icnf.w_d, app_icnf.w_d)))
#     pt_1st = sum(exp.(-forw * μ) * N₀ - exp.(-y * μ) * N₀ .* (-forw * μ .+ log(N₀)))
#     # pt_1st = sum((exp.(-forw * μ)*N₀ - exp.(-y * μ)*N₀) .* (-forw * μ .+ log(N₀)))
#     pt_1st
# end

@inline function recn_loss_pt1(app_icnf::PatchNR, x, y)
    x = reshape(rotl90(reshape(x, (362, 362))), (1, 1, 362, 362))
    y = reshape(rotl90(y), (1, 1, 1000, 513))
    first_part(x, y)
end

@inline function recn_loss_pt2(app_icnf::PatchNR, x, y, use_gpu = use_gpu_nn_test)
    new_pts = nr_patchs(app_icnf, x)
    if use_gpu
        new_pts = gdev(new_pts)
    end
    # app_icnf.λ * mean(app_icnf.icnf_f.(eachcol(new_pts)))
    app_icnf.λ * app_icnf.icnf_f(new_pts)
end

# @inline function recn_loss_pt2(app_icnf::PatchNR, x, y)
#     app_icnf.λ * extract_patch_one(
#         app_icnf.icnf_f,
#         reshape(x, (app_icnf.w_d, app_icnf.w_d, 1, :)),
#         app_icnf.p_s,
#         app_icnf.p_s,
#     )
# end

@inline function nr_patchs(app_icnf::PatchNR, x)
    p_s = app_icnf.p_s
    w_d = app_icnf.w_d
    img = reshape(x, (w_d, w_d))
    # img = reshape(x, (w_d, w_d, 1, :))
    ptchs = extract_patch_2_rand(img, p_s, p_s, app_icnf.Nₚ)
    # ptchs = extract_patch_rand(img, p_s, p_s, app_icnf.Nₚ)
    # ptchs = extract_patch_2(img, p_s, p_s)
    # ptchs = extract_patch(img, p_s, p_s)
    # ptchs = reshape(ptchs, (p_s, p_s, 1, :))
    x_pts = MLUtils.flatten(ptchs)
    x_pts
    # x_pts[:, app_icnf.sel_pts]
    # sel_pts = rand(1:(app_icnf.n_pts), app_icnf.Nₚ)
    # x_pts[:, sel_pts]
end

# main
# @inline function recn_loss_pt1_grad(ptchnr, ps, obs_y)
#     # ForwardDiff.gradient(x -> recn_loss_pt1(ptchnr, x, obs_y), ps)
#     ReverseDiff.gradient(x -> recn_loss_pt1(ptchnr, x, obs_y), ps)
# end

@inline function recn_loss_pt1_grad(ptchnr, ps, obs_y)
    ps = reshape(rotl90(reshape(ps, (362, 362))), (1, 1, 362, 362))
    obs_y = reshape(rotl90(obs_y), (1, 1, 1000, 513))
    vec(rotr90(grad_first_part(ps, obs_y)[1, 1, :, :]))
end

@inline function recn_loss_pt2_grad(ptchnr, ps, obs_y)
    # ForwardDiff.gradient(x -> recn_loss_pt2(ptchnr, x, obs_y), ps)

    only(Zygote.gradient(let obs_y = obs_y
        x -> recn_loss_pt2(ptchnr, x, obs_y)
    end, ps))
end

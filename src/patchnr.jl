# i = 1
# i = 3
# i = 9 -> 112
# i = 27 -> 38
# i = 37
# i = 111
# i = 333
# i = 999

MU_WATER = 20
MU_AIR = 0.02
MU_MAX = 3071 * (MU_WATER - MU_AIR) / 1000 + MU_WATER

cstm_radon(x) = radon_transform_new(prep_img_radon(x), range(0, π; length = 38), -256:256)
# cstm_radon(x) = radon_transform_new(prep_img_radon(x), range(0, π; length=1000), -256:256)
# cstm_radon(x) = radon_transform_new(prep_img_radon(x), range(0, 1; length=1000), -256:256)

# 4 * 4 = 1600
# 6 * 6 = 700
# 8 * 8 = 400
# 10 * 10 = 250

mutable struct PatchNR
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

    function PatchNR(;
        icnf_f::Function,
        n_pts::Integer,
        p_s::Integer = 6,
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
        λ::AbstractFloat = convert(Float32, 700 * (s / Nₚ)),
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

# function recn_loss(app_icnf::PatchNR, x, y)
#     y = y[:, 1:(app_icnf.n_skp):1000]
#     N₀ = app_icnf.N₀
#     forw = app_icnf.forward_op(reshape(x, (app_icnf.w_d, app_icnf.w_d)))
#     pt_1st = sum(exp.(-forw) * N₀ + exp.(-y) * N₀ .* (forw .- log(N₀)))
#     pt_2ed = app_icnf.icnf_f(nr_patchs(app_icnf, x))
#     # pt_2ed = extract_patch_50(app_icnf.icnf_f, reshape(x, (app_icnf.w_d, app_icnf.w_d, 1, :)), app_icnf.p_s, app_icnf.p_s)
#     # pt_2ed = mean(app_icnf.icnf_f.(eachcol(nr_patchs(app_icnf, x))))
#     pt_1st + app_icnf.λ * pt_2ed
# end

# function recn_loss_pt1(app_icnf::PatchNR, x, y)
#     y = y[:, 1:(app_icnf.n_skp):1000]
#     N₀ = app_icnf.N₀
#     forw = app_icnf.forward_op(reshape(x, (app_icnf.w_d, app_icnf.w_d)))
#     pt_1st = sum(exp.(-forw) * N₀ + exp.(-y) * N₀ .* (forw .- log(N₀)))
#     pt_1st
# end

# main
# function recn_loss_pt1(app_icnf::PatchNR, x, y)
#     y = y[:, 1:(app_icnf.n_skp):1000]
#     N₀ = app_icnf.N₀
#     μ = app_icnf.μ
#     forw = app_icnf.forward_op(reshape(x, (app_icnf.w_d, app_icnf.w_d)))
#     pt_1st = sum(exp.(-forw * μ) * N₀ - exp.(-y * μ) * N₀ .* (-forw * μ .+ log(N₀)))
#     # pt_1st = sum((exp.(-forw * μ)*N₀ - exp.(-y * μ)*N₀) .* (-forw * μ .+ log(N₀)))
#     pt_1st
# end

function recn_loss_pt1(app_icnf::PatchNR, x, y)
    x = reshape(rotl90(reshape(x, (362, 362))), (1, 1, 362, 362))
    y = reshape(rotl90(y), (1, 1, 1000, 513))
    first_part(x, y)
end

function recn_loss_pt2(app_icnf::PatchNR, x, y)
    # app_icnf.λ * mean(app_icnf.icnf_f.(eachcol(nr_patchs(app_icnf, x))))
    app_icnf.λ * app_icnf.icnf_f(nr_patchs(app_icnf, x))
end

# function recn_loss_pt2(app_icnf::PatchNR, x, y)
#     app_icnf.λ * extract_patch_50(
#         app_icnf.icnf_f,
#         reshape(x, (app_icnf.w_d, app_icnf.w_d, 1, :)),
#         app_icnf.p_s,
#         app_icnf.p_s,
#     )
# end

function nr_patchs(app_icnf::PatchNR, x)
    p_s = app_icnf.p_s
    w_d = app_icnf.w_d
    img = reshape(x, (w_d, w_d, 1, :))
    ptchs = extract_patch(img, p_s, p_s)
    # ptchs = extract_patch_33(img, p_s, p_s)
    ptchs = reshape(ptchs, (p_s, p_s, 1, :))
    x_pts = MLUtils.flatten(ptchs)
    # x_pts
    # x_pts[:, app_icnf.sel_pts]
    sel_pts = rand(1:(app_icnf.n_pts), app_icnf.Nₚ)
    x_pts[:, sel_pts]
end

# main
# function recn_loss_pt1_grad(ptchnr, ps, obs_y)
#     # ForwardDiff.gradient(x -> recn_loss_pt1(ptchnr, x, obs_y), ps)
#     ReverseDiff.gradient(x -> recn_loss_pt1(ptchnr, x, obs_y), ps)
# end

function recn_loss_pt1_grad(ptchnr, ps, obs_y)
    ps = reshape(rotl90(reshape(ps, (362, 362))), (1, 1, 362, 362))
    obs_y = reshape(rotl90(obs_y), (1, 1, 1000, 513))
    vec(rotr90(grad_first_part(ps, obs_y)[1, 1, :, :]))
end

function recn_loss_pt2_grad(ptchnr, ps, obs_y)
    # ForwardDiff.gradient(x -> recn_loss_pt2(ptchnr, x, obs_y), ps)
    only(Zygote.gradient(x -> recn_loss_pt2(ptchnr, x, obs_y), ps))
end

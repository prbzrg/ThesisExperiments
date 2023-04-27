# function radon_transform_old(I::AbstractMatrix{T}, θ::AbstractRange, t::AbstractRange)::AbstractMatrix{T} where T <: Real
#     P = zeros(T, length(t), length(θ))
#     I_h, I_w = size(I)

#     for j in 1:I_w, i in 1:I_h, (k, θₖ) in enumerate(θ)
#         x = j - I_w / 2 + 0.5
#         y = i - I_h / 2 + 0.5
#         sp, cp = sincospi(θₖ)
#         t′ = x * cp + y * sp

#         a = round(Int, (t′ - minimum(t)) / step(t) + 1)

#         (a < 1 || a > length(t)) && continue
#         α = abs(t′ - t[a])

#         I′ = I[i, j]
#         P[a, k] += (1 - α) * I′

#         (a+1 > length(t)) && continue
#         P[a+1, k] += α * I′
#     end
#     P
# end

# function radon_transform_new(I::AbstractMatrix{T}, θ::AbstractRange, t::AbstractRange)::AbstractMatrix{T} where T <: Real
#     P = zeros(T, length(t), length(θ))
#     I_tp = zeros(T, length(t), length(t))
#     I_h, I_w = size(I)

#     m_h = round(Int, length(t) / 2 - I_h / 2)
#     m_w = round(Int, length(t) / 2 - I_w / 2)

#     for (i, j) in collect(product(1:I_h, 1:I_w))
#         I_tp[i+m_h, j+m_w] = I[i, j]
#     end
#     for (k, θₖ) in collect(enumerate(θ))
#         P[:, k] .= sum(imrotate(I_tp, θₖ, size(I_tp); fillvalue=zero(T)), dims=2)
#     end
#     P
# end

function radon_transform_new(I::AbstractMatrix{T}, θ::AbstractRange, t::AbstractRange)::AbstractMatrix{T} where T <: Real
    P = Zygote.Buffer(I, length(t), length(θ))
    I_tp = Zygote.Buffer(I, length(t), length(t))
    I_tp[:, :] = zeros(T, length(t), length(t))
    I_h, I_w = size(I)

    m_h = round(Int, length(t) / 2 - I_h / 2)
    m_w = round(Int, length(t) / 2 - I_w / 2)

    for (i, j) in collect(product(1:I_h, 1:I_w))
        I_tp[i+m_h, j+m_w] = I[i, j]
    end
    for (k, θₖ) in collect(enumerate(θ))
        P[:, k] = sum(imrotate(copy(I_tp), θₖ, size(I_tp); fillvalue=zero(T)), dims=2)
    end
    # P
    copy(P)
end

# function radon_transform_new_thrd(I::AbstractMatrix{T}, θ::AbstractRange, t::AbstractRange)::AbstractMatrix{T} where T <: Real
#     P = zeros(T, length(t), length(θ))
#     I_tp = zeros(T, length(t), length(t))
#     I_h, I_w = size(I)

#     m_h = round(Int, length(t) / 2 - I_h / 2)
#     m_w = round(Int, length(t) / 2 - I_w / 2)

#     @threads for (i, j) in collect(product(1:I_h, 1:I_w))
#         I_tp[i+m_h, j+m_w] = I[i, j]
#     end
#     @threads for (k, θₖ) in collect(enumerate(θ))
#         P[:, k] .= sum(imrotate(I_tp, θₖ, size(I_tp); fillvalue=zero(eltype(I_tp))), dims=2)
#     end
#     P
# end

# function radon_transform_new(I::AbstractMatrix{T}, θ::AbstractRange)::AbstractMatrix{T} where T <: Real
#     I_h, I_w = size(I)
#     t = ceil(Int, sqrt(I_h^2 + I_w^2))
#     t = iseven(t) ? t+1 : t
#     tl = (t-1) ÷ 2
#     t = -tl:tl
#     radon_transform_new(I, θ, t)
# end

prep_img_radon(img) = reverse(img; dims=2) |> rotl90

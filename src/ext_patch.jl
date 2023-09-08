@inline function extract_patch(
    img::AbstractArray{T, 4},
    p_w::Integer,
    p_h::Integer,
)::AbstractArray{T, 5} where {T <: Real}
    n_h, n_w, n_c, n_b = size(img)
    n_h -= p_h + 1
    n_w -= p_w + 1
    # out_img = zeros(T, p_h, p_w, n_c, n_h * n_w, n_b)
    out_img = Zygote.Buffer(img, p_h, p_w, n_c, n_h * n_w, n_b)
    @inbounds for i_b in 1:n_b
        i_pc = 1
        @inbounds for i_w in 1:n_w, i_h in 1:n_h
            out_img[:, :, :, i_pc, i_b] =
                img[i_h:(i_h + p_h - 1), i_w:(i_w + p_w - 1), :, i_b]
            i_pc += one(i_pc)
        end
    end
    # out_img
    copy(out_img)
end

@inline function extract_patch_rand(
    img::AbstractArray{T, 4},
    p_w::Integer,
    p_h::Integer,
    n_pts::Integer,
)::AbstractArray{T, 5} where {T <: Real}
    n_r_pts = round(Int, sqrt(n_pts))
    n_h, n_w, n_c, n_b = size(img)
    n_h -= p_h + 1
    n_w -= p_w + 1
    # out_img = zeros(T, p_h, p_w, n_c, n_r_pts * n_r_pts, n_b)
    out_img = Zygote.Buffer(img, p_h, p_w, n_c, n_r_pts * n_r_pts, n_b)
    @inbounds for i_b in 1:n_b
        i_pc = 1
        @inbounds for i_w in rand(1:n_w, n_r_pts), i_h in rand(1:n_h, n_r_pts)
            out_img[:, :, :, i_pc, i_b] =
                img[i_h:(i_h + p_h - 1), i_w:(i_w + p_w - 1), :, i_b]
            i_pc += one(i_pc)
        end
    end
    # out_img
    copy(out_img)
end

@inline function extract_patch_2(
    img::AbstractArray{T, 2},
    p_w::Integer,
    p_h::Integer,
)::AbstractArray{T, 3} where {T <: Real}
    n_h, n_w = size(img)
    n_h -= p_h + 1
    n_w -= p_w + 1
    # out_img = zeros(T, p_h, p_w, n_h * n_w)
    out_img = Zygote.Buffer(img, p_h, p_w, n_h * n_w)
    i_pc = 1
    @inbounds for i_w in 1:n_w, i_h in 1:n_h
        out_img[:, :, i_pc] = img[i_h:(i_h + p_h - 1), i_w:(i_w + p_w - 1)]
        i_pc += one(i_pc)
    end
    # out_img
    copy(out_img)
end

@inline function extract_patch_2_rand(
    img::AbstractArray{T, 2},
    p_w::Integer,
    p_h::Integer,
    n_pts::Integer,
)::AbstractArray{T, 3} where {T <: Real}
    n_r_pts = round(Int, sqrt(n_pts))
    n_h, n_w = size(img)
    n_h -= p_h + 1
    n_w -= p_w + 1
    # out_img = zeros(T, p_h, p_w, n_r_pts * n_r_pts)
    out_img = Zygote.Buffer(img, p_h, p_w, n_r_pts * n_r_pts)
    i_pc = 1
    @inbounds for i_w in rand(1:n_w, n_r_pts), i_h in rand(1:n_h, n_r_pts)
        out_img[:, :, i_pc] = img[i_h:(i_h + p_h - 1), i_w:(i_w + p_w - 1)]
        i_pc += one(i_pc)
    end
    # out_img
    copy(out_img)
end

# @inline function extract_patch_one(
#     icnf_f::Function,
#     img::AbstractArray{T, 4},
#     p_w::Integer,
#     p_h::Integer,
# ) where {T <: Real}
#     n_h, n_w, n_c, n_b = size(img)
#     n_h -= p_h + 1
#     n_w -= p_w + 1
#     mean(
#         ((i_b, i_w, i_h),) ->
#             icnf_f(vec(view(img, i_h:(i_h + p_h - 1), i_w:(i_w + p_w - 1), :, i_b))),
#         collect(product(1:n_b, rand(1:n_w, 50), rand(1:n_h, 50))),
#     )
# end

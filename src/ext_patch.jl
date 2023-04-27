function extract_patch(img::AbstractArray{T, 4}, p_w::Integer, p_h::Integer)::AbstractArray{T, 5} where T <: Real
    n_h, n_w, n_c, n_b = size(img)
    n_h -= p_h + 1
    n_w -= p_w + 1
    # out_img = zeros(T, p_h, p_w, n_c, n_h * n_w, n_b)
    out_img = Zygote.Buffer(img, p_h, p_w, n_c, n_h * n_w, n_b)
    for i_b ∈ 1:n_b
        i_pc = 1
        for i_w ∈ 1:n_w, i_h ∈ 1:n_h
            out_img[:, :, :, i_pc, i_b] = img[i_h:i_h+p_h-1, i_w:i_w+p_w-1, :, i_b]
            i_pc += one(i_pc)
        end
    end
    # out_img
    copy(out_img)
end

# ChainRulesCore.@non_differentiable extract_patch(img, p_w, p_h)

# function extract_patch_3(img::AbstractArray{T, 4}, p_w::Integer, p_h::Integer, n_pts::Integer)::AbstractArray{T, 4} where T <: Real
#     n_h, n_w, n_c, n_b = size(img)
#     n_h -= p_h + 1
#     n_w -= p_w + 1
#     pts = sample([view(img, i_h:i_h+p_h-1, i_w:i_w+p_w-1, :, i_b) for i_b ∈ 1:n_b, i_w ∈ 1:n_w, i_h ∈ 1:n_h], n_pts)
#     cat(pts...; dims=4)
# end

# function extract_patch_33(img::AbstractArray{T, 4}, p_w::Integer, p_h::Integer)::AbstractArray{T, 4} where T <: Real
#     n_h, n_w, n_c, n_b = size(img)
#     n_h -= p_h + 1
#     n_w -= p_w + 1
#     pts = [view(img, i_h:i_h+p_h-1, i_w:i_w+p_w-1, :, i_b) for i_b ∈ 1:n_b, i_w ∈ 1:n_w, i_h ∈ 1:n_h]
#     cat(pts...; dims=4)
# end

# function extract_patch_33(img::AbstractArray{T, 4}, p_w::Integer, p_h::Integer)::AbstractArray{T, 4} where T <: Real
#     n_h, n_w, n_c, n_b = size(img)
#     n_h -= p_h + 1
#     n_w -= p_w + 1
#     pts = [view(img, i_h:i_h+p_h-1, i_w:i_w+p_w-1, :, i_b) for i_b ∈ 1:n_b, i_w ∈ rand(1:n_w, 50), i_h ∈ rand(1:n_h, 50)]
#     cat(pts...; dims=4)
# end

# function extract_patch_50(icnf_f::Function, img::AbstractArray{T, 4}, p_w::Integer, p_h::Integer) where T <: Real
#     n_h, n_w, n_c, n_b = size(img)
#     n_h -= p_h + 1
#     n_w -= p_w + 1
#     mean(((i_b, i_w, i_h),) -> icnf_f(vec(view(img, i_h:i_h+p_h-1, i_w:i_w+p_w-1, :, i_b))), collect(product(1:n_b, rand(1:n_w, 50), rand(1:n_h, 50))))
# end

# function extract_patch_2(img::AbstractArray{T, 2}, p_s::Integer)::AbstractArray{T, 3} where T <: Real
#     n_h, n_w = size(img)
#     n_h -= p_s + 1
#     n_w -= p_s + 1
#     out_img = zeros(T, p_s, p_s, n_h * n_w)
#     i_pc = 1
#     for i_w ∈ 1:n_w, i_h ∈ 1:n_h
#         out_img[:, :, i_pc] .= img[i_h:i_h+p_s-1, i_w:i_w+p_s-1]
#         i_pc += one(i_pc)
#     end
#     out_img
# end

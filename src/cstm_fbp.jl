# @inline function cstm_fbp(obs_y)
#     image = rotr90(fbp(obs_y))
#     image = reverse(image; dims = 2)
#     image = image[76:(end - 76), 76:(end - 76)]
#     # image = reshape(standardize(UnitRangeTransform, vec(image)), size(image))
# end

@inline function stan_img(img)
    reshape(standardize(UnitRangeTransform, vec(img)), size(img))
end

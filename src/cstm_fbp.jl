# function cstm_fbp(obs_y)
#     image = rotr90(fbp(obs_y))
#     image = reverse(image; dims = 2)
#     image = image[76:(end - 76), 76:(end - 76)]
#     # image = reshape(standardize(UnitRangeTransform, vec(image)), size(image))
# end

function cstm_fbp_2(obs_y)
    obs_y2 = reshape(rotl90(obs_y), (1, 1, 1000, 513))
    rotr90(fbp_t(obs_y2)[1, 1, :, :])
end

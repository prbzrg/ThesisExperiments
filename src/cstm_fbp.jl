function cstm_fbp(obs_y)
    image = obs_y |> fbp |> rotr90
    image = reverse(image, dims = 2)
    image = image[76:end-76, 76:end-76]
    # image = reshape(standardize(UnitRangeTransform, vec(image)), size(image))
end

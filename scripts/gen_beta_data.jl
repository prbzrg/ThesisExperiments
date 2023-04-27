using DrWatson
@quickactivate "ThesisExperiments"

include(scriptsdir("import_pkgs.jl"))

function gen_data(nvars, n, data_dist=Beta(2, 4))
    r = rand(data_dist, nvars, n)
    # df = DataFrame(transpose(r), :auto)
end

function makesim(d::Dict)
    @unpack nvars, n, data_dist = d
    r = gen_data(nvars, n, data_dist)
    fulld = copy(d)
    fulld["r"] = r
    return fulld
end

allparams = Dict(
    "nvars" => [1, 2, 4, 8],
    "n" => [2^7, 2^10, 2^13],
    "data_dist" => Beta(2, 4),
)
dicts = dict_list(allparams)

for (i, d) in enumerate(dicts)
    f = makesim(d)
    @tagsave(datadir("synthetic-data", savename(d, "jld2")), f)
end

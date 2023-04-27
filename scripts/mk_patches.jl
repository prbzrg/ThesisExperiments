using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

allparams = Dict(
	"p_s" => [4, 6, 8],
)
dicts = dict_list(allparams)
dicts = convert.(Dict{String, Any}, dicts)

gt_fn = datadir("lodoct", "ground_truth_train_000.hdf5")

function makesim(d::Dict)
    @unpack p_s, = d

    fulld = copy(d)
    fulld["p_w"] = p_s
    fulld["p_h"] = p_s

    imgs = load(gt_fn)["data"]
    imgs = reshape(imgs, (362, 362, 1, 128))

    ptchs = extract_patch(imgs, p_s, p_s)
    fulld["ptchs"] = ptchs

    fulld
end

for (i, d) in enumerate(dicts)
    # f = makesim(d)
    produce_or_load(makesim, d, datadir("gen-ld-patch"))
    # @tagsave(datadir("simulations", savename(d, "jld2")), f)
end

df = collect_results(datadir("gen-ld-patch"))

using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

sel_t_img = 25

imgs = load(gt_test_fn)["data"]
imgs = imgs[:, :, sel_t_img]

obs_y = load(obs_test_fn)["data"]
obs_y = obs_y[:, :, sel_t_img]

# sinogram
resl_h, resl_w = size(obs_y)
resl = (resl_w * 2, resl_h * 2)
f = Figure(; resolution = resl)
ax1 = Makie.Axis(f[1, 1])
image!(ax1, rotr90(obs_y))

save(plotsdir("ex-imgs", "obs_img.svg"), f)
save(plotsdir("ex-imgs", "obs_img.png"), f)

# reconstruction

resl_h, resl_w = size(imgs)
resl = (resl_w * 2, resl_h * 2)
f = Figure(; resolution = resl)
ax1 = Makie.Axis(f[1, 1])
image!(ax1, rotr90(imgs))

save(plotsdir("ex-imgs", "gt_img.svg"), f)
save(plotsdir("ex-imgs", "gt_img.png"), f)

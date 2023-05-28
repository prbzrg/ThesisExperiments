using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

df = collect_results(datadir("patchnr-sims"))

for i in 1:size(df, 1)
    save(
        plotsdir("patchnr-sims-imgs", "$(i)_result.png"),
        convert.(Gray, stan_img(df[i, "res_img"])),
    )
    save(
        plotsdir("patchnr-sims-imgs", "$(i)_fbp.png"),
        convert.(Gray, stan_img(df[i, "fbp_img"])),
    )
    save(
        plotsdir("patchnr-sims-imgs", "$(i)_goal.png"),
        convert.(Gray, stan_img(df[i, "gt_x"])),
    )
end

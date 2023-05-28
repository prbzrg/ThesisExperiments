using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

df_o = collect_results(datadir("patchnr-sims"))
df = copy(df_o)
df[!, "res_img"] .= missing
df[!, "fbp_img"] .= missing
df[!, "gt_x"] .= missing
df[!, "time_obj"] .= missing

CSV.write(plotsdir("patchnr-sims-csv", "patchnr-sims.csv"), df)

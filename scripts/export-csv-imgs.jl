using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

df = collect_results(datadir("patchnr-sims"))

res_sz = (362 * 2 * 3, 362 * 2)
r_dgt = 4

for i in axes(df, 1)
    f = Figure(; resolution = res_sz)
    ax1 = Makie.Axis(
        f[1, 1];
        title = "Filtered Back-projection",
        subtitle = "PSNR: $(round(df[i, "fbp_a_psnr"]; digits=r_dgt)), SSIM: $(round(df[i, "fbp_a_ssim"]; digits=r_dgt)), MSSSIM: $(round(df[i, "fbp_a_msssim"]; digits=r_dgt))",
    )
    image!(ax1, rotr90(df[i, "fbp_img"]))
    ax2 = Makie.Axis(f[1, 2]; title = "Ground Truth")
    image!(ax2, rotr90(df[i, "gt_x"]))
    ax3 = Makie.Axis(
        f[1, 3];
        title = "Result",
        subtitle = "PSNR: $(round(df[i, "a_psnr"]; digits=r_dgt)), SSIM: $(round(df[i, "a_ssim"]; digits=r_dgt)), MSSSIM: $(round(df[i, "a_msssim"]; digits=r_dgt))",
    )
    image!(ax3, rotr90(df[i, "res_img"]))
    sel_a = df[i, "sel_a"]
    save(plotsdir("patchnr-sims-imgs", "$(i)_plot_($sel_a).svg"), f)
    save(plotsdir("patchnr-sims-imgs", "$(i)_plot_($sel_a).png"), f)
end

df2c = copy(df)

df2c[!, "res_img"] .= missing
df2c[!, "fbp_img"] .= missing
df2c[!, "gt_x"] .= missing
df2c[!, "time_obj"] .= missing
df2c[!, "sel_pol"] .= map(x -> isnothing(x) ? missing : x, df2c[!, "sel_pol"])
df2c[!, "have_bias"] .= map(x -> isnothing(x) ? missing : x, df2c[!, "have_bias"])

CSV.write(plotsdir("patchnr-sims-csv", "patchnr-sims.csv"), df2c)

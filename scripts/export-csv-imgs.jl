using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

df = collect_results(datadir("postp-sims"))

res_sz = (362 * 2 * 3, 362 * 2 * 2)
r_dgt = 4
nb = 512

for i in axes(df, 1)
    f = Figure(; resolution = res_sz)

    ax1 = Makie.Axis(
        f[1, 1];
        title = "Filtered Back-projection",
        subtitle = "PSNR: $(round(df[i, "post_fbp_a_psnr"]; digits=r_dgt)), SSIM: $(round(df[i, "post_fbp_a_ssim"]; digits=r_dgt)), MSSSIM: $(round(df[i, "post_fbp_a_msssim"]; digits=r_dgt))",
    )
    image!(ax1, rotr90(df[i, "post_fbp_img"]))

    ax21 = Makie.Axis(f[2, 1]; title = "Histogram of Filtered Back-projection")
    hist!(ax21, convert.(Float32, vec(rotr90(df[i, "post_fbp_img"]))); bins = nb)

    ax2 = Makie.Axis(f[1, 2]; title = "Ground Truth")
    image!(ax2, rotr90(df[i, "post_gt_x"]))

    ax22 = Makie.Axis(f[2, 2]; title = "Histogram of Ground Truth")
    hist!(ax22, convert.(Float32, vec(rotr90(df[i, "post_gt_x"]))); bins = nb)

    ax3 = Makie.Axis(
        f[1, 3];
        title = "Result",
        subtitle = "PSNR: $(round(df[i, "post_a_psnr"]; digits=r_dgt)), SSIM: $(round(df[i, "post_a_ssim"]; digits=r_dgt)), MSSSIM: $(round(df[i, "post_a_msssim"]; digits=r_dgt))",
    )
    image!(ax3, rotr90(df[i, "post_res_img"]))

    ax23 = Makie.Axis(f[2, 3]; title = "Histogram of Result")
    hist!(ax23, convert.(Float32, vec(rotr90(df[i, "post_res_img"]))); bins = nb)

    sel_a = df[i, "sel_a"]
    save(plotsdir("patchnr-sims-imgs", "$(i)_plot_$(sel_a).svg"), f)
    save(plotsdir("patchnr-sims-imgs", "$(i)_plot_$(sel_a).png"), f)
end

df2c = copy(df)

df2c[!, "gt_x"] .= missing
df2c[!, "fbp_img"] .= missing
df2c[!, "res_img"] .= missing
df2c[!, "post_gt_x"] .= missing
df2c[!, "post_fbp_img"] .= missing
df2c[!, "post_res_img"] .= missing

df2c[!, "time_obj"] .= missing

df2c[!, "sel_pol"] .= map(x -> isnothing(x) ? missing : x, df2c[!, "sel_pol"])
df2c[!, "have_bias"] .= map(x -> isnothing(x) ? missing : x, df2c[!, "have_bias"])

CSV.write(plotsdir("patchnr-sims-csv", "patchnr-sims.csv"), df2c)

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
    save(plotsdir("patchnr-sims-imgs", "$(i)_plot.svg"), f)
    save(plotsdir("patchnr-sims-imgs", "$(i)_plot.png"), f)
end

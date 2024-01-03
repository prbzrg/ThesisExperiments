using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

df = collect_results(datadir("postp-sims"))

res_sz = (362 * 2 * 4, 362 * 2 * 2)
r_dgt = 4
nb = 512

for i in 1:26
    f = Figure(; size = res_sz)

    ax11 = Makie.Axis(f[1, 1]; title = "Ground Truth")
    image!(ax11, rotr90(df[i, "post_gt_x"]))

    ax21 = Makie.Axis(f[2, 1]; title = "Histogram of Ground Truth")
    hist!(ax21, convert.(Float32, vec(rotr90(df[i, "post_gt_x"]))); bins = nb)

    ax12 = Makie.Axis(
        f[1, 2];
        title = "Filtered Back-projection",
        subtitle = "PSNR: $(round(df[i,"post_fbp_a_psnr"]; digits=r_dgt)), SSIM: $(round(df[i,"post_fbp_a_ssim"]; digits=r_dgt)), MSSSIM: $(round(df[i,"post_fbp_a_msssim"]; digits=r_dgt))",
    )
    image!(ax12, rotr90(df[i, "post_fbp_img"]))

    ax22 = Makie.Axis(f[2, 2]; title = "Histogram of Filtered Back-projection")
    hist!(ax22, convert.(Float32, vec(rotr90(df[i, "post_fbp_img"]))); bins = nb)

    ax13 = Makie.Axis(
        f[1, 3];
        title = "PatchRegICNF-1",
        subtitle = "PSNR: $(round(df[26+i,"post_a_psnr"]; digits=r_dgt)), SSIM: $(round(df[26+i,"post_a_ssim"]; digits=r_dgt)), MSSSIM: $(round(df[26+i,"post_a_msssim"]; digits=r_dgt))",
    )
    image!(ax13, rotr90(df[26 + i, "post_res_img"]))

    ax23 = Makie.Axis(f[2, 3]; title = "Histogram of PatchRegICNF-1")
    hist!(ax23, convert.(Float32, vec(rotr90(df[26 + i, "post_res_img"]))); bins = nb)

    ax14 = Makie.Axis(
        f[1, 4];
        title = "PatchRegICNF-13",
        subtitle = "PSNR: $(round(df[i,"post_a_psnr"]; digits=r_dgt)), SSIM: $(round(df[i,"post_a_ssim"]; digits=r_dgt)), MSSSIM: $(round(df[i,"post_a_msssim"]; digits=r_dgt))",
    )
    image!(ax14, rotr90(df[i, "post_res_img"]))

    ax24 = Makie.Axis(f[2, 4]; title = "Histogram of PatchRegICNF-13")
    hist!(ax24, convert.(Float32, vec(rotr90(df[i, "post_res_img"]))); bins = nb)

    save(plotsdir("results-join", "thesis-results-$i.svg"), f)
    save(plotsdir("results-join", "thesis-results-$i.png"), f)
end

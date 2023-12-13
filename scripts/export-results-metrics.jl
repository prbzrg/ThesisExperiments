using DrWatson
@quickactivate "ThesisExperiments" # <- project name

include(scriptsdir("import_pkgs.jl"))

df = collect_results(datadir("export-imgs-sims"))

@show round(mean(df[begin:26, "post_fbp_a_psnr"]); digits = 2)
@show round(std(df[begin:26, "post_fbp_a_psnr"]); digits = 2)
@show round(mean(df[begin:26, "post_fbp_a_ssim"]); digits = 3)
@show round(std(df[begin:26, "post_fbp_a_ssim"]); digits = 3)
@show round(mean(df[begin:26, "post_fbp_a_msssim"]); digits = 3)
@show round(std(df[begin:26, "post_fbp_a_msssim"]); digits = 3)

@show round(mean(df[begin:26, "post_a_psnr"]); digits = 2)
@show round(std(df[begin:26, "post_a_psnr"]); digits = 2)
@show round(mean(df[begin:26, "post_a_ssim"]); digits = 3)
@show round(std(df[begin:26, "post_a_ssim"]); digits = 3)
@show round(mean(df[begin:26, "post_a_msssim"]); digits = 3)
@show round(std(df[begin:26, "post_a_msssim"]); digits = 3)

@show round(mean(df[27:end, "post_fbp_a_psnr"]); digits = 2)
@show round(std(df[27:end, "post_fbp_a_psnr"]); digits = 2)
@show round(mean(df[27:end, "post_fbp_a_ssim"]); digits = 3)
@show round(std(df[27:end, "post_fbp_a_ssim"]); digits = 3)
@show round(mean(df[27:end, "post_fbp_a_msssim"]); digits = 3)
@show round(std(df[27:end, "post_fbp_a_msssim"]); digits = 3)

@show round(mean(df[27:end, "post_a_psnr"]); digits = 2)
@show round(std(df[27:end, "post_a_psnr"]); digits = 2)
@show round(mean(df[27:end, "post_a_ssim"]); digits = 3)
@show round(std(df[27:end, "post_a_ssim"]); digits = 3)
@show round(mean(df[27:end, "post_a_msssim"]); digits = 3)
@show round(std(df[27:end, "post_a_msssim"]); digits = 3)

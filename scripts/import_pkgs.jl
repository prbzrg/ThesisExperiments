ENV["PYTHON"] = raw"C:\Program Files\Python39\python.exe"

using Pkg

Pkg.precompile(; strict = true)

using AbstractDifferentiation,
    ADTypes,
    Base.Iterators,
    Base.Threads,
    Colors,
    ComputationalResources,
    ContinuousNormalizingFlows,
    CSV,
    CUDA,
    cuDNN,
    CairoMakie,
    DataFrames,
    Dates,
    DifferentialEquations,
    Distances,
    Distributions,
    Documenter,
    DrWatson,
    Enzyme,
    Flux,
    ForwardDiff,
    ImageQualityIndexes,
    Images,
    ImageTransformations,
    JLD2,
    Logging,
    Lux,
    LuxCUDA,
    MLDatasets,
    MLJBase,
    MLUtils,
    Makie,
    Optim,
    Optimisers,
    Optimization,
    OptimizationOptimisers,
    OptimizationOptimJL,
    OrdinaryDiffEq,
    ProgressMeter,
    PyCall,
    Random,
    ReverseDiff,
    SciMLSensitivity,
    Sinograms,
    Sixel,
    Statistics,
    StatsBase,
    Tracker,
    Zygote

const cdev = cpu_device()
const gdev = gpu_device()
# const debuglogger = Logging.ConsoleLogger(Logging.Debug)
# Logging.global_logger(debuglogger)

include(srcdir("ext_patch.jl"))
include(srcdir("radon_transform.jl"))
include(srcdir("patchnr.jl"))
include(srcdir("cstm_fbp.jl"))
include(srcdir("train_loop.jl"))
include(srcdir("frst_prt.jl"))
# include(srcdir("mrcnf.jl"))

# defaults
const sol_kwargs = Dict(
    :alg_hints => [:nonstiff, :memorybound],
    :dense => false,
    :save_everystep => false,
    :save_on => false,
    :calck => false,
    :alias_u0 => true,
    :verbose => true,
    :merge_callbacks => true,
    :wrap => Val(true),
    :alg => BS3(),
    # :alg => BS3(; thread = OrdinaryDiffEq.True()),
    # :alg => BS5(; thread = OrdinaryDiffEq.True()),
    # :alg => Tsit5(; thread = OrdinaryDiffEq.True()),
    # :alg => Vern6(; thread = OrdinaryDiffEq.True()),
    # :alg => Vern9(; thread = OrdinaryDiffEq.True()),
    :sensealg => BacksolveAdjoint(;
        autodiff = true,
        autojacvec = ZygoteVJP(),
        checkpointing = true,
    ),
    # :sensealg => InterpolatingAdjoint(;
    #     autodiff = true,
    #     autojacvec = ZygoteVJP(),
    #     checkpointing = true,
    # ),
    # :sensealg => QuadratureAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
    :reltol => 1.0f-2 + eps(1.0f-2),
    # :reltol => eps(one(Float32)),
    :abstol => eps(one(Float32)),
    :maxiters => typemax(Int32),
)
const optimizers = Any[Optimisers.Lion(),]

if !isempty(ARGS)
    const use_gpu_nn_train = ARGS[1] == "train_gpu"
    const use_gpu_nn_test = ARGS[2] == "test_gpu"
else
    @warn "please have `train_cpu test_cpu` as arguments of julia"
    const use_gpu_nn_train = false
    const use_gpu_nn_test = false
end

@show (use_gpu_nn_train, use_gpu_nn_test)

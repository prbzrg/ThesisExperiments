ENV["PYTHON"] = raw"C:\Program Files\Python37\python.exe"

using Pkg

Pkg.precompile()

using AbstractDifferentiation,
    Base.Iterators,
    Base.Threads,
    ChainRules,
    ChainRulesCore,
    Colors,
    ComputationalResources,
    ContinuousNormalizingFlows,
    CSV,
    CUDA,
    DataFrames,
    Dates,
    DifferentialEquations,
    Distances,
    Distributions,
    Documenter,
    DrWatson,
    Enzyme,
    FileIO,
    FiniteDiff,
    Flux,
    ForwardDiff,
    ImageInTerminal,
    ImageIO,
    ImageQualityIndexes,
    Images,
    ImageShow,
    ImageTransformations,
    ImageView,
    JLD2,
    Logging,
    Lux,
    LuxCUDA,
    MLDatasets,
    MLJBase,
    MLUtils,
    Optim,
    Optimisers,
    Optimization,
    OptimizationOptimisers,
    OptimizationOptimJL,
    Plots,
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

# debuglogger = Logging.ConsoleLogger(Logging.Debug)
# Logging.global_logger(debuglogger)

include(srcdir("ext_patch.jl"))
include(srcdir("radon_transform.jl"))
include(srcdir("patchnr.jl"))
include(srcdir("cstm_fbp.jl"))
include(srcdir("train_loop.jl"))
include(srcdir("frst_prt.jl"))
# include(srcdir("mrcnf.jl"))

# defaults
sol_kwargs = Dict(
    :alg_hints => [:nonstiff, :memorybound],
    :dense => false,
    :save_everystep => false,
    :save_on => false,
    :calck => false,
    :alg => BS3(; thread = OrdinaryDiffEq.True()),
    # :alg => BS5(; thread = OrdinaryDiffEq.True()),
    # :alg => Tsit5(; thread = OrdinaryDiffEq.True()),
    # :alg => Vern6(; thread = OrdinaryDiffEq.True()),
    # :alg => Vern9(; thread = OrdinaryDiffEq.True()),
    :sensealg => BacksolveAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
    # :sensealg => InterpolatingAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
    # :sensealg => QuadratureAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
    :abstol => eps(one(Float32)),
    :reltol => 1.0f-2 + eps(1.0f-2),
    # :reltol => eps(one(Float32)),
)
optimizers = Any[Optimisers.Lion(),]

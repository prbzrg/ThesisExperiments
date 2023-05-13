using ICNF
using CUDA
using Colors,
    FileIO,
    ImageIO,
    ImageInTerminal,
    ImageQualityIndexes,
    ImageShow,
    ImageTransformations,
    ImageView,
    Images,
    Sixel
using ComputationalResources, DataFrames
using Dates, Logging, Random, Statistics
using DifferentialEquations, SciMLSensitivity
using Distances, Distributions, StatsBase
using Documenter, DrWatson, JLD2, ProgressMeter
using FiniteDiff, ForwardDiff, ReverseDiff, Zygote, Enzyme, Tracker, AbstractDifferentiation
using ChainRules, ChainRulesCore
using Flux, Lux, LuxCUDA
using Optim, Optimisers, Optimization, OptimizationOptimJL, OptimizationOptimisers
using MLDatasets, MLJBase, MLUtils, Plots
using ImageReconstruction, Sinograms
using Base.Threads, Base.Iterators

# debuglogger = Logging.ConsoleLogger(Logging.Debug)
# Logging.global_logger(debuglogger)

include(srcdir("ext_patch.jl"))
include(srcdir("radon_transform.jl"))
include(srcdir("patchnr.jl"))
include(srcdir("cstm_fbp.jl"))
include(srcdir("train_loop.jl"))
# include(srcdir("mrcnf.jl"))

#defaults
sol_kwargs = Dict(
    # :alg_hints => [:nonstiff],
    # :dense => true,
    # :adaptive => true,
    :alg => BS3(; thread = OrdinaryDiffEq.True()),
    # :alg => Vern9(; thread = OrdinaryDiffEq.True()),
    :sensealg => BacksolveAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
    # :sensealg => InterpolatingAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
    # :sensealg => QuadratureAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
    # :abstol => eps(Float32),
    # :reltol => eps(Float32),
    # :maxiters => typemax(Int),
)
optimizers = Any[Optimisers.Lion(),]

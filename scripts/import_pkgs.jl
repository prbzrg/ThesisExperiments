using Pkg

Pkg.precompile(; strict = true)

# using MKL

using Base.Threads,
    ADTypes,
    BenchmarkTools,
    CairoMakie,
    Colors,
    ComponentArrays,
    ConcreteStructs,
    Conda,
    ContinuousNormalizingFlows,
    CSV,
    DataFrames,
    Dates,
    DifferentiationInterface,
    Distances,
    Distributions,
    Documenter,
    DrWatson,
    HDF5,
    ImageFiltering,
    ImageQualityIndexes,
    Images,
    JLD2,
    LinearAlgebra,
    Logging,
    Lux,
    Makie,
    MLJBase,
    MLUtils,
    Optim,
    Optimisers,
    Optimization,
    OptimizationOptimisers,
    OptimizationOptimJL,
    OrdinaryDiffEq,
    ProgressLogging,
    PyCall,
    Random,
    SciMLSensitivity,
    Statistics,
    StatsBase,
    TerminalLoggers,
    UnPack,
    Zygote

# using AbstractDifferentiation, Enzyme, ForwardDiff, ReverseDiff, Tracker
# using Base.Iterators, ImageGeoms, ImageTransformations, Sinograms
# using SimpleChains
# using MLDatasets
# using GFlops

global_logger(TerminalLogger())

# const debuglogger = TerminalLoggers.TerminalLogger(stderr, Logging.Debug)
# Logging.global_logger(debuglogger)

# const nthd = nthreads(:default)
# if nthd > 1 && nthd > BLAS.get_num_threads()
#     BLAS.set_num_threads(nthd)
# end

# Enzyme.API.runtimeActivity!(true)

const use_thrds = false
const n_data_b = 128
const cdev = cpu_device()
const eps_sq = Float32[
    eps(one(Float32)), # 1.1920929f-7
    sqrt(eps(one(Float32))), # 0.00034526698
    sqrt(sqrt(eps(one(Float32)))), # 0.01858136
    sqrt(sqrt(sqrt(eps(one(Float32))))), # 0.13631347
    sqrt(sqrt(sqrt(sqrt(eps(one(Float32)))))), # 0.36920655f0
]

include(srcdir("ext_patch.jl"))
include(srcdir("radon_transform.jl"))
include(srcdir("patchnr.jl"))
include(srcdir("cstm_fbp.jl"))
include(srcdir("train_loop.jl"))
include(srcdir("frst_prt.jl"))
# include(srcdir("mrcnf.jl"))

# defaults
const sol_kwargs_base = (
    save_everystep = false,
    # alg = VCABM3(),
    # alg = Vern6(; thread = OrdinaryDiffEq.True()),
    # alg = BS5(; thread = OrdinaryDiffEq.True()),
    # alg = BS3(; thread = OrdinaryDiffEq.True()),
    # alg = Tsit5(; thread = OrdinaryDiffEq.True()),
    alg = VCABM(),
    # sensealg = GaussAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
    sensealg = BacksolveAdjoint(;
        autodiff = true,
        autojacvec = ZygoteVJP(),
        checkpointing = true,
    ),
    # sensealg = QuadratureAdjoint(; autodiff = true, autojacvec = ZygoteVJP()),
    # sensealg = InterpolatingAdjoint(;
    #     autodiff = true,
    #     autojacvec = ZygoteVJP(),
    #     checkpointing = true,
    # ),
    reltol = eps_sq[2],
    abstol = eps_sq[1],
    maxiters = typemax(Int32),
    alg_hints = [:nonstiff, :memorybound],
    dense = false,
    save_on = false,
    calck = false,
    alias_u0 = true,
    verbose = true,
    merge_callbacks = true,
    wrap = Val(true),
)
const optimizers = (Lion(),)

if !isempty(ARGS) && length(ARGS) >= 2
    const use_gpu_nn_train = ARGS[1] == "train_gpu"
    const use_gpu_nn_test = ARGS[2] == "test_gpu"
else
    @warn "please have `train_cpu test_cpu` as arguments of julia"
    const use_gpu_nn_train = false
    const use_gpu_nn_test = false
end
if use_gpu_nn_train || use_gpu_nn_test
    using ComputationalResources, CUDA, cuDNN, LuxCUDA
    const gdev = gpu_device()
else
    const gdev = cpu_device()
end

@info (use_gpu_nn_train, use_gpu_nn_test)

const gt_train_fn = datadir("lodoct", "data", "ground_truth_train_000.hdf5")
const gt_test_fn = datadir("lodoct", "data", "ground_truth_test_000.hdf5")
const obs_train_fn = datadir("lodoct", "data", "observation_train_000.hdf5")
const obs_test_fn = datadir("lodoct", "data", "observation_test_000.hdf5")

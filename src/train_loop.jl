# @inline function _loss(ps, ptchnr, obs_y)
#     recn_loss(ptchnr, ps, obs_y)
# end

@inline function _loss(ps, ptchnr, obs_y)
    pt1 = recn_loss_pt1(ptchnr, ps, obs_y)
    pt2 = recn_loss_pt2(ptchnr, ps, obs_y)
    pt1 + pt2
end

@inline function _loss_gd_o(ps, ptchnr, obs_y)
    pt1 = recn_loss_pt1_grad(ptchnr, ps, obs_y)
    pt2 = recn_loss_pt2_grad(ptchnr, ps, obs_y)
    pt1 + pt2
end

@inline function _loss_gd_i(G, ps, ptchnr, obs_y)
    fill!(G, zero(eltype(ps)))
    G .+= recn_loss_pt1_grad(ptchnr, ps, obs_y)
    G .+= recn_loss_pt2_grad(ptchnr, ps, obs_y)
    nothing
end

@inline function train_loop(ps, ptchnr, obs_y, opt, n_iter)
    # optfunc = OptimizationFunction(_loss2, AutoZygote())
    optfunc = OptimizationFunction(
        let ptchnr = ptchnr, obs_y = obs_y
            (ps, θ) -> _loss(ps, ptchnr, obs_y)
        end;
        grad = let ptchnr = ptchnr, obs_y = obs_y
            (ps_i, ps, θ) -> _loss_gd_i(ps_i, ps, ptchnr, obs_y)
        end,
    )
    optprob = OptimizationProblem(optfunc, ps)
    tst_one = @timed res = solve(optprob, opt; maxiters = n_iter, progress = true)
    @info(
        "Fitting - Overall",
        "elapsed time (seconds)" = tst_one.time,
        "garbage collection time (seconds)" = tst_one.gctime,
        "allocated (bytes)" = tst_one.bytes,
        "final loss value" = res.objective,
    )
    res.u, tst_one
end

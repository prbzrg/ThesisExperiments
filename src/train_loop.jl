# @inline function _loss(ps, ptchnr, obs_y)
#     recn_loss(ptchnr, ps, obs_y)
# end

@inline function _loss(ps, ptchnr, obs_y)
    pt1 = recn_loss_pt1(ptchnr, ps, obs_y)
    pt2 = recn_loss_pt2(ptchnr, ps, obs_y)
    pt1 + pt2
end

@inline function _loss_gd(G, ps, ptchnr, obs_y)
    pt1 = recn_loss_pt1_grad(ptchnr, ps, obs_y)
    pt2 = recn_loss_pt2_grad(ptchnr, ps, obs_y)

    G .= pt1 + pt2
    nothing
end

@inline function train_loop(ps, ptchnr, obs_y, opt, n_iter)
    opt_state = Optimisers.setup(opt, ps)
    G = copy(ps)
    # G = zeros(eltype(ps), length(ps))
    prgr = Progress(n_iter; desc = "Min for CT: ", showspeed = true)
    for i in 1:n_iter
        _loss_gd(G, ps, ptchnr, obs_y)
        opt_state, ps = Optimisers.update!(opt_state, ps, G)
        lv = _loss(ps, ptchnr, obs_y)
        ProgressMeter.next!(
            prgr;
            showvalues = [(:loss_value, lv), (:last_update, Dates.now())],
        )
    end
    ProgressMeter.finish!(prgr)
    ps
end

@inline function train_loop_optpkg(ps, ptchnr, obs_y, opt, n_iter)
    prgr = Progress(n_iter; desc = "Min for CT: ", showspeed = true)

    @inline function _callback(ps, l)
        ProgressMeter.next!(
            prgr;
            showvalues = [(:loss_value, l), (:last_update, Dates.now())],
        )
        false
    end

    @inline function _loss2(ps, θ)
        _loss(ps, ptchnr, obs_y)
    end

    @inline function _loss_gd2(ps_i, ps, θ)
        _loss_gd(ps_i, ps, ptchnr, obs_y)
    end

    # optfunc = OptimizationFunction(_loss2, AutoTracker())
    # optfunc = OptimizationFunction(_loss2, AutoForwardDiff())
    # optfunc = OptimizationFunction(_loss2, AutoReverseDiff())
    # optfunc = OptimizationFunction(_loss2, AutoZygote())
    optfunc = OptimizationFunction(_loss2; grad = _loss_gd2)
    # optfunc = OptimizationFunction(_loss2)
    optprob = OptimizationProblem(optfunc, ps)
    res = solve(optprob, opt; callback = _callback, maxiters = n_iter)
    ProgressMeter.finish!(prgr)
    res.u
end

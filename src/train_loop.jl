function _loss(ps, ptchnr, obs_y)
    pt1 = recn_loss_pt1(ptchnr, ps, obs_y)
    pt2 = recn_loss_pt2(ptchnr, ps, obs_y)
    pt1 + pt2
end

function _loss_gd(G, ps, ptchnr, obs_y)
    pt1 = ReverseDiff.gradient(x -> recn_loss_pt1(ptchnr, x, obs_y), ps)
    # pt1 = ForwardDiff.gradient(x -> recn_loss_pt1(ptchnr, x, obs_y), ps)

    # pt2 = ForwardDiff.gradient(x -> recn_loss_pt2(ptchnr, x, obs_y), ps)
    pt2 = only(Zygote.gradient(x -> recn_loss_pt2(ptchnr, x, obs_y), ps))
    G .= pt1 + pt2
    nothing
end

function train_loop(ps, ptchnr, obs_y, opt, n_iter)
    opt_state = Optimisers.setup(opt, ps)
    G = zeros(eltype(ps), length(ps))
    prgr = Progress(n_iter; dt = eps(Float32), desc = "Min for CT: ", showspeed = true)
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

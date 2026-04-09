@recipe function f(h::H2Matrix)
    legend --> false
    grid --> false
    aspect_ratio --> :equal
    yflip := true
    seriestype := :shape
    linecolor --> :gray80
    linewidth --> 0.3
    m, n = size(h)
    xlims --> (1, n)
    ylims --> (1, m)

    lvs = leaves(h)
    for leaf in lvs
        rc = leaf.row_basis.cluster
        cc = leaf.col_basis.cluster
        ir = index_range(rc)
        jr = index_range(cc)
        y1 = ir.start
        y2 = ir.stop
        x1 = jr.start
        x2 = jr.stop
        xs = [x1, x2, x2, x1, x1]
        ys = [y1, y1, y2, y2, y1]
        if leaf.uniform !== nothing
            # Admissible block — dodger blue, alpha logic same as HMatrices
            m_blk, n_blk = size(leaf.uniform)
            r = size(leaf.uniform.S, 1)
            alpha = m_blk * n_blk / (r * (m_blk + n_blk))
            @series begin
                fillcolor --> "#1E90FF"   # dodger blue
                seriesalpha --> 1 / alpha
                xs, ys
            end
        elseif leaf.dense !== nothing
            # Dense block — goldenrod, same alpha as HMatrices dense
            @series begin
                fillcolor --> "#FDB515"   # goldenrod
                seriesalpha --> 0.8
                xs, ys
            end
        end
    end
end

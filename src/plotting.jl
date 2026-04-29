function _rank_color_hex(rank::Int, minrank::Int, maxrank::Int)
    if maxrank <= minrank
        t = 0.0
    else
        t = (rank - minrank) / (maxrank - minrank)
    end
    t = clamp(t, 0.0, 1.0)

    # Low rank: dark blue. High rank: bright gold.
    lo = (0x0b, 0x2c, 0x6b)
    hi = (0xff, 0xc4, 0x28)
    r = round(Int, (1 - t) * lo[1] + t * hi[1])
    g = round(Int, (1 - t) * lo[2] + t * hi[2])
    b = round(Int, (1 - t) * lo[3] + t * hi[3])
    return "#" * string(r, base=16, pad=2) *
                 string(g, base=16, pad=2) *
                 string(b, base=16, pad=2)
end

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
    ranks = [min(size(leaf.uniform.S)...) for leaf in lvs if leaf.uniform !== nothing]
    minrank = isempty(ranks) ? 1 : minimum(ranks)
    maxrank = isempty(ranks) ? 1 : maximum(ranks)

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
            r = min(size(leaf.uniform.S)...)
            @series begin
                fillcolor --> _rank_color_hex(r, minrank, maxrank)
                seriesalpha --> 0.95
                xs, ys
            end
        elseif leaf.dense !== nothing
            @series begin
                fillcolor --> "#d95f02"
                seriesalpha --> 0.85
                xs, ys
            end
        end
    end
end

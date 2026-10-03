# Saturated bases can be represented in physical coordinates without a
# truncation. Their effects move into couplings and unsaturated-parent transfers.
# A pass-through node uses the concatenated coefficients of its children as its
# coefficients: its transfer from the children is folded into its couplings and
# its own transfer to the parent, which is exact up to rounding.
struct _CompactBasisNode
    V::Matrix{Float64}
    E::Matrix{Float64}
    coeff::UnitRange{Int}
    indices::UnitRange{Int}
    children::Vector{Int}
    identity::Bool
    passthrough::Bool
end
_CompactBasisNode(V,E,coeff,indices,children,identity)=_CompactBasisNode(V,E,coeff,indices,children,identity,false)
# `weights[cb]` is the summed coefficient width of the coupling partners of
# `cb`. With weights, an internal node becomes pass-through when this lowers
# the stored numbers: (D-k)(k_parent+W) < D*k for D child coefficients.
function _compact_basis(root,weights=nothing)
    originals=typeof(root)[]
    function visit(cb)
        push!(originals,cb)
        foreach(visit,cb.children)
    end
    visit(root)
    ids=IdDict(cb=>i for (i,cb) in enumerate(originals))
    saturated=IdDict{typeof(root),Matrix{Float64}}()
    for cb in originals
        if cb.k>=length(cb) && cb.k>0
            saturated[cb]=_full_basis(cb)
        elseif weights!==nothing && isleaf(cb) && cb.k>0
            # The same storage rule with D=|t|: physical coordinates for a
            # nearly saturated leaf whose few couplings do not amortize V.
            m=length(cb);kp=isroot(cb) || haskey(saturated,cb.parent) ? 0 : cb.parent.k
            (m-cb.k)*(kp+get(weights,cb,0))<m*cb.k && (saturated[cb]=cb.V)
        end
    end
    # Expansion of a node's own k coefficients into the coordinates it uses.
    expand=IdDict{typeof(root),Matrix{Float64}}(saturated)
    passthrough=IdDict{typeof(root),Int}()
    if weights!==nothing
        dims=IdDict{typeof(root),Int}()
        for cb in Iterators.reverse(originals)
            if haskey(saturated,cb)
                dims[cb]=length(cb)
            elseif isleaf(cb) || cb.k==0
                dims[cb]=cb.k
            else
                active=[c for c in cb.children if dims[c]>0]
                D=sum((dims[c] for c in active);init=0)
                kp=isroot(cb) || haskey(saturated,cb.parent) ? 0 : cb.parent.k
                W=get(weights,cb,0)
                if D>0 && (D-cb.k)*(kp+W)<D*cb.k
                    expand[cb]=reduce(vcat,[haskey(expand,c) ? expand[c]*c.E : c.E for c in active])
                    passthrough[cb]=D;dims[cb]=D
                else
                    dims[cb]=cb.k
                end
            end
        end
    end
    nodes=_CompactBasisNode[];offset=first(index_range(root.cluster))-1;total=0
    for cb in originals
        identity=haskey(saturated,cb);pass=haskey(passthrough,cb)
        k=identity ? length(cb) : pass ? passthrough[cb] : cb.k
        cr=total+1:total+k;total+=k
        # Transfers to a saturated parent are unused: that parent's basis
        # expands directly in physical coordinates rather than through children.
        # A pass-through parent copies child coefficients instead.
        E=if isroot(cb) || haskey(saturated,cb.parent) || haskey(passthrough,cb.parent)
            zeros(Float64,0,0)
        elseif haskey(expand,cb)
            expand[cb]*cb.E
        else
            cb.E
        end
        V=identity || pass ? zeros(Float64,0,0) : cb.V
        push!(nodes,_CompactBasisNode(V,E,cr,index_range(cb.cluster).-offset,
            [ids[c] for c in cb.children],identity,pass))
    end
    nodes,ids,expand,zeros(total)
end
"""
    H2CompactMatvecPlan(h2; coupling_rtol=nothing, coupling_scale=:block, passthrough=false)

Compact reusable matvec representation of the stored H2 operator. Saturated
cluster bases (rank at least cluster size) are replaced by implicit identity
bases; their numerical action is moved to couplings and parent transfers.
This is an algebraic representation change with only floating-point rounding,
not a tolerance relaxation. Works for nonorthogonal and overcomplete bases.

`passthrough=true` additionally lets an internal basis node use its children's
concatenated coefficients when that stores fewer numbers: its transfer
matrices are folded into its couplings and into its own transfer to the
parent. This is also exact up to rounding and mainly removes nearly square
transfer matrices of weakly compressing upper levels that serve few couplings.

An optional `coupling_rtol` additionally enables local SVD coupling truncation,
which is an approximation and requires separate application validation.
With `coupling_scale=:block` (default) singular values below `coupling_rtol`
times the coupling's own largest singular value are discarded. With
`coupling_scale=:global` the threshold is `coupling_rtol` times the largest
spectral norm over all stored blocks (couplings and near field), so weak
blocks are not resolved to a tighter absolute accuracy than strong ones. A coupling is factorized only
when its factors use less storage than the coupling itself.
The plan retains numerical data but not the source operator or unused bases.
Its adjoint applies the same stored approximation. Use one plan per concurrent
worker and do not mutate shared source data while a plan is in use.
"""
struct H2CompactMatvecPlan{C} <: AbstractMatrix{Float64}
    shape::Tuple{Int,Int}
    rows::Vector{_CompactBasisNode}
    cols::Vector{_CompactBasisNode}
    couplings::Vector{C}
    dense::Vector{_PlanDense}
    rowcoeff::Vector{Float64}
    colcoeff::Vector{Float64}
    rowbuffer::Vector{Float64}
    colbuffer::Vector{Float64}
    rowperm::Vector{Int}
    colperm::Vector{Int}
end
# Independent per-item work (e.g. local SVDs) on the available threads; the
# result does not depend on scheduling.
function _threaded_map(f,::Type{T},items) where {T}
    out=Vector{T}(undef,length(items))
    next=Threads.Atomic{Int}(1)
    @sync for _ in 1:min(Threads.nthreads(),max(length(items),1))
        Threads.@spawn while true
            i=Threads.atomic_add!(next,1)
            i>length(items) && break
            out[i]=f(items[i])
        end
    end
    out
end
# `scale===nothing`: block-relative threshold; otherwise absolute `rtol*scale`.
# A coupling is replaced by factors only when they are cheaper to store.
function _factor_plan_coupling(S,row,col,rtol,scale=nothing)
    isempty(S) && return _LowRankPlanCoupling(S,nothing,Float64[],row,col)
    F=svd(S)
    k=scale===nothing ? _truncation_rank(F.S,rtol,min(size(S)...)) : count(>(rtol*scale),F.S)
    if k*sum(size(S))<length(S)
        _LowRankPlanCoupling(F.U[:,1:k]*Diagonal(F.S[1:k]),F.V[:,1:k],zeros(k),row,col)
    else
        _LowRankPlanCoupling(S,nothing,Float64[],row,col)
    end
end
function H2CompactMatvecPlan(h2::H2Matrix;coupling_rtol::Union{Nothing,Float64}=nothing,coupling_scale::Symbol=:block,
                             passthrough::Bool=false)
    coupling_rtol===nothing || (isfinite(coupling_rtol) && coupling_rtol>=0) ||
        throw(ArgumentError("coupling_rtol must be finite and nonnegative"))
    coupling_scale in (:block,:global) || throw(ArgumentError("coupling_scale must be :block or :global"))
    blocks=Tuple{typeof(h2.row_basis),typeof(h2.col_basis),UniformBlock}[];denseblocks=typeof(h2)[]
    function collect_blocks(h)
        if isleaf(h)
            if h.uniform!==nothing
                push!(blocks,(h.row_basis,h.col_basis,h.uniform))
            elseif h.dense!==nothing
                push!(denseblocks,h)
            end
        else
            foreach(collect_blocks,h.children)
        end
    end
    collect_blocks(h2)
    rowweights=colweights=nothing
    if passthrough
        width(cb)=cb.k>=length(cb) && cb.k>0 ? length(cb) : cb.k
        rowweights=IdDict{typeof(h2.row_basis),Int}()
        for (r,c,_) in blocks;rowweights[r]=get(rowweights,r,0)+width(c);end
    end
    rows,ri,ru,rc=_compact_basis(h2.row_basis,rowweights)
    if passthrough
        colweights=IdDict{typeof(h2.col_basis),Int}()
        for (r,c,_) in blocks;colweights[c]=get(colweights,c,0)+length(rows[ri[r]].coeff);end
    end
    cols,ci,cu,cc=_compact_basis(h2.col_basis,colweights)
    exact=_PlanCoupling[]
    dense=_PlanDense[]
    for (r,c,u) in blocks
        S=u.S
        # Physical row expansion and column projection of the original
        # basis are preserved, without relying on its orthogonality.
        haskey(ru,r) && (S=ru[r]*S)
        haskey(cu,c) && (S=S*cu[c]')
        push!(exact,_PlanCoupling(S,ri[r],ci[c]))
    end
    for h in denseblocks
        push!(dense,_PlanDense(h.dense,rows[ri[h.row_basis]].indices,cols[ci[h.col_basis]].indices))
    end
    couplings=if coupling_rtol===nothing
        exact
    else
        scale=if coupling_scale===:global
            # Largest stored block norm (near field included) as the operator scale.
            max(maximum(_threaded_map(b->isempty(b.S) ? 0. : opnorm(b.S),Float64,exact);init=0.),
                maximum(_threaded_map(b->isempty(b.D) ? 0. : opnorm(b.D),Float64,dense);init=0.))
        else
            nothing
        end
        # Release each transformed coupling once factorized to bound transient memory.
        out=Vector{_LowRankPlanCoupling}(undef,length(exact));next=Threads.Atomic{Int}(1)
        @sync for _ in 1:min(Threads.nthreads(),max(length(exact),1))
            Threads.@spawn while true
                i=Threads.atomic_add!(next,1)
                i>length(exact) && break
                b=exact[i]
                out[i]=_factor_plan_coupling(b.S,b.row,b.col,coupling_rtol,scale)
                exact[i]=_PlanCoupling(zeros(0,0),b.row,b.col)
            end
        end
        out
    end
    rp=h2.global_index ? collect(loc2glob(h2.row_basis.cluster)) : collect(1:size(h2,1))
    cp=h2.global_index ? collect(loc2glob(h2.col_basis.cluster)) : collect(1:size(h2,2))
    H2CompactMatvecPlan(size(h2),rows,cols,couplings,dense,rc,cc,zeros(size(h2,1)),zeros(size(h2,2)),rp,cp)
end
Base.size(p::H2CompactMatvecPlan)=p.shape
function _plan_up!(coeff,nodes::Vector{_CompactBasisNode},x)
    fill!(coeff,0.)
    for i in reverse(eachindex(nodes))
        n=nodes[i];isempty(n.coeff) && continue
        dest=view(coeff,n.coeff)
        if n.identity
            copyto!(dest,view(x,n.indices))
        elseif isempty(n.children)
            mul!(dest,n.V',view(x,n.indices))
        elseif n.passthrough
            offset=first(n.coeff)-1
            for j in n.children
                child=nodes[j];isempty(child.coeff) && continue
                for (a,b) in enumerate(child.coeff);coeff[offset+a]=coeff[b];end
                offset+=length(child.coeff)
            end
        else
            for j in n.children
                child=nodes[j];isempty(child.coeff) && continue
                mul!(dest,child.E',view(coeff,child.coeff),1.,1.)
            end
        end
    end
end
function _plan_down!(y,coeff,nodes::Vector{_CompactBasisNode})
    for n in nodes
        isempty(n.coeff) && continue
        if n.identity
            dest=view(y,n.indices);src=view(coeff,n.coeff)
            for i in eachindex(dest);dest[i]+=src[i];end
        elseif isempty(n.children)
            mul!(view(y,n.indices),n.V,view(coeff,n.coeff),1.,1.)
        elseif n.passthrough
            offset=first(n.coeff)-1
            for j in n.children
                child=nodes[j];isempty(child.coeff) && continue
                for (a,b) in enumerate(child.coeff);coeff[b]+=coeff[offset+a];end
                offset+=length(child.coeff)
            end
        else
            for j in n.children
                child=nodes[j];isempty(child.coeff) && continue
                mul!(view(coeff,child.coeff),child.E,view(coeff,n.coeff),1.,1.)
            end
        end
    end
end
const TransposedCompactH2Plan=Union{Transpose{Float64,<:H2CompactMatvecPlan},Adjoint{Float64,<:H2CompactMatvecPlan}}
LinearAlgebra.mul!(y::AbstractVector,p::H2CompactMatvecPlan,x::AbstractVector,alpha::Number=1,beta::Number=0)=_planned_mul!(y,p,x,alpha,beta,false)
LinearAlgebra.mul!(y::AbstractVector,p::TransposedCompactH2Plan,x::AbstractVector,alpha::Number=1,beta::Number=0)=_planned_mul!(y,parent(p),x,alpha,beta,true)
Base.:*(p::Union{H2CompactMatvecPlan,TransposedCompactH2Plan},x::AbstractVector)=mul!(zeros(size(p,1)),p,x)
function storage_bytes(p::H2CompactMatvecPlan)
    base=sum((sizeof(n.V)+sizeof(n.E) for ns in (p.rows,p.cols) for n in ns);init=0)+sum((sizeof(b.D) for b in p.dense);init=0)
    for b in p.couplings
        base+=b isa _PlanCoupling ? sizeof(b.S) : sizeof(b.L)+(b.R===nothing ? 0 : sizeof(b.R))
    end
    base
end
# Plans are intentionally matvec-only; displaying one must not index its matrix.
function Base.show(io::IO,p::Union{H2CompactMatvecPlan,H2LowRankMatvecPlan})
    print(io,nameof(typeof(p)),"(",size(p,1)," × ",size(p,2),", ",storage_bytes(p)," numeric bytes)")
end
Base.show(io::IO,::MIME"text/plain",p::Union{H2CompactMatvecPlan,H2LowRankMatvecPlan})=show(io,p)
# Copy only worker scratch. Geometry, bases and stored matrices remain shared.
_copy_plan_couplings(c::Vector{_PlanCoupling})=c
_copy_plan_couplings(c::Vector{_LowRankPlanCoupling})=[_LowRankPlanCoupling(b.L,b.R,zeros(length(b.scratch)),b.row,b.col) for b in c]
"""
    copy(plan)

Create an independent worker workspace while sharing the plan's numerical
operator data. Scratch used by factorized couplings is also copied. Source
matrix data and shared plan metadata must remain unchanged while workers run.
"""
function Base.copy(p::H2CompactMatvecPlan)
    H2CompactMatvecPlan(p.shape,p.rows,p.cols,_copy_plan_couplings(p.couplings),p.dense,
        zeros(length(p.rowcoeff)),zeros(length(p.colcoeff)),zeros(length(p.rowbuffer)),zeros(length(p.colbuffer)),p.rowperm,p.colperm)
end
function Base.copy(p::H2MatvecPlan)
    H2MatvecPlan(p.h2,p.rows,p.cols,p.couplings,p.dense,zeros(length(p.rowcoeff)),zeros(length(p.colcoeff)),zeros(length(p.rowbuffer)),zeros(length(p.colbuffer)),p.rowperm,p.colperm)
end
function Base.copy(p::H2LowRankMatvecPlan)
    H2LowRankMatvecPlan(p.shape,p.rows,p.cols,_copy_plan_couplings(p.couplings),p.dense,
        zeros(length(p.rowcoeff)),zeros(length(p.colcoeff)),zeros(length(p.rowbuffer)),zeros(length(p.colbuffer)),p.rowperm,p.colperm)
end

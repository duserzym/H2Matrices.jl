# Saturated bases can be represented in physical coordinates without a
# truncation. Their effects move into couplings and unsaturated-parent transfers.
struct _CompactBasisNode
    V::Matrix{Float64}
    E::Matrix{Float64}
    coeff::UnitRange{Int}
    indices::UnitRange{Int}
    children::Vector{Int}
    identity::Bool
end
function _compact_basis(root)
    originals=typeof(root)[]
    function visit(cb)
        push!(originals,cb)
        foreach(visit,cb.children)
    end
    visit(root)
    ids=IdDict(cb=>i for (i,cb) in enumerate(originals))
    # `nothing` marks a saturated basis whose expansion is exactly the identity
    # (as produced by the condensed H-matrix conversion): multiplying by it is
    # skipped, which gives bitwise the same couplings and transfers.
    saturated=IdDict{typeof(root),Union{Nothing,Matrix{Float64}}}()
    exact=IdDict{typeof(root),Bool}()
    for cb in Iterators.reverse(originals)
        exact[cb]=_is_exact_identity_basis(cb,exact)
    end
    for cb in originals
        if cb.k>=length(cb) && cb.k>0
            saturated[cb]=exact[cb] ? nothing : _full_basis(cb)
        end
    end
    nodes=_CompactBasisNode[];offset=first(index_range(root.cluster))-1;total=0
    for cb in originals
        identity=haskey(saturated,cb)
        k=identity ? length(cb) : cb.k
        cr=total+1:total+k;total+=k
        # Transfers to a saturated parent are unused: that parent's basis
        # expands directly in physical coordinates rather than through children.
        E=if isroot(cb) || haskey(saturated,cb.parent)
            zeros(Float64,0,0)
        elseif identity
            _expand_left(saturated[cb],cb.E)
        else
            cb.E
        end
        V=identity ? zeros(Float64,0,0) : cb.V
        push!(nodes,_CompactBasisNode(V,E,cr,index_range(cb.cluster).-offset,
            [ids[c] for c in cb.children],identity))
    end
    nodes,ids,saturated,zeros(total)
end
_expand_left(::Nothing,S)=S
_expand_left(V::Matrix{Float64},S)=V*S
_expand_right(S,::Nothing)=S
_expand_right(S,W::Matrix{Float64})=S*W'
# Is the expanded basis of `cb` exactly the identity? `exact` holds the answer
# for all children (reverse preorder).
function _is_exact_identity_basis(cb,exact)
    m=length(cb)
    cb.k==m && m>0 || return false
    isleaf(cb) && return size(cb.V)==(m,m) && cb.V==I
    off=0
    for child in cb.children
        exact[child] || return false
        E=child.E
        size(E)==(child.k,cb.k) || return false
        iszero(view(E,:,1:off)) && view(E,:,off+1:off+child.k)==I &&
            iszero(view(E,:,off+child.k+1:cb.k)) || return false
        off+=child.k
    end
    off==cb.k
end
"""
    H2CompactMatvecPlan(h2; coupling_rtol=nothing, coupling_scale=:block)

Compact reusable matvec representation of the stored H2 operator. Saturated
cluster bases (rank at least cluster size) are replaced by implicit identity
bases; their numerical action is moved to couplings and parent transfers.
This is an algebraic representation change with only floating-point rounding,
not a tolerance relaxation. Works for nonorthogonal and overcomplete bases.

An optional `coupling_rtol` additionally enables local SVD coupling truncation,
which is an approximation and requires separate application validation.
A coupling is replaced by its truncated SVD factors when these are smaller,
and dropped when no singular value survives. `coupling_scale` sets the
threshold: `:block` (default) discards singular values below `coupling_rtol`
times the coupling's own largest one; `:global` discards those below
`coupling_rtol` times the largest spectral norm over all stored blocks
(couplings and near field); a positive number `s` discards those below
`coupling_rtol*s` (e.g. `s` from [`estimate_operator_scale`](@ref)). Global
scales stop weak far-field blocks from being resolved to the same relative
accuracy as the strongest blocks. Packet plans built from this plan keep the
factors.
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
# `scale===nothing`: block-relative threshold `rtol*σ₁`; otherwise the absolute
# threshold `rtol*scale`. Factors replace a coupling only when they are smaller.
function _factor_plan_coupling(S,row,col,rtol,scale=nothing)
    isempty(S) && return _LowRankPlanCoupling(S,nothing,Float64[],row,col)
    F=svd(S)
    k=scale===nothing ? _truncation_rank(F.S,rtol,min(size(S)...)) : _truncation_rank(F.S,0.0,min(size(S)...),rtol*scale)
    if k*sum(size(S))<length(S)
        _LowRankPlanCoupling(F.U[:,1:k]*Diagonal(F.S[1:k]),F.V[:,1:k],zeros(k),row,col)
    else
        _LowRankPlanCoupling(S,nothing,Float64[],row,col)
    end
end
function H2CompactMatvecPlan(h2::H2Matrix;coupling_rtol::Union{Nothing,Float64}=nothing,coupling_scale=:block)
    coupling_rtol===nothing || (isfinite(coupling_rtol) && coupling_rtol>=0) ||
        throw(ArgumentError("coupling_rtol must be finite and nonnegative"))
    coupling_scale===:block || coupling_scale===:global ||
        (coupling_scale isa Real && !(coupling_scale isa Bool) && isfinite(coupling_scale) && coupling_scale>0) ||
        throw(ArgumentError("coupling_scale must be :block, :global or a positive finite number"))
    coupling_rtol===nothing && coupling_scale!==:block &&
        throw(ArgumentError("coupling_scale=$coupling_scale requires coupling_rtol"))
    rows,ri,ru,rc=_compact_basis(h2.row_basis)
    cols,ci,cu,cc=_compact_basis(h2.col_basis)
    couplings=_PlanCoupling[]
    dense=_PlanDense[]
    function visit(h)
        if isleaf(h)
            if h.uniform!==nothing
                S=h.uniform.S
                # Physical row expansion and column projection of the original
                # basis are preserved, without relying on its orthogonality.
                haskey(ru,h.row_basis) && (S=_expand_left(ru[h.row_basis],S))
                haskey(cu,h.col_basis) && (S=_expand_right(S,cu[h.col_basis]))
                r=ri[h.row_basis];c=ci[h.col_basis]
                push!(couplings,_PlanCoupling(S,r,c))
            elseif h.dense!==nothing
                push!(dense,_PlanDense(h.dense,rows[ri[h.row_basis]].indices,cols[ci[h.col_basis]].indices))
            end
        else
            foreach(visit,h.children)
        end
    end
    visit(h2)
    if coupling_rtol!==nothing
        scale=if coupling_scale===:global
            max(maximum(_threaded_map(i->isempty(couplings[i].S) ? 0. : opnorm(couplings[i].S),Float64,length(couplings));init=0.),
                maximum(_threaded_map(i->isempty(dense[i].D) ? 0. : opnorm(dense[i].D),Float64,length(dense));init=0.))
        elseif coupling_scale isa Real
            Float64(coupling_scale)
        else
            nothing
        end
        # Each transformed coupling is dropped once factorized (bounds transient memory).
        exact=couplings
        couplings=_threaded_map(_LowRankPlanCoupling,length(exact)) do i
            b=exact[i];exact[i]=_PlanCoupling(_RELEASED_BLOCK,b.row,b.col)
            _factor_plan_coupling(b.S,b.row,b.col,coupling_rtol,scale)
        end
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

# Apply `f(i)` for `i in 1:n` on the available threads (independent work such
# as local SVDs); the result does not depend on scheduling.
function _threaded_map(f,::Type{T},n::Integer) where {T}
    out=Vector{T}(undef,n)
    next=Threads.Atomic{Int}(1)
    @sync for _ in 1:min(Threads.nthreads(),max(n,1))
        Threads.@spawn while true
            i=Threads.atomic_add!(next,1)
            i>n && break
            out[i]=f(i)
        end
    end
    out
end

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
# Placeholder for a numerical block released by a consuming builder.
const _RELEASED_BLOCK=zeros(Float64,0,0)
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
        elseif weights!==nothing && isleaf(cb) && cb.k>0
            # The same storage rule with D=|t|: physical coordinates for a
            # nearly saturated leaf whose few couplings do not amortize V.
            m=length(cb);kp=isroot(cb) || haskey(saturated,cb.parent) ? 0 : cb.parent.k
            (m-cb.k)*(kp+get(weights,cb,0))<m*cb.k && (saturated[cb]=cb.V)
        end
    end
    # Expansion of a node's own k coefficients into the coordinates it uses
    # (`nothing`: exactly the identity).
    expand=IdDict{typeof(root),Union{Nothing,Matrix{Float64}}}(saturated)
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
                    expand[cb]=reduce(vcat,[haskey(expand,c) ? _expand_left(expand[c],c.E) : c.E for c in active])
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
            _expand_left(expand[cb],cb.E)
        else
            cb.E
        end
        V=identity || pass ? zeros(Float64,0,0) : cb.V
        push!(nodes,_CompactBasisNode(V,E,cr,index_range(cb.cluster).-offset,
            [ids[c] for c in cb.children],identity,pass))
    end
    nodes,ids,expand,zeros(total)
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
    H2CompactMatvecPlan(h2; coupling_rtol=nothing, coupling_scale=:block, passthrough=false,
                        coupling_precision=Float64, consume=false)

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
It adds no tolerance by itself; combined with `coupling_rtol`, the folded
couplings are larger (rectangular) blocks, so their truncation differs from
the truncation without pass-through and must be validated as such.

An optional `coupling_rtol` additionally enables local SVD coupling truncation,
which is an approximation and requires separate application validation.
With `coupling_scale=:block` (default) singular values below `coupling_rtol`
times the coupling's own largest singular value are discarded. With
`coupling_scale=:global` the threshold is `coupling_rtol` times the largest
spectral norm over all stored blocks (couplings and near field), so weak
blocks are not resolved to a tighter absolute accuracy than strong ones; a
positive number `s` gives the absolute threshold `coupling_rtol*s` (for
example with `s` from [`estimate_operator_scale`](@ref)), which skips the
spectral norms. A coupling is replaced by factors only when they use less
storage than the coupling itself (a coupling without any retained singular
value is dropped). `coupling_scale` other than `:block` and
`coupling_precision` qualify the truncation and therefore require
`coupling_rtol`.

`coupling_precision=Float32` additionally stores the retained singular
components below `tau/eps(Float32)` (`tau` the truncation threshold) in
Float32, either as factors or as a dense remainder, keeping the larger
components in Float64. `coupling_precision=Float16` adds a third tier:
components below `tau/eps(Float16)` are stored as Float16 factors. Factors are
scaled column by column with exact powers of two where needed (always for
Float16), so the reduced-precision tiers keep their relative precision for
operators of any magnitude; a Float32 dense remainder is used only when its
entries are within the Float32 range. Products still use Float64 arithmetic,
so the adjoint remains the exact transpose of the stored operator up to
Float64 rounding. Rounding a component `σ u v'` perturbs it by about
`2 eps(T) σ`, at most about `2 tau` in each tier; a coupling's rounding error
is the sum over its reduced-precision components, so it can exceed the
truncation error (an estimate, not a bound): validate it like any coupling
truncation.

`consume=true` releases the source operator's coupling and near-field blocks
while the plan is built (each transformed coupling is in turn released once
factorized); `h2` is unusable afterwards. The plan is identical to
`consume=false`.

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
# Float32/Float16-stored matrices applied in Float64 arithmetic: the stored
# numbers are exact in Float64, so forward and transposed products apply the
# same operator up to Float64 rounding. y .+= A*x
const _LowFloat=Union{Float32,Float16}
function _mixed_mul!(y::AbstractVector{Float64},A::Matrix{<:_LowFloat},x::AbstractVector{Float64})
    m,n=size(A)
    (length(y)==m && length(x)==n) || throw(DimensionMismatch("mixed-precision product dimensions"))
    j=1
    @inbounds while j+3<=n
        x1=x[j];x2=x[j+1];x3=x[j+2];x4=x[j+3]
        @simd for i in 1:m
            y[i]=muladd(Float64(A[i,j]),x1,muladd(Float64(A[i,j+1]),x2,muladd(Float64(A[i,j+2]),x3,muladd(Float64(A[i,j+3]),x4,y[i]))))
        end
        j+=4
    end
    @inbounds while j<=n
        xj=x[j]
        @simd for i in 1:m
            y[i]=muladd(Float64(A[i,j]),xj,y[i])
        end
        j+=1
    end
    y
end
# y = A'*x (accumulate=false) or y .+= A'*x
function _mixed_tmul!(y::AbstractVector{Float64},A::Matrix{<:_LowFloat},x::AbstractVector{Float64},accumulate::Bool)
    m,n=size(A)
    (length(y)==n && length(x)==m) || throw(DimensionMismatch("mixed-precision product dimensions"))
    @inbounds for j in 1:n
        acc=0.
        @simd for i in 1:m
            acc=muladd(Float64(A[i,j]),x[i],acc)
        end
        y[j]=accumulate ? y[j]+acc : acc
    end
    y
end
# A coupling stored as a Float64 part, a Float32 part and a Float16 part.
# Float64 part: dense `L` (`R===nothing`) or factors `L*R'`; Float32 part:
# dense remainder `L32` (`R32===nothing`) or factors `L32*Diagonal(c32)*R32'`
# (`c32` empty: unscaled); Float16 part: factors `L16*Diagonal(c16)*R16'`.
# Column scales are exact powers of two.
struct _MixedPlanCoupling
    L::Matrix{Float64}
    R::Union{Nothing,Matrix{Float64}}
    L32::Matrix{Float32}
    R32::Union{Nothing,Matrix{Float32}}
    c32::Vector{Float64}
    L16::Matrix{Float16}
    R16::Matrix{Float16}
    c16::Vector{Float64}
    scratch::Vector{Float64}
    row::Int
    col::Int
end
const _NO16=Matrix{Float16}(undef,0,0)
const _NO32=Matrix{Float32}(undef,0,0)
# Factors `A*B'` stored as T with columns scaled by powers of two (exact), so
# every column's largest entry lies in [1,2): T keeps its relative precision
# whatever the operator's magnitude. Returns (A_T, B_T, scales).
function _scaled_factors(::Type{T},A,B) where {T}
    ea=[iszero(x) ? 0 : exponent(x) for x in vec(maximum(abs,A;dims=1))]
    eb=[iszero(x) ? 0 : exponent(x) for x in vec(maximum(abs,B;dims=1))]
    T.(A*Diagonal(exp2.(-ea))),T.(B*Diagonal(exp2.(-eb))),exp2.(ea.+eb)
end
_half_factors(A,B)=_scaled_factors(Float16,A,B)
# Can `A` be stored as T without leaving T's range? Its largest entry must be
# a normal T number (then the absolute rounding error of every entry, subnormal
# ones included, is at most eps(T) times that entry) and must not overflow.
function _in_range(::Type{T},A) where {T}
    m=maximum(abs,A;init=0.)
    iszero(m) || floatmin(T)<=m<=floatmax(T)/4
end
# Float32 factors: unscaled when every column is in the Float32 range (the
# common case: identical numbers and no scale storage), else scaled.
function _single_factors(A,B)
    all(j->_in_range(Float32,view(A,:,j)),axes(A,2)) && all(j->_in_range(Float32,view(B,:,j)),axes(B,2)) &&
        return Float32.(A),Float32.(B),Float64[]
    _scaled_factors(Float32,A,B)
end
# Keep singular components above `tau=rtol*scale` (block norm if `scale===nothing`).
# Components below `tau/eps(Float32)` are stored in Float32 and, with `half`,
# those below `tau/eps(Float16)` in scaled Float16. Rounding a component
# sigma*u*v' to T perturbs it by about 2*eps(T)*sigma in norm, which is at
# most about 2*tau for the components each tier holds; a block's rounding error
# is the sum over its reduced-precision components (in practice it grows like
# their square root count times tau; this is an estimate, not a bound). The
# cheapest of exact dense Float64, mixed factors, or Float64 factors plus a
# Float32 dense remainder is kept; the remainder is used only when its entries
# are within the Float32 range.
function _mixed_plan_coupling(S,row,col,rtol,scale=nothing,half::Bool=false)
    m,n=size(S)
    dense_coupling()=_MixedPlanCoupling(S,nothing,_NO32,nothing,Float64[],_NO16,_NO16,Float64[],Float64[],row,col)
    isempty(S) && return dense_coupling()
    F=svd(S);s=F.S
    tau=rtol*(scale===nothing ? s[1] : scale)
    k=count(>(tau),s);hi=count(>(tau/eps(Float32)),s)
    mid=half ? count(>(tau/eps(Float16)),s)-hi : k-hi;lo=k-hi-mid
    dense=8m*n;fact=(8hi+4mid+2lo)*(m+n)+8lo;split=8hi*(m+n)+4m*n
    dense<=min(fact,split) && return dense_coupling()
    L=F.U[:,1:hi]*Diagonal(s[1:hi]);R=F.V[:,1:hi]
    if split<fact
        rem=S-L*R'
        _in_range(Float32,rem) &&
            return _MixedPlanCoupling(L,R,Float32.(rem),nothing,Float64[],_NO16,_NO16,Float64[],zeros(hi),row,col)
        dense<=fact && return dense_coupling()
    end
    a,b=hi+1:hi+mid,hi+mid+1:k
    L32,R32,c32=mid>0 ? _single_factors(F.U[:,a]*Diagonal(s[a]),F.V[:,a]) : (_NO32,_NO32,Float64[])
    L16,R16,c16=lo>0 ? _half_factors(F.U[:,b]*Diagonal(s[b]),F.V[:,b]) : (_NO16,_NO16,Float64[])
    _MixedPlanCoupling(L,R,L32,R32,c32,L16,R16,c16,zeros(max(hi,mid,lo)),row,col)
end
function _plan_coupling_mul!(y,b::_MixedPlanCoupling,x,t)
    if b.R===nothing
        isempty(b.L) || mul!(y,t ? b.L' : b.L,x,1.,1.)
    elseif size(b.R,2)>0
        _lowrank_coupling_apply!(y,b.L,b.R,view(b.scratch,1:size(b.R,2)),x,t)
    end
    if b.R32===nothing
        isempty(b.L32) || (t ? _mixed_tmul!(y,b.L32,x,true) : _mixed_mul!(y,b.L32,x))
    elseif size(b.R32,2)>0
        z=view(b.scratch,1:size(b.R32,2))
        _mixed_tmul!(z,t ? b.L32 : b.R32,x,false)
        isempty(b.c32) || (for i in eachindex(z);z[i]*=b.c32[i];end)
        _mixed_mul!(y,t ? b.R32 : b.L32,z)
    end
    if !isempty(b.c16)
        z=view(b.scratch,1:length(b.c16))
        _mixed_tmul!(z,t ? b.L16 : b.R16,x,false)
        for i in eachindex(z);z[i]*=b.c16[i];end
        _mixed_mul!(y,t ? b.R16 : b.L16,z)
    end
    y
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
# Largest spectral norm over `mats`. A matrix whose Frobenius norm does not
# exceed a spectral norm already found cannot raise the maximum, so its
# spectral norm is skipped: the result equals the maximum over all of them.
function _max_opnorm(mats)
    isempty(mats) && return 0.
    fro=_threaded_map(A->isempty(A) ? 0. : norm(A),Float64,mats)
    order=sortperm(fro;rev=true)
    best=0.;i=1
    while i<=length(order) && fro[order[i]]>best
        # Batches in decreasing Frobenius norm, each evaluated on all threads.
        j=i;while j<length(order) && j-i+1<4Threads.nthreads() && fro[order[j+1]]>best;j+=1;end
        best=max(best,maximum(_threaded_map(t->opnorm(mats[t]),Float64,order[i:j])))
        i=j+1
    end
    best
end
function _validate_compact_options(coupling_rtol,coupling_scale,coupling_precision)
    coupling_rtol===nothing || (isfinite(coupling_rtol) && coupling_rtol>=0) ||
        throw(ArgumentError("coupling_rtol must be finite and nonnegative"))
    coupling_scale===:block || coupling_scale===:global ||
        (coupling_scale isa Real && !(coupling_scale isa Bool) && isfinite(coupling_scale) && coupling_scale>0) ||
        throw(ArgumentError("coupling_scale must be :block, :global or a positive finite number"))
    coupling_precision in (Float64,Float32,Float16) || throw(ArgumentError("coupling_precision must be Float64, Float32 or Float16"))
    # Both options only qualify coupling truncation: without it they would be
    # silently ignored, so both are rejected alike.
    coupling_rtol===nothing && coupling_scale!==:block &&
        throw(ArgumentError("coupling_scale=$(repr(coupling_scale)) requires coupling_rtol"))
    coupling_rtol===nothing && coupling_precision!==Float64 &&
        throw(ArgumentError("coupling_precision=$coupling_precision requires coupling_rtol"))
    nothing
end
function H2CompactMatvecPlan(h2::H2Matrix;coupling_rtol::Union{Nothing,Real}=nothing,coupling_scale::Union{Symbol,Real}=:block,
                             passthrough::Bool=false,coupling_precision::Type=Float64,consume::Bool=false)
    coupling_rtol===nothing || (coupling_rtol=Float64(coupling_rtol))
    _validate_compact_options(coupling_rtol,coupling_scale,coupling_precision)
    leaves=typeof(h2)[];denseleaves=typeof(h2)[]
    function collect_blocks(h)
        if isleaf(h)
            if h.uniform!==nothing
                push!(leaves,h)
            elseif h.dense!==nothing
                push!(denseleaves,h)
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
        for h in leaves;rowweights[h.row_basis]=get(rowweights,h.row_basis,0)+width(h.col_basis);end
    end
    rows,ri,ru,rc=_compact_basis(h2.row_basis,rowweights)
    if passthrough
        colweights=IdDict{typeof(h2.col_basis),Int}()
        for h in leaves;colweights[h.col_basis]=get(colweights,h.col_basis,0)+length(rows[ri[h.row_basis]].coeff);end
    end
    cols,ci,cu,cc=_compact_basis(h2.col_basis,colweights)
    tracker=consume ? _ReleaseTracker(sum((sizeof(h.uniform.S) for h in leaves);init=0)) : nothing
    exact=Vector{_PlanCoupling}(undef,length(leaves))
    for (i,h) in enumerate(leaves)
        r=h.row_basis;c=h.col_basis;S0=h.uniform.S;S=S0
        # Physical row expansion and column projection of the original
        # basis are preserved, without relying on its orthogonality.
        haskey(ru,r) && (S=_expand_left(ru[r],S))
        haskey(cu,c) && (S=_expand_right(S,cu[c]))
        exact[i]=_PlanCoupling(S,ri[r],ci[c])
        # Drop the source block once transformed (an exact-identity transform
        # keeps the same array, now referenced only by the plan).
        if consume
            h.uniform=nothing
            S===S0 || _released!(tracker,sizeof(S0))
        end
    end
    dense=_PlanDense[]
    for h in denseleaves
        push!(dense,_PlanDense(h.dense,rows[ri[h.row_basis]].indices,cols[ci[h.col_basis]].indices))
        consume && (h.dense=nothing)
    end
    empty!(leaves);empty!(denseleaves)
    couplings=if coupling_rtol===nothing
        exact
    else
        # :global: largest stored block norm (near field included) as the operator scale.
        scale=coupling_scale===:global ? max(_max_opnorm([b.S for b in exact]),_max_opnorm([b.D for b in dense])) :
            coupling_scale isa Real ? Float64(coupling_scale) : nothing
        mixed=coupling_precision!==Float64;half=coupling_precision===Float16
        # Release each transformed coupling once factorized to bound transient memory.
        out=Vector{mixed ? _MixedPlanCoupling : _LowRankPlanCoupling}(undef,length(exact));next=Threads.Atomic{Int}(1)
        @sync for _ in 1:min(Threads.nthreads(),max(length(exact),1))
            Threads.@spawn while true
                i=Threads.atomic_add!(next,1)
                i>length(exact) && break
                b=exact[i]
                out[i]=mixed ? _mixed_plan_coupling(b.S,b.row,b.col,coupling_rtol,scale,half) :
                    _factor_plan_coupling(b.S,b.row,b.col,coupling_rtol,scale)
                exact[i]=_PlanCoupling(_RELEASED_BLOCK,b.row,b.col)
                out[i].L===b.S || _released!(tracker,sizeof(b.S))
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
        base+=_coupling_bytes(b)
    end
    base
end
# Plans are intentionally matvec-only; displaying one must not index its matrix.
function Base.show(io::IO,p::Union{H2CompactMatvecPlan,H2LowRankMatvecPlan})
    print(io,nameof(typeof(p)),"(",size(p,1)," × ",size(p,2),", ",storage_bytes(p)," numeric bytes)")
end
Base.show(io::IO,::MIME"text/plain",p::Union{H2CompactMatvecPlan,H2LowRankMatvecPlan})=show(io,p)
# Copy only worker scratch. Geometry, bases and stored matrices remain shared.
_coupling_bytes(b::_PlanCoupling)=sizeof(b.S)
_coupling_bytes(b::_LowRankPlanCoupling)=sizeof(b.L)+(b.R===nothing ? 0 : sizeof(b.R))
_coupling_bytes(b::_MixedPlanCoupling)=sizeof(b.L)+(b.R===nothing ? 0 : sizeof(b.R))+sizeof(b.L32)+(b.R32===nothing ? 0 : sizeof(b.R32))+
    sizeof(b.c32)+sizeof(b.L16)+sizeof(b.R16)+sizeof(b.c16)
_copy_plan_couplings(c::Vector{_MixedPlanCoupling})=[_MixedPlanCoupling(b.L,b.R,b.L32,b.R32,b.c32,b.L16,b.R16,b.c16,zeros(length(b.scratch)),b.row,b.col) for b in c]
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

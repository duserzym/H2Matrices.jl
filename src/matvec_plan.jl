# A plan owns mutable scratch buffers: use one plan per concurrent worker and
# rebuild it after changing/recompressing the underlying H2 matrix.
struct _PlanBasisNode
    V::Matrix{Float64}
    E::Matrix{Float64}
    coeff::UnitRange{Int}
    indices::UnitRange{Int}
    children::Vector{Int}
end
struct _PlanCoupling
    S::Matrix{Float64}
    row::Int
    col::Int
end
struct _PlanDense
    D::Matrix{Float64}
    rows::UnitRange{Int}
    cols::UnitRange{Int}
end
"""
    H2MatvecPlan(h2)

Flatten traversal and reuse scratch storage for forward and exact adjoint
products of a stored Float64 H2 operator. Numerical matrices are shared, not
copied. Plans are not thread-safe: create one per concurrent worker. Rebuild
all plans after mutating or recompressing `h2`. Global/local ordering follows
`h2.global_index` at construction time.
"""
struct H2MatvecPlan{H<:H2Matrix} <: AbstractMatrix{Float64}
    h2::H
    rows::Vector{_PlanBasisNode}
    cols::Vector{_PlanBasisNode}
    couplings::Vector{_PlanCoupling}
    dense::Vector{_PlanDense}
    rowcoeff::Vector{Float64}
    colcoeff::Vector{Float64}
    rowbuffer::Vector{Float64}
    colbuffer::Vector{Float64}
    rowperm::Vector{Int}
    colperm::Vector{Int}
end
function _plan_basis(root)
    nodes = _PlanBasisNode[]
    ids = IdDict{typeof(root),Int}()
    offset = first(index_range(root.cluster))-1
    total = Ref(0)
    function visit(cb)
        id=length(nodes)+1; ids[cb]=id
        cr=total[]+1:total[]+cb.k; total[]+=cb.k
        children=Int[]
        push!(nodes,_PlanBasisNode(cb.V,cb.E,cr,index_range(cb.cluster).-offset,children))
        for child in cb.children
            push!(children,visit(child))
        end
        id
    end
    visit(root)
    nodes,ids,zeros(total[])
end
function H2MatvecPlan(h2::H2Matrix)
    rows,ri,rc=_plan_basis(h2.row_basis)
    cols,ci,cc=_plan_basis(h2.col_basis)
    couplings=_PlanCoupling[]; dense=_PlanDense[]
    function visit(h)
        if isleaf(h)
            if h.uniform !== nothing
                push!(couplings,_PlanCoupling(h.uniform.S,ri[h.row_basis],ci[h.col_basis]))
            elseif h.dense !== nothing
                push!(dense,_PlanDense(h.dense,rows[ri[h.row_basis]].indices,cols[ci[h.col_basis]].indices))
            end
        else
            foreach(visit,h.children)
        end
    end
    visit(h2)
    rp=h2.global_index ? collect(loc2glob(h2.row_basis.cluster)) : collect(1:size(h2,1))
    cp=h2.global_index ? collect(loc2glob(h2.col_basis.cluster)) : collect(1:size(h2,2))
    H2MatvecPlan(h2,rows,cols,couplings,dense,rc,cc,zeros(size(h2,1)),zeros(size(h2,2)),rp,cp)
end
Base.size(p::H2MatvecPlan)=size(p.h2)
Base.getindex(p::H2MatvecPlan,i::Int,j::Int)=p.h2[i,j]
function _plan_up!(coeff,nodes,x)
    fill!(coeff,0.0)
    for i in reverse(eachindex(nodes))
        n=nodes[i]
        isempty(n.coeff) && continue
        dest=view(coeff,n.coeff)
        if isempty(n.children)
            mul!(dest,n.V',view(x,n.indices))
        else
            for j in n.children
                child=nodes[j]
                isempty(child.coeff) && continue
                mul!(dest,child.E',view(coeff,child.coeff),1.0,1.0)
            end
        end
    end
end
function _plan_down!(y,coeff,nodes)
    for n in nodes
        isempty(n.coeff) && continue
        if isempty(n.children)
            mul!(view(y,n.indices),n.V,view(coeff,n.coeff),1.0,1.0)
        else
            for j in n.children
                child=nodes[j]
                isempty(child.coeff) && continue
                mul!(view(coeff,child.coeff),child.E,view(coeff,n.coeff),1.0,1.0)
            end
        end
    end
end
function _planned_mul!(y,p,x,alpha,beta,transposed)
    inputnodes,outputnodes = transposed ? (p.rows,p.cols) : (p.cols,p.rows)
    inputcoeff,outputcoeff = transposed ? (p.rowcoeff,p.colcoeff) : (p.colcoeff,p.rowcoeff)
    input,output = transposed ? (p.rowbuffer,p.colbuffer) : (p.colbuffer,p.rowbuffer)
    ip,op = transposed ? (p.rowperm,p.colperm) : (p.colperm,p.rowperm)
    length(x)==length(input) && length(y)==length(output) || throw(DimensionMismatch("incompatible planned H2 matvec dimensions"))
    if iszero(alpha)
        iszero(beta) ? fill!(y,0.0) : rmul!(y,beta)
        return y
    end
    for i in eachindex(input)
        input[i]=x[ip[i]]
    end
    fill!(output,0.0);fill!(outputcoeff,0.0)
    _plan_up!(inputcoeff,inputnodes,input)
    for b in p.couplings
        ir=transposed ? p.rows[b.row].coeff : p.cols[b.col].coeff
        or=transposed ? p.cols[b.col].coeff : p.rows[b.row].coeff
        isempty(ir) || isempty(or) || _plan_coupling_mul!(view(outputcoeff,or),b,view(inputcoeff,ir),transposed)
    end
    _plan_down!(output,outputcoeff,outputnodes)
    for b in p.dense
        ir,or=transposed ? (b.rows,b.cols) : (b.cols,b.rows)
        mul!(view(output,or),transposed ? b.D' : b.D,view(input,ir),1.0,1.0)
    end
    for i in eachindex(output)
        j=op[i]
        y[j]=iszero(beta) ? alpha*output[i] : alpha*output[i]+beta*y[j]
    end
    y
end
LinearAlgebra.mul!(y::AbstractVector,p::H2MatvecPlan,x::AbstractVector,alpha::Number=1,beta::Number=0)=_planned_mul!(y,p,x,alpha,beta,false)
const TransposedH2Plan=Union{Transpose{Float64,<:H2MatvecPlan},Adjoint{Float64,<:H2MatvecPlan}}
LinearAlgebra.mul!(y::AbstractVector,p::TransposedH2Plan,x::AbstractVector,alpha::Number=1,beta::Number=0)=_planned_mul!(y,parent(p),x,alpha,beta,true)
Base.:*(p::Union{H2MatvecPlan,TransposedH2Plan},x::AbstractVector)=mul!(zeros(size(p,1)),p,x)

_plan_coupling_mul!(y,b::_PlanCoupling,x,t)=mul!(y,t ? b.S' : b.S,x,1.0,1.0)
struct _LowRankPlanCoupling
    L::Matrix{Float64}
    R::Union{Nothing,Matrix{Float64}}
    scratch::Vector{Float64}
    row::Int
    col::Int
end
"""
    H2LowRankMatvecPlan(h2; rtol=1e-10)

Experimental matvec-only H2 plan with selected coupling matrices replaced by
SVD factors when those factors use less numeric storage. This adds a local
relative spectral truncation, not a global error certificate. It retains
bases and near-field matrices, but not the original operator or discarded
couplings. Its adjoint is the exact transpose of the stored approximation.
Use a separate plan per concurrent worker; do not mutate shared bases.
"""
struct H2LowRankMatvecPlan <: AbstractMatrix{Float64}
    shape::Tuple{Int,Int}
    rows::Vector{_PlanBasisNode}
    cols::Vector{_PlanBasisNode}
    couplings::Vector{_LowRankPlanCoupling}
    dense::Vector{_PlanDense}
    rowcoeff::Vector{Float64}
    colcoeff::Vector{Float64}
    rowbuffer::Vector{Float64}
    colbuffer::Vector{Float64}
    rowperm::Vector{Int}
    colperm::Vector{Int}
end
function H2LowRankMatvecPlan(h2::H2Matrix;rtol::Float64=1e-10)
    isfinite(rtol) && rtol>=0 || throw(ArgumentError("rtol must be finite and nonnegative"))
    p=H2MatvecPlan(h2);couplings=_LowRankPlanCoupling[]
    for b in p.couplings
        F=svd(b.S);k=_truncation_rank(F.S,rtol,min(size(b.S)...))
        if k*sum(size(b.S))<length(b.S)
            L=F.U[:,1:k]*Diagonal(F.S[1:k]);R=F.V[:,1:k]
            push!(couplings,_LowRankPlanCoupling(L,R,zeros(k),b.row,b.col))
        else
            push!(couplings,_LowRankPlanCoupling(b.S,nothing,Float64[],b.row,b.col))
        end
    end
    H2LowRankMatvecPlan(size(p),p.rows,p.cols,couplings,p.dense,p.rowcoeff,p.colcoeff,p.rowbuffer,p.colbuffer,p.rowperm,p.colperm)
end
Base.size(p::H2LowRankMatvecPlan)=p.shape
@inline function _plan_coupling_mul!(y,b::_LowRankPlanCoupling,x,t)
    R=b.R
    if R===nothing
        mul!(y,t ? b.L' : b.L,x,1.0,1.0)
    elseif !isempty(b.scratch)
        _lowrank_coupling_apply!(y,b.L,R,b.scratch,x,t)
    end
end
@inline function _lowrank_coupling_apply!(y,L,R,scratch,x,t)
    mul!(scratch,t ? L' : R',x)
    mul!(y,t ? R : L,scratch,1.0,1.0)
end
const TransposedLowRankH2Plan=Union{Transpose{Float64,H2LowRankMatvecPlan},Adjoint{Float64,H2LowRankMatvecPlan}}
LinearAlgebra.mul!(y::AbstractVector,p::H2LowRankMatvecPlan,x::AbstractVector,alpha::Number=1,beta::Number=0)=_planned_mul!(y,p,x,alpha,beta,false)
LinearAlgebra.mul!(y::AbstractVector,p::TransposedLowRankH2Plan,x::AbstractVector,alpha::Number=1,beta::Number=0)=_planned_mul!(y,parent(p),x,alpha,beta,true)
Base.:*(p::Union{H2LowRankMatvecPlan,TransposedLowRankH2Plan},x::AbstractVector)=mul!(zeros(size(p,1)),p,x)
function _lowrank_plan_storage_bytes(p::H2LowRankMatvecPlan)
    sum((sizeof(n.V)+sizeof(n.E) for nodes in (p.rows,p.cols) for n in nodes);init=0)+
    sum((sizeof(b.D) for b in p.dense);init=0)+
    sum((sizeof(b.L)+(b.R===nothing ? 0 : sizeof(b.R)) for b in p.couplings);init=0)
end

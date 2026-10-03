struct _CouplingPacket
    matrix::Matrix{Float64}
    row::UnitRange{Int}
    columns::Vector{UnitRange{Int}}
    scratch::Vector{Float64}
end
"""
    H2PacketMatvecPlan(h2; workers=1)
    H2PacketMatvecPlan(compact_plan; workers=1)

Pack interactions with the same row basis into contiguous GEMV packets.
Packing does not truncate data. Combined with implicit saturated bases, it
preserves the source operator up to floating-point rounding. Numeric blocks
are retained without the source operator. `workers>1` partitions packets
among tasks, including near-field packets with disjoint output ranges,
with separate transpose reduction buffers and deterministic
worker-order reduction. Use BLAS threads=1 when enabling packet workers.

One plan is not safe for concurrent calls; `copy(plan)` shares numerical data
and allocates independent scratch for an additional caller. Source data must
remain unchanged while plans are used.

`consume=true` (H2 input only) releases the source operator's coupling and
near-field blocks as they are packed, bounding peak construction memory by
roughly one operator plus one packet. The source `h2` is unusable afterwards
(its leaves hold no numerical blocks); the plan is bitwise identical.
"""
struct H2PacketMatvecPlan <: AbstractMatrix{Float64}
    shape::Tuple{Int,Int}
    rows::Vector{_CompactBasisNode}
    cols::Vector{_CompactBasisNode}
    packets::Vector{_CouplingPacket}
    nearpackets::Vector{_CouplingPacket}
    near_parallel::Bool
    rowcoeff::Vector{Float64}
    colcoeff::Vector{Float64}
    rowbuffer::Vector{Float64}
    colbuffer::Vector{Float64}
    rowperm::Vector{Int}
    colperm::Vector{Int}
    partials::Vector{Vector{Float64}}
    nearpartials::Vector{Vector{Float64}}
end
function H2PacketMatvecPlan(h::H2Matrix;workers::Int=1,consume::Bool=false)
    workers>0 || throw(ArgumentError("workers must be positive"))
    core=H2CompactMatvecPlan(h)
    # The compact plan now references every numerical block it needs, so the
    # source blocks can be dropped and each packed group released in turn.
    consume && _release_h2_blocks!(h)
    H2PacketMatvecPlan(core;workers,_release=consume)
end
const _RELEASED_BLOCK=zeros(Float64,0,0)
function _release_h2_blocks!(h::H2Matrix)
    if isleaf(h)
        h.uniform=nothing;h.dense=nothing
    else
        foreach(_release_h2_blocks!,h.children)
    end
    h
end
function H2PacketMatvecPlan(p::H2CompactMatvecPlan;workers::Int=1,_release::Bool=false)
    workers>0 || throw(ArgumentError("workers must be positive"))
    tracker=_release ? _ReleaseTracker(storage_bytes(p)) : nothing
    groups=Dict{Int,Vector{Int}}();order=Int[]
    for (i,b) in enumerate(p.couplings)
        if isempty(p.rows[b.row].coeff) || isempty(p.cols[b.col].coeff)
            _release && (p.couplings[i]=_released_coupling(b))
        else
            haskey(groups,b.row) || (groups[b.row]=Int[];push!(order,b.row))
            push!(groups[b.row],i)
        end
    end
    packets=_CouplingPacket[]
    for row in order
        columns=UnitRange{Int}[];blocks=Matrix{Float64}[]
        for i in groups[row]
            b=p.couplings[i]
            push!(columns,p.cols[b.col].coeff)
            S=b isa _PlanCoupling ? b.S : (b.R===nothing ? b.L : b.L*b.R')
            push!(blocks,S)
        end
        matrix=hcat(blocks...)
        if _release
            empty!(blocks)
            for i in groups[row];p.couplings[i]=_released_coupling(p.couplings[i]);end
            _released!(tracker,sizeof(matrix))
        end
        push!(packets,_CouplingPacket(matrix,p.rows[row].coeff,columns,zeros(size(matrix,2))))
    end
    neargroups=Dict{UnitRange{Int},Vector{Int}}();nearorder=UnitRange{Int}[]
    for (i,b) in enumerate(p.dense)
        haskey(neargroups,b.rows) || (neargroups[b.rows]=Int[];push!(nearorder,b.rows))
        push!(neargroups[b.rows],i)
    end
    nearpackets=_CouplingPacket[]
    for row in nearorder
        bs=p.dense[neargroups[row]]
        matrix=hcat((b.D for b in bs)...)
        if _release
            for i in neargroups[row];b=p.dense[i];p.dense[i]=_PlanDense(_RELEASED_BLOCK,b.rows,b.cols);end
            bs=nothing
            _released!(tracker,sizeof(matrix))
        end
        cols=[p.dense[i].cols for i in neargroups[row]]
        push!(nearpackets,_CouplingPacket(matrix,row,cols,zeros(size(matrix,2))))
    end
    sortedrows=sort(nearorder;by=first)
    near_parallel=all(last(sortedrows[i])<first(sortedrows[i+1]) for i in 1:length(sortedrows)-1)
    H2PacketMatvecPlan(p.shape,p.rows,p.cols,packets,nearpackets,near_parallel,
        zeros(length(p.rowcoeff)),zeros(length(p.colcoeff)),zeros(length(p.rowbuffer)),zeros(length(p.colbuffer)),
        p.rowperm,p.colperm,[zeros(length(p.colcoeff)) for _ in 1:workers],[zeros(length(p.colbuffer)) for _ in 1:workers])
end
_released_coupling(b::_PlanCoupling)=_PlanCoupling(_RELEASED_BLOCK,b.row,b.col)
_released_coupling(b::_LowRankPlanCoupling)=_LowRankPlanCoupling(_RELEASED_BLOCK,nothing,Float64[],b.row,b.col)
Base.size(p::H2PacketMatvecPlan)=p.shape
function _packet_forward!(rowcoeff,b::_CouplingPacket,colcoeff)
    offset=0
    for cr in b.columns
        for (i,j) in enumerate(cr);b.scratch[offset+i]=colcoeff[j];end
        offset+=length(cr)
    end
    mul!(view(rowcoeff,b.row),b.matrix,b.scratch,1.,1.)
end
function _packet_transpose!(colcoeff,b::_CouplingPacket,rowcoeff)
    mul!(b.scratch,b.matrix',view(rowcoeff,b.row))
    offset=0
    for cr in b.columns
        for (i,j) in enumerate(cr);colcoeff[j]+=b.scratch[offset+i];end
        offset+=length(cr)
    end
end
function _packet_worker!(p,worker,transposed)
    count=length(p.partials)
    for i in worker:count:length(p.packets)
        b=p.packets[i]
        transposed ? _packet_transpose!(p.partials[worker],b,p.rowcoeff) : _packet_forward!(p.rowcoeff,b,p.colcoeff)
    end
end
function _packet_interactions!(p,transposed)
    if length(p.partials)==1
        for b in p.packets
            transposed ? _packet_transpose!(p.colcoeff,b,p.rowcoeff) : _packet_forward!(p.rowcoeff,b,p.colcoeff)
        end
    else
        transposed && foreach(v->fill!(v,0.),p.partials)
        @sync for worker in eachindex(p.partials)
            Threads.@spawn _packet_worker!(p,worker,transposed)
        end
        if transposed
            for partial in p.partials
                for i in eachindex(p.colcoeff);p.colcoeff[i]+=partial[i];end
            end
        end
    end
end
function _near_packet_worker!(p,worker,t)
    input=t ? p.rowbuffer : p.colbuffer
    output=t ? p.nearpartials[worker] : p.rowbuffer
    for i in worker:length(p.partials):length(p.nearpackets)
        b=p.nearpackets[i]
        t ? _packet_transpose!(output,b,input) : _packet_forward!(output,b,input)
    end
end
function _near_packet_interactions!(p,t)
    input=t ? p.rowbuffer : p.colbuffer
    output=t ? p.colbuffer : p.rowbuffer
    if length(p.partials)==1 || (!t && !p.near_parallel)
        for b in p.nearpackets
            t ? _packet_transpose!(output,b,input) : _packet_forward!(output,b,input)
        end
    else
        t && foreach(v->fill!(v,0.),p.nearpartials)
        @sync for worker in eachindex(p.partials)
            Threads.@spawn _near_packet_worker!(p,worker,t)
        end
        if t
            for partial in p.nearpartials
                for i in eachindex(output);output[i]+=partial[i];end
            end
        end
    end
end
function _packet_mul!(y,p::H2PacketMatvecPlan,x,alpha,beta,t)
    inputnodes,outputnodes=t ? (p.rows,p.cols) : (p.cols,p.rows)
    inputcoeff,outputcoeff=t ? (p.rowcoeff,p.colcoeff) : (p.colcoeff,p.rowcoeff)
    input,output=t ? (p.rowbuffer,p.colbuffer) : (p.colbuffer,p.rowbuffer)
    ip,op=t ? (p.rowperm,p.colperm) : (p.colperm,p.rowperm)
    length(x)==length(input) && length(y)==length(output) || throw(DimensionMismatch("incompatible packet H2 matvec dimensions"))
    if iszero(alpha)
        iszero(beta) ? fill!(y,0.) : rmul!(y,beta)
        return y
    end
    for i in eachindex(input);input[i]=x[ip[i]];end
    fill!(output,0.);fill!(outputcoeff,0.)
    _plan_up!(inputcoeff,inputnodes,input)
    _packet_interactions!(p,t)
    _plan_down!(output,outputcoeff,outputnodes)
    _near_packet_interactions!(p,t)
    for i in eachindex(output)
        j=op[i];y[j]=iszero(beta) ? alpha*output[i] : alpha*output[i]+beta*y[j]
    end
    y
end
const TransposedPacketH2Plan=Union{Transpose{Float64,H2PacketMatvecPlan},Adjoint{Float64,H2PacketMatvecPlan}}
LinearAlgebra.mul!(y::AbstractVector,p::H2PacketMatvecPlan,x::AbstractVector,alpha::Number=1,beta::Number=0)=_packet_mul!(y,p,x,alpha,beta,false)
LinearAlgebra.mul!(y::AbstractVector,p::TransposedPacketH2Plan,x::AbstractVector,alpha::Number=1,beta::Number=0)=_packet_mul!(y,parent(p),x,alpha,beta,true)
Base.:*(p::Union{H2PacketMatvecPlan,TransposedPacketH2Plan},x::AbstractVector)=mul!(zeros(size(p,1)),p,x)
function storage_bytes(p::H2PacketMatvecPlan)
    sum((sizeof(n.V)+sizeof(n.E) for ns in (p.rows,p.cols) for n in ns);init=0)+
    sum((sizeof(b.matrix) for b in p.nearpackets);init=0)+sum((sizeof(b.matrix) for b in p.packets);init=0)
end
function Base.copy(p::H2PacketMatvecPlan)
    packets=[_CouplingPacket(b.matrix,b.row,b.columns,zeros(length(b.scratch))) for b in p.packets]
    nearpackets=[_CouplingPacket(b.matrix,b.row,b.columns,zeros(length(b.scratch))) for b in p.nearpackets]
    H2PacketMatvecPlan(p.shape,p.rows,p.cols,packets,nearpackets,p.near_parallel,
        zeros(length(p.rowcoeff)),zeros(length(p.colcoeff)),zeros(length(p.rowbuffer)),zeros(length(p.colbuffer)),p.rowperm,p.colperm,
        [zeros(length(v)) for v in p.partials],[zeros(length(v)) for v in p.nearpartials])
end
Base.show(io::IO,p::H2PacketMatvecPlan)=print(io,"H2PacketMatvecPlan(",size(p,1)," × ",size(p,2),", ",length(p.partials)," workers, ",storage_bytes(p)," numeric bytes)")
Base.show(io::IO,::MIME"text/plain",p::H2PacketMatvecPlan)=show(io,p)

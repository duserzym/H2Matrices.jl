# A packet applies `matrix * [segment_1; segment_2; ...]` to one row coefficient
# range. A direct segment gathers column coefficients; a factored segment
# (`factors[i]` nonempty, `direct[i]==false`) stores the right factor `R` of
# `S = L*R'` and contributes `R'*x` while `L` lives in `matrix`. An empty
# `factors` vector marks an all-direct packet.
# Mixed-precision couplings add Float32 and Float16 packet parts with the same
# layout, applied in Float64 arithmetic; Float16 segments carry column scales.
struct _LowPart{T}
    matrix::Matrix{T}
    columns::Vector{UnitRange{Int}}
    factors::Vector{Matrix{T}}          # 0x0: direct segment
    scales::Vector{Vector{Float64}}     # empty: unscaled part
    scratch::Vector{Float64}
end
_LowPart{T}() where {T}=_LowPart{T}(Matrix{T}(undef,0,0),UnitRange{Int}[],Matrix{T}[],Vector{Float64}[],Float64[])
_copy_lowpart(p::_LowPart{T}) where {T}=_LowPart{T}(p.matrix,p.columns,p.factors,p.scales,zeros(length(p.scratch)))
_lowpart_bytes(p::_LowPart)=sizeof(p.matrix)+sum(sizeof,p.factors;init=0)+sum(sizeof,p.scales;init=0)
struct _CouplingPacket
    matrix::Matrix{Float64}
    row::UnitRange{Int}
    columns::Vector{UnitRange{Int}}
    scratch::Vector{Float64}
    factors::Vector{Matrix{Float64}}
    p32::_LowPart{Float32}
    p16::_LowPart{Float16}
end
_CouplingPacket(matrix,row,columns,scratch,factors=Matrix{Float64}[])=
    _CouplingPacket(matrix,row,columns,scratch,factors,_LowPart{Float32}(),_LowPart{Float16}())
_copy_packet(b::_CouplingPacket)=_CouplingPacket(b.matrix,b.row,b.columns,zeros(length(b.scratch)),b.factors,
    _copy_lowpart(b.p32),_copy_lowpart(b.p16))
"""
    H2PacketMatvecPlan(h2; workers=1)
    H2PacketMatvecPlan(compact_plan; workers=1, keep_factors=true)

Pack interactions with the same row basis into contiguous GEMV packets.
Packing does not truncate data. Combined with implicit saturated bases, it
preserves the source operator up to floating-point rounding. Numeric blocks
are retained without the source operator. `workers>1` partitions packets
among tasks, including near-field packets with disjoint output ranges,
with separate transpose reduction buffers and deterministic
worker-order reduction. Use BLAS threads=1 when enabling packet workers.

Couplings factorized by `H2CompactMatvecPlan(h2; coupling_rtol)` keep their
factors (`S = L*R'`): left factors join the packet matrix and right factors
are applied per segment, so the packet stores exactly the compact plan's
numbers. `keep_factors=false` re-materializes `L*R'` instead. Float32 and
Float16 parts of mixed-precision couplings (`coupling_precision`) form
further packet parts that keep their storage precision and are applied in
Float64 arithmetic.

One plan is not safe for concurrent calls; `copy(plan)` shares numerical data
and allocates independent scratch for an additional caller. Source data must
remain unchanged while plans are used.
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
H2PacketMatvecPlan(h::H2Matrix;workers::Int=1)=H2PacketMatvecPlan(H2CompactMatvecPlan(h);workers)
function H2PacketMatvecPlan(p::H2CompactMatvecPlan;workers::Int=1,keep_factors::Bool=true)
    workers>0 || throw(ArgumentError("workers must be positive"))
    groups=Dict{Int,Vector{Int}}();order=Int[]
    for (i,b) in enumerate(p.couplings)
        isempty(p.rows[b.row].coeff) || isempty(p.cols[b.col].coeff) || begin
            haskey(groups,b.row) || (groups[b.row]=Int[];push!(order,b.row))
            push!(groups[b.row],i)
        end
    end
    packets=_CouplingPacket[]
    for row in order
        columns=UnitRange{Int}[];blocks=Matrix{Float64}[];factors=Matrix{Float64}[];anyfactor=false
        columns32=UnitRange{Int}[];blocks32=Matrix{Float32}[];factors32=Matrix{Float32}[]
        columns16=UnitRange{Int}[];blocks16=Matrix{Float16}[];factors16=Matrix{Float16}[];scales16=Vector{Float64}[]
        rc=p.rows[row].coeff
        for i in groups[row]
            b=p.couplings[i];cc=p.cols[b.col].coeff
            if b isa _PlanCoupling || b.R===nothing
                S=b isa _PlanCoupling ? b.S : b.L
                isempty(S) || (push!(columns,cc);push!(blocks,S);push!(factors,zeros(0,0)))
            elseif size(b.R,2)>0
                push!(columns,cc)
                if keep_factors
                    push!(blocks,b.L);push!(factors,b.R);anyfactor=true
                else
                    push!(blocks,b.L*b.R');push!(factors,zeros(0,0))
                end
            end
            b isa _MixedPlanCoupling || continue
            if b.R32===nothing
                isempty(b.L32) || (push!(columns32,cc);push!(blocks32,b.L32);push!(factors32,Matrix{Float32}(undef,0,0)))
            elseif size(b.R32,2)>0
                push!(columns32,cc);push!(blocks32,b.L32);push!(factors32,b.R32)
            end
            isempty(b.c16) || (push!(columns16,cc);push!(blocks16,b.L16);push!(factors16,b.R16);push!(scales16,b.c16))
        end
        matrix=isempty(blocks) ? zeros(length(rc),0) : reduce(hcat,blocks)
        p32=isempty(blocks32) ? _LowPart{Float32}() :
            (m32=reduce(hcat,blocks32);_LowPart{Float32}(m32,columns32,factors32,Vector{Float64}[],zeros(size(m32,2))))
        p16=isempty(blocks16) ? _LowPart{Float16}() :
            (m16=reduce(hcat,blocks16);_LowPart{Float16}(m16,columns16,factors16,scales16,zeros(size(m16,2))))
        push!(packets,_CouplingPacket(matrix,rc,columns,zeros(size(matrix,2)),anyfactor ? factors : Matrix{Float64}[],p32,p16))
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
        push!(nearpackets,_CouplingPacket(matrix,row,[b.cols for b in bs],zeros(size(matrix,2))))
    end
    sortedrows=sort(nearorder;by=first)
    near_parallel=all(last(sortedrows[i])<first(sortedrows[i+1]) for i in 1:length(sortedrows)-1)
    H2PacketMatvecPlan(p.shape,p.rows,p.cols,packets,nearpackets,near_parallel,
        zeros(length(p.rowcoeff)),zeros(length(p.colcoeff)),zeros(length(p.rowbuffer)),zeros(length(p.colbuffer)),
        p.rowperm,p.colperm,[zeros(length(p.colcoeff)) for _ in 1:workers],[zeros(length(p.colbuffer)) for _ in 1:workers])
end
Base.size(p::H2PacketMatvecPlan)=p.shape
function _packet_forward!(rowcoeff,b::_CouplingPacket,colcoeff)
    offset=0
    if isempty(b.factors)
        for cr in b.columns
            for (i,j) in enumerate(cr);b.scratch[offset+i]=colcoeff[j];end
            offset+=length(cr)
        end
    else
        for (s,cr) in enumerate(b.columns)
            R=b.factors[s]
            if size(R,1)==0
                for (i,j) in enumerate(cr);b.scratch[offset+i]=colcoeff[j];end
                offset+=length(cr)
            else
                r=size(R,2)
                r>0 && mul!(view(b.scratch,offset+1:offset+r),R',view(colcoeff,cr))
                offset+=r
            end
        end
    end
    size(b.matrix,2)>0 && mul!(view(rowcoeff,b.row),b.matrix,b.scratch,1.,1.)
    _lowpart_forward!(rowcoeff,b.row,b.p32,colcoeff)
    _lowpart_forward!(rowcoeff,b.row,b.p16,colcoeff)
    rowcoeff
end
function _lowpart_forward!(rowcoeff,row,p::_LowPart,colcoeff)
    isempty(p.columns) && return nothing
    offset=0;scaled=!isempty(p.scales)
    for (s,cr) in enumerate(p.columns)
        R=p.factors[s]
        if size(R,1)==0
            for (i,j) in enumerate(cr);p.scratch[offset+i]=colcoeff[j];end
            offset+=length(cr)
        else
            r=size(R,2)
            _mixed_tmul!(view(p.scratch,offset+1:offset+r),R,view(colcoeff,cr),false)
            if scaled
                c=p.scales[s]
                for i in 1:r;p.scratch[offset+i]*=c[i];end
            end
            offset+=r
        end
    end
    _mixed_mul!(view(rowcoeff,row),p.matrix,p.scratch)
    nothing
end
function _lowpart_transpose!(colcoeff,row,p::_LowPart,rowcoeff)
    isempty(p.columns) && return nothing
    _mixed_tmul!(p.scratch,p.matrix,view(rowcoeff,row),false)
    offset=0;scaled=!isempty(p.scales)
    for (s,cr) in enumerate(p.columns)
        R=p.factors[s]
        if size(R,1)==0
            for (i,j) in enumerate(cr);colcoeff[j]+=p.scratch[offset+i];end
            offset+=length(cr)
        else
            r=size(R,2)
            if scaled
                c=p.scales[s]
                for i in 1:r;p.scratch[offset+i]*=c[i];end
            end
            _mixed_mul!(view(colcoeff,cr),R,view(p.scratch,offset+1:offset+r))
            offset+=r
        end
    end
    nothing
end
function _packet_transpose!(colcoeff,b::_CouplingPacket,rowcoeff)
    _lowpart_transpose!(colcoeff,b.row,b.p32,rowcoeff)
    _lowpart_transpose!(colcoeff,b.row,b.p16,rowcoeff)
    size(b.matrix,2)>0 || return nothing
    mul!(b.scratch,b.matrix',view(rowcoeff,b.row))
    offset=0
    if isempty(b.factors)
        for cr in b.columns
            for (i,j) in enumerate(cr);colcoeff[j]+=b.scratch[offset+i];end
            offset+=length(cr)
        end
    else
        for (s,cr) in enumerate(b.columns)
            R=b.factors[s]
            if size(R,1)==0
                for (i,j) in enumerate(cr);colcoeff[j]+=b.scratch[offset+i];end
                offset+=length(cr)
            else
                r=size(R,2)
                r>0 && mul!(view(colcoeff,cr),R,view(b.scratch,offset+1:offset+r),1.,1.)
                offset+=r
            end
        end
    end
    nothing
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
    sum((sizeof(b.matrix) for b in p.nearpackets);init=0)+
    sum((sizeof(b.matrix)+sum(sizeof,b.factors;init=0)+_lowpart_bytes(b.p32)+_lowpart_bytes(b.p16) for b in p.packets);init=0)
end
function Base.copy(p::H2PacketMatvecPlan)
    packets=[_copy_packet(b) for b in p.packets]
    nearpackets=[_copy_packet(b) for b in p.nearpackets]
    H2PacketMatvecPlan(p.shape,p.rows,p.cols,packets,nearpackets,p.near_parallel,
        zeros(length(p.rowcoeff)),zeros(length(p.colcoeff)),zeros(length(p.rowbuffer)),zeros(length(p.colbuffer)),p.rowperm,p.colperm,
        [zeros(length(v)) for v in p.partials],[zeros(length(v)) for v in p.nearpartials])
end
Base.show(io::IO,p::H2PacketMatvecPlan)=print(io,"H2PacketMatvecPlan(",size(p,1)," × ",size(p,2),", ",length(p.partials)," workers, ",storage_bytes(p)," numeric bytes)")
Base.show(io::IO,::MIME"text/plain",p::H2PacketMatvecPlan)=show(io,p)

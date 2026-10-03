# Packet plan: contiguous interaction packets with explicit write ownership.
#
# Couplings are stored as row packets (all couplings of one row coefficient
# range side by side); near-field blocks are stored as column packets (all
# dense blocks meeting one elementary column interval stacked vertically).
# Each packet is applied by one long-column kernel in both directions:
#   forward  coupling: owned rowcoeff range        += M  * colcoeff segments
#   forward  near:     private slot                 = Q  * x[column interval]
#   adjoint  coupling: private slot                 = M' * rowcoeff range
#   adjoint  near:     owned column interval       += Q' * gathered row segments
# Slots are reduced in a fixed order by the downward-pass task that owns the
# destination entries. A product therefore runs three phases of independent
# tasks (upward pass with near-field packets, coupling packets, reduction plus
# downward pass; forward downward passes start when their subtree's last
# coupling packet completes), no two
# tasks of a phase write the same entries, and every task has a fixed
# evaluation order: products are deterministic and bitwise independent of the
# worker count and of the dynamic task assignment.
#
# A coupling packet stores its numbers in up to three parts by storage
# precision (Float64, and the Float32/Float16 tiers of mixed-precision
# couplings); every part is applied in Float64 arithmetic, so forward and
# adjoint products apply the same stored operator. A part segment is direct
# (the coupling block itself, or its Float64/Float32 dense part) or factored
# (`L*Diagonal(c)*R'`: `L` lives in the part matrix, `R` in `factors`, and the
# optional power-of-two scales `c` in `scales`). Each coupling owns `length` of
# its column range consecutive adjoint slots, whatever parts it has.
struct _PacketPart{T}
    matrix::Matrix{T}
    columns::Vector{UnitRange{Int}}     # column coefficient range of each segment
    offsets::Vector{Int}                # slot offset of each segment in its packet
    factors::Vector{Matrix{T}}          # empty: all segments direct; 0x0: direct segment
    scales::Vector{Vector{Float64}}     # empty: no segment scaled; empty vector: unscaled segment
end
_PacketPart{T}() where {T}=_PacketPart{T}(Matrix{T}(undef,0,0),UnitRange{Int}[],Int[],Matrix{T}[],Vector{Float64}[])
_part_bytes(q::_PacketPart)=sizeof(q.matrix)+sum(sizeof,q.factors;init=0)+sum(sizeof,q.scales;init=0)
_part_flops(q::_PacketPart)=length(q.matrix)+sum(length,q.factors;init=0)
struct _CouplingPacket
    matrix::Matrix{Float64}             # the Float64 part's matrix
    row::UnitRange{Int}
    columns::Vector{UnitRange{Int}}     # every coupling's column range, in slot order
    slot::Int
    width::Int                          # adjoint slots: total length of `columns`
    f64::_PacketPart{Float64}
    f32::_PacketPart{Float32}
    f16::_PacketPart{Float16}
    plain::Bool                         # Float64 direct segments only, one per coupling
end
struct _NearPacket
    matrix::Matrix{Float64}
    col::UnitRange{Int}
    rows::Vector{UnitRange{Int}}
    slot::Int
end
# dest[target+1:target+length] += slots[source+1:source+length]
struct _SlotRef
    source::Int
    target::Int
    length::Int
end
# Pre-order node lists of coefficient-bearing subtrees, plus the slot
# reductions that the subtree's owner applies before its downward pass.
struct _TreeJob
    nodes::UnitRange{Int}
    refs::UnitRange{Int}
end
struct _TreeSchedule
    order::Vector{Int}
    jobs::Vector{_TreeJob}
    refs::Vector{_SlotRef}
end
# Scratch for products with several right-hand sides; allocated on first use
# and grown to the largest block width used. Column-major `n × k` blocks.
mutable struct _MultiWorkspace
    k::Int
    rowcoeff::Vector{Float64}
    colcoeff::Vector{Float64}
    rowbuffer::Vector{Float64}
    colbuffer::Vector{Float64}
    slots::Vector{Float64}
    scratch::Vector{Vector{Float64}}
end
_MultiWorkspace()=_MultiWorkspace(0,Float64[],Float64[],Float64[],Float64[],Float64[],Vector{Float64}[])
"""
    H2PacketMatvecPlan(h2; workers=1, consume=false, keep_factors=true, compact_options...)
    H2PacketMatvecPlan(compact_plan; workers=1, keep_factors=true)

Pack interactions into contiguous packets applied by fused long-column
kernels. Couplings sharing a row coefficient range form row packets; dense
near-field blocks are split at elementary column intervals and stacked into
column packets. Packing does not truncate data. Combined with implicit
saturated bases, it preserves the source operator up to floating-point
rounding. Numeric blocks are retained without the source operator.

Products run three phases of independent tasks (upward pass together with
near-field packets, coupling packets, slot reduction plus downward pass) with
explicit ownership of every written entry, so no private reduction buffers are
needed. `workers>1` applies tasks with dynamic, cost-ordered scheduling;
every task has a fixed evaluation order, so products are deterministic and
bitwise independent of the worker count. Use BLAS threads=1 when enabling
packet workers.

`mul!(Y, plan, X)` and `mul!(Y, adjoint(plan), X)` with matrices apply several
right-hand sides while streaming the stored operator once.

From an H2 matrix, `compact_options` (`passthrough`, `coupling_rtol`,
`coupling_scale`, `coupling_precision`) are passed to
[`H2CompactMatvecPlan`](@ref), whose stored operator the packets then hold.
Couplings factorized by `coupling_rtol` keep their factors (`S = L*R'`): `L`
joins the packet matrix and `R'` is applied per segment, so the packets store
exactly the compact plan's numbers; `keep_factors=false` re-materializes
`L*R'` instead. Float32 and Float16 parts of mixed-precision couplings form
further packet parts that keep their storage precision and are applied in
Float64 arithmetic, so the adjoint stays the exact transpose of the stored
operator. Pass-through basis nodes copy coefficients in the upward and
downward passes.

One plan is not safe for concurrent calls; `copy(plan)` shares numerical data
and allocates independent scratch for an additional caller. Source data must
remain unchanged while plans are used.

`consume=true` (H2 input only) releases the source operator's coupling and
near-field blocks as they are transformed and packed (a near-field block once
its last column interval is packed; with `coupling_rtol`, each transformed
coupling once factorized), bounding peak construction memory by roughly one
operator plus one packet. The source `h2` is unusable afterwards (its leaves
hold no numerical blocks); the plan is bitwise identical to `consume=false`.
"""
struct H2PacketMatvecPlan <: AbstractMatrix{Float64}
    shape::Tuple{Int,Int}
    rows::Vector{_CompactBasisNode}
    cols::Vector{_CompactBasisNode}
    packets::Vector{_CouplingPacket}
    nearpackets::Vector{_NearPacket}
    tasks::Vector{Int}
    neartasks::Vector{Int}
    fwdtasks::Vector{Int}
    packetjob::Vector{Int}
    jobdeps::Vector{Int}
    rowup::_TreeSchedule
    colup::_TreeSchedule
    rowdown::_TreeSchedule
    coldown::_TreeSchedule
    workers::Int
    rowcoeff::Vector{Float64}
    colcoeff::Vector{Float64}
    rowbuffer::Vector{Float64}
    colbuffer::Vector{Float64}
    slots::Vector{Float64}
    scratch::Vector{Vector{Float64}}
    rowperm::Vector{Int}
    colperm::Vector{Int}
    counter::Threads.Atomic{Int}
    pending::Vector{Threads.Atomic{Int}}
    multi::_MultiWorkspace
end
function H2PacketMatvecPlan(h::H2Matrix;workers::Int=1,consume::Bool=false,keep_factors::Bool=true,compact_options...)
    workers>0 || throw(ArgumentError("workers must be positive"))
    # A consuming compact plan drops each source block once it holds it, so
    # each packed group can then be released in turn.
    core=H2CompactMatvecPlan(h;consume,compact_options...)
    consume && _release_h2_blocks!(h)
    H2PacketMatvecPlan(core;workers,keep_factors,_release=consume)
end
function _release_h2_blocks!(h::H2Matrix)
    if isleaf(h)
        h.uniform=nothing;h.dense=nothing
    else
        foreach(_release_h2_blocks!,h.children)
    end
    h
end
_node_cost(n::_CompactBasisNode)=length(n.E)+length(n.V)+(n.identity || n.passthrough ? length(n.coeff) : 0)
# Coefficient-bearing subtrees (top nodes: nonempty coefficients and no such
# ancestor). Their coefficient ranges and physical index ranges are disjoint.
function _tree_tops(nodes::Vector{_CompactBasisNode})
    order=Int[];spans=UnitRange{Int}[]
    isempty(nodes) && return order,spans
    ischild=falses(length(nodes))
    for n in nodes, c in n.children;ischild[c]=true;end
    function subtree!(i)
        push!(order,i)
        foreach(subtree!,nodes[i].children)
    end
    function tops!(i)
        if isempty(nodes[i].coeff)
            foreach(tops!,nodes[i].children)
        else
            start=length(order)+1;subtree!(i);push!(spans,start:length(order))
        end
    end
    foreach(tops!,findall(!,ischild))
    order,spans
end
# Assemble a schedule: `refs[k]` are the reductions owned by subtree `k`;
# `extra` are reductions outside every subtree, each becoming its own job.
function _tree_schedule(nodes,order,spans,refs::Vector{Vector{_SlotRef}},extra::Vector{Vector{_SlotRef}})
    flat=_SlotRef[];jobs=_TreeJob[];costs=Int[]
    for (k,span) in enumerate(spans)
        start=length(flat)+1;append!(flat,refs[k])
        push!(jobs,_TreeJob(span,start:length(flat)))
        push!(costs,sum(_node_cost(nodes[order[t]]) for t in span)+sum((r.length for r in refs[k]);init=0))
    end
    for g in extra
        isempty(g) && continue
        start=length(flat)+1;append!(flat,g)
        push!(jobs,_TreeJob(1:0,start:length(flat)));push!(costs,sum(r.length for r in g))
    end
    _TreeSchedule(order,jobs[sortperm(costs;rev=true)],flat)
end
function H2PacketMatvecPlan(p::H2CompactMatvecPlan;workers::Int=1,keep_factors::Bool=true,_release::Bool=false)
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
    packets=_CouplingPacket[];slot=0
    for row in order
        packet=_build_coupling_packet(p,groups[row],p.rows[row].coeff,slot,keep_factors)
        if _release
            for i in groups[row];p.couplings[i]=_released_coupling(p.couplings[i]);end
            _released!(tracker,_packet_bytes(packet))
        end
        push!(packets,packet);slot+=packet.width
    end
    rowslots=slot
    rorder,rspans=_tree_tops(p.rows);corder,cspans=_tree_tops(p.cols)
    # Near field: split dense blocks at elementary column intervals and stack
    # all pieces of one interval. Row segments are split at elementary row
    # intervals (including row-subtree boundaries) only for slot reduction.
    keep=[i for (i,b) in enumerate(p.dense) if !isempty(b.rows) && !isempty(b.cols)]
    dense=p.dense[keep]
    cb=sort!(unique!(vcat(Int[first(b.cols) for b in dense],Int[last(b.cols)+1 for b in dense])))
    colgroups=[Int[] for _ in 1:max(length(cb)-1,0)]
    for (i,b) in enumerate(dense)
        for k in searchsortedfirst(cb,first(b.cols)):searchsortedfirst(cb,last(b.cols)+1)-1
            push!(colgroups[k],i)
        end
    end
    rowtop=zeros(Int,size(p,1))
    for (k,span) in enumerate(rspans);rowtop[p.rows[rorder[first(span)]].indices].=k;end
    rb=sort!(unique!(vcat(Int[first(b.rows) for b in dense],Int[last(b.rows)+1 for b in dense],
        Int[first(p.rows[rorder[first(s)]].indices) for s in rspans],Int[last(p.rows[rorder[first(s)]].indices)+1 for s in rspans])))
    bandrefs=[_SlotRef[] for _ in 1:max(length(rb)-1,0)]
    nearpackets=_NearPacket[];slot=0
    remaining=zeros(Int,length(dense))
    for g in colgroups, i in g;remaining[i]+=1;end
    for (k,g) in enumerate(colgroups)
        isempty(g) && continue
        e=cb[k]:cb[k+1]-1
        # Views: the column pieces are copied once, straight into the packet.
        matrix=reduce(vcat,[view(dense[i].D,:,e.-(first(dense[i].cols)-1)) for i in g])
        rows=[dense[i].rows for i in g]
        offset=slot
        for r in rows
            for j in searchsortedfirst(rb,first(r)):searchsortedfirst(rb,last(r)+1)-1
                band=rb[j]:rb[j+1]-1
                push!(bandrefs[j],_SlotRef(offset+first(band)-first(r),first(band)-1,length(band)))
            end
            offset+=length(r)
        end
        push!(nearpackets,_NearPacket(matrix,e,rows,slot));slot+=size(matrix,1)
        # A dense block is released once its last column interval is packed.
        if _release
            for i in g
                (remaining[i]-=1)==0 || continue
                d=dense[i];_released!(tracker,sizeof(d.D))
                dense[i]=p.dense[keep[i]]=_PlanDense(_RELEASED_BLOCK,d.rows,d.cols)
            end
        end
    end
    rowrefs=[_SlotRef[] for _ in rspans];extra=Vector{_SlotRef}[]
    for (j,g) in enumerate(bandrefs)
        isempty(g) && continue
        k=rowtop[rb[j]]
        k==0 ? push!(extra,g) : append!(rowrefs[k],g)
    end
    # Adjoint coupling slots are reduced by the column subtree owning the range.
    coltop=Dict{Int,Int}()
    for (k,span) in enumerate(cspans), t in span
        n=p.cols[corder[t]];isempty(n.coeff) || (coltop[first(n.coeff)]=k)
    end
    colrefs=[_SlotRef[] for _ in cspans]
    for b in packets
        offset=b.slot
        for cr in b.columns
            push!(colrefs[coltop[first(cr)]],_SlotRef(offset,first(cr)-1,length(cr)));offset+=length(cr)
        end
    end
    none=[_SlotRef[] for _ in rspans];cnone=[_SlotRef[] for _ in cspans]
    # Near-field packets need no upward pass; they share the first phase with it.
    tasks=sortperm([_packet_flops(b) for b in packets];rev=true)
    neartasks=.-sortperm([length(b.matrix) for b in nearpackets];rev=true)
    # Worker scratch: gathered near-field rows (adjoint) or the inner vector
    # of a factored coupling segment.
    maxrows=max(maximum((size(b.matrix,1) for b in nearpackets);init=0),maximum(_max_factor_rank,packets;init=0))
    rowdown=_tree_schedule(p.rows,rorder,rspans,rowrefs,extra)
    # Forward products start the downward pass of a row subtree as soon as its
    # last coupling packet completes. Subtrees with the costliest downward
    # passes come first, so those passes overlap with the remaining packets.
    rownode=Dict{Int,Int}()
    for (k,job) in enumerate(rowdown.jobs), t in job.nodes
        n=p.rows[rowdown.order[t]];isempty(n.coeff) || (rownode[first(n.coeff)]=k)
    end
    packetjob=[rownode[first(b.row)] for b in packets]
    jobdeps=zeros(Int,length(rowdown.jobs));foreach(k->jobdeps[k]+=1,packetjob)
    jobcost=[sum((length(p.rows[rowdown.order[t]].E) for t in job.nodes);init=0) for job in rowdown.jobs]
    fwdtasks=sort(eachindex(packets);by=k->(-jobcost[packetjob[k]],packetjob[k],-_packet_flops(packets[k])))
    append!(fwdtasks,.-findall(iszero,jobdeps))
    H2PacketMatvecPlan(p.shape,p.rows,p.cols,packets,nearpackets,tasks,neartasks,fwdtasks,packetjob,jobdeps,
        _tree_schedule(p.rows,rorder,rspans,none,Vector{_SlotRef}[]),_tree_schedule(p.cols,corder,cspans,cnone,Vector{_SlotRef}[]),
        rowdown,_tree_schedule(p.cols,corder,cspans,colrefs,Vector{_SlotRef}[]),
        workers,zeros(length(p.rowcoeff)),zeros(length(p.colcoeff)),zeros(length(p.rowbuffer)),zeros(length(p.colbuffer)),
        zeros(max(rowslots,slot)),[zeros(maxrows) for _ in 1:workers],p.rowperm,p.colperm,Threads.Atomic{Int}(0),
        [Threads.Atomic{Int}(0) for _ in jobdeps],_MultiWorkspace())
end
_released_coupling(b::_PlanCoupling)=_PlanCoupling(_RELEASED_BLOCK,b.row,b.col)
_released_coupling(b::_LowRankPlanCoupling)=_LowRankPlanCoupling(_RELEASED_BLOCK,nothing,Float64[],b.row,b.col)
_released_coupling(b::_MixedPlanCoupling)=_MixedPlanCoupling(_RELEASED_BLOCK,nothing,_NO32,nothing,Float64[],_NO16,_NO16,Float64[],Float64[],b.row,b.col)
const _DIRECT64=zeros(Float64,0,0)
const _DIRECT32=zeros(Float32,0,0)
const _DIRECT16=zeros(Float16,0,0)
const _UNSCALED=Float64[]
_direct_marker(::Type{Float64})=_DIRECT64
_direct_marker(::Type{Float32})=_DIRECT32
_direct_marker(::Type{Float16})=_DIRECT16
# Segments of one storage precision, collected while a packet is built.
struct _PartBuilder{T}
    blocks::Vector{Matrix{T}}
    columns::Vector{UnitRange{Int}}
    offsets::Vector{Int}
    factors::Vector{Matrix{T}}
    scales::Vector{Vector{Float64}}
end
_PartBuilder{T}() where {T}=_PartBuilder{T}(Matrix{T}[],UnitRange{Int}[],Int[],Matrix{T}[],Vector{Float64}[])
function _add_segment!(q::_PartBuilder{T},block,cr,offset,R=nothing,c=_UNSCALED) where {T}
    push!(q.blocks,block);push!(q.columns,cr);push!(q.offsets,offset)
    push!(q.factors,R===nothing ? _direct_marker(T) : R);push!(q.scales,c)
    q
end
function _finish_part(q::_PartBuilder{T}) where {T}
    isempty(q.blocks) && return _PacketPart{T}()
    factors=all(R->R===_direct_marker(T),q.factors) ? Matrix{T}[] : q.factors
    scales=all(isempty,q.scales) ? Vector{Float64}[] : q.scales
    _PacketPart{T}(reduce(hcat,q.blocks),q.columns,q.offsets,factors,scales)
end
# Float64 part of a coupling: dense `S`/`L`, or factors `L*R'` (kept as factors
# with `keep_factors`, else multiplied out).
function _add_f64!(q,b::_PlanCoupling,cr,offset,keep_factors)
    _add_segment!(q,b.S,cr,offset)
end
function _add_f64!(q,b::Union{_LowRankPlanCoupling,_MixedPlanCoupling},cr,offset,keep_factors)
    if b.R===nothing
        isempty(b.L) || _add_segment!(q,b.L,cr,offset)
    elseif size(b.R,2)>0
        keep_factors ? _add_segment!(q,b.L,cr,offset,b.R) : _add_segment!(q,b.L*b.R',cr,offset)
    end
    q
end
_add_low!(q32,q16,b,cr,offset)=nothing
function _add_low!(q32,q16,b::_MixedPlanCoupling,cr,offset)
    if b.R32===nothing
        isempty(b.L32) || _add_segment!(q32,b.L32,cr,offset)
    elseif size(b.R32,2)>0
        _add_segment!(q32,b.L32,cr,offset,b.R32,isempty(b.c32) ? _UNSCALED : b.c32)
    end
    isempty(b.c16) || _add_segment!(q16,b.L16,cr,offset,b.R16,b.c16)
    nothing
end
# One row packet from the couplings `group` of row coefficient range `rc`.
function _build_coupling_packet(p::H2CompactMatvecPlan,group,rc,slot,keep_factors)
    columns=UnitRange{Int}[];width=0
    q64=_PartBuilder{Float64}();q32=_PartBuilder{Float32}();q16=_PartBuilder{Float16}()
    for i in group
        b=p.couplings[i];cr=p.cols[b.col].coeff
        push!(columns,cr)
        _add_f64!(q64,b,cr,width,keep_factors)
        _add_low!(q32,q16,b,cr,width)
        width+=length(cr)
    end
    f64=_finish_part(q64);f32=_finish_part(q32);f16=_finish_part(q16)
    empty!(q64.blocks);empty!(q32.blocks);empty!(q16.blocks)
    plain=isempty(f32.columns) && isempty(f16.columns) && isempty(f64.factors) && length(f64.columns)==length(columns)
    _CouplingPacket(f64.matrix,rc,columns,slot,width,f64,f32,f16,plain)
end
_packet_bytes(b::_CouplingPacket)=_part_bytes(b.f64)+_part_bytes(b.f32)+_part_bytes(b.f16)
_packet_flops(b::_CouplingPacket)=_part_flops(b.f64)+_part_flops(b.f32)+_part_flops(b.f16)
# Largest factor rank (worker scratch for factored segments).
_max_factor_rank(b::_CouplingPacket)=max(_max_factor_rank(b.f64),_max_factor_rank(b.f32),_max_factor_rank(b.f16))
_max_factor_rank(q::_PacketPart)=maximum((size(R,2) for R in q.factors);init=0)
Base.size(p::H2PacketMatvecPlan)=p.shape

# Kernels on column-major `M` with `m` rows; `c0` is a 0-based column offset.
# y[y0+1:y0+m] += M[:, c0+1:c0+n] * x[x0+1:x0+n]
@inline function _kernel_n!(y,y0,M,c0,n,x,x0)
    m=size(M,1);j=0
    @inbounds while j+4<=n
        b=(c0+j)*m
        x1=x[x0+j+1];x2=x[x0+j+2];x3=x[x0+j+3];x4=x[x0+j+4]
        @simd for i in 1:m
            y[y0+i]=muladd(M[b+i],x1,muladd(M[b+m+i],x2,muladd(M[b+2m+i],x3,muladd(M[b+3m+i],x4,y[y0+i]))))
        end
        j+=4
    end
    @inbounds while j<n
        b=(c0+j)*m;xj=x[x0+j+1]
        @simd for i in 1:m
            y[y0+i]=muladd(M[b+i],xj,y[y0+i])
        end
        j+=1
    end
    nothing
end
# x[x0+1:x0+n] += M[:, c0+1:c0+n]' * y[y0+1:y0+m]
@inline function _kernel_t!(x,x0,M,c0,n,y,y0)
    m=size(M,1)
    @inbounds for j in 1:n
        b=(c0+j-1)*m;s=0.
        @simd for i in 1:m
            s=muladd(M[b+i],y[y0+i],s)
        end
        x[x0+j]+=s
    end
    nothing
end
function _up_job!(coeff,nodes,x,tree::_TreeSchedule,k)
    order=tree.order
    @inbounds for t in reverse(tree.jobs[k].nodes)
        n=nodes[order[t]];isempty(n.coeff) && continue
        c0=first(n.coeff)-1
        if n.identity
            i0=first(n.indices)-1
            for i in 1:length(n.coeff);coeff[c0+i]=x[i0+i];end
        elseif n.passthrough
            # Concatenated child coefficients.
            for j in n.children
                child=nodes[j];isempty(child.coeff) && continue
                d0=first(child.coeff)-1
                for i in 1:length(child.coeff);coeff[c0+i]=coeff[d0+i];end
                c0+=length(child.coeff)
            end
        else
            for i in n.coeff;coeff[i]=0.;end
            if isempty(n.children)
                _kernel_t!(coeff,c0,n.V,0,size(n.V,2),x,first(n.indices)-1)
            else
                for j in n.children
                    child=nodes[j];isempty(child.coeff) && continue
                    _kernel_t!(coeff,c0,child.E,0,size(child.E,2),coeff,first(child.coeff)-1)
                end
            end
        end
    end
end
# Reductions owned by the job (into `dest`), then the job's downward pass.
function _down_job!(y,coeff,dest,slots,nodes,tree::_TreeSchedule,k)
    job=tree.jobs[k];order=tree.order
    @inbounds for t in job.refs
        r=tree.refs[t]
        @simd for i in 1:r.length;dest[r.target+i]+=slots[r.source+i];end
    end
    @inbounds for t in job.nodes
        n=nodes[order[t]];isempty(n.coeff) && continue
        c0=first(n.coeff)-1
        if n.identity
            i0=first(n.indices)-1
            @simd for i in 1:length(n.coeff);y[i0+i]+=coeff[c0+i];end
        elseif n.passthrough
            for j in n.children
                child=nodes[j];isempty(child.coeff) && continue
                d0=first(child.coeff)-1
                @simd for i in 1:length(child.coeff);coeff[d0+i]+=coeff[c0+i];end
                c0+=length(child.coeff)
            end
        elseif isempty(n.children)
            _kernel_n!(y,first(n.indices)-1,n.V,0,size(n.V,2),coeff,c0)
        else
            for j in n.children
                child=nodes[j];isempty(child.coeff) && continue
                _kernel_n!(coeff,first(child.coeff)-1,child.E,0,size(child.E,2),coeff,c0)
            end
        end
    end
end
# Forward: rowcoeff[row] += part * segment inputs. `z` is worker scratch.
function _part_forward!(rowcoeff,y0,q::_PacketPart,colcoeff,z)
    M=q.matrix;offset=0;direct=isempty(q.factors);scaled=!isempty(q.scales)
    @inbounds for (s,cr) in enumerate(q.columns)
        R=direct ? q.matrix : q.factors[s]
        if direct || size(R,1)==0
            _kernel_n!(rowcoeff,y0,M,offset,length(cr),colcoeff,first(cr)-1)
            offset+=length(cr)
        else
            r=size(R,2)
            for i in 1:r;z[i]=0.;end
            _kernel_t!(z,0,R,0,r,colcoeff,first(cr)-1)
            if scaled
                c=q.scales[s]
                isempty(c) || (for i in 1:r;z[i]*=c[i];end)
            end
            _kernel_n!(rowcoeff,y0,M,offset,r,z,0)
            offset+=r
        end
    end
    nothing
end
# Adjoint: slots[s0 + segment offset .+ (1:length(cr))] += (part segment)' * rowcoeff[row].
function _part_adjoint!(slots,s0,q::_PacketPart,rowcoeff,y0,z)
    M=q.matrix;offset=0;direct=isempty(q.factors);scaled=!isempty(q.scales)
    @inbounds for (s,cr) in enumerate(q.columns)
        o=s0+q.offsets[s]
        R=direct ? q.matrix : q.factors[s]
        if direct || size(R,1)==0
            _kernel_t!(slots,o,M,offset,length(cr),rowcoeff,y0)
            offset+=length(cr)
        else
            r=size(R,2)
            for i in 1:r;z[i]=0.;end
            _kernel_t!(z,0,M,offset,r,rowcoeff,y0)
            if scaled
                c=q.scales[s]
                isempty(c) || (for i in 1:r;z[i]*=c[i];end)
            end
            _kernel_n!(slots,o,R,0,r,z,0)
            offset+=r
        end
    end
    nothing
end
function _interaction_task!(p,k,w,t)
    if k>0
        b=p.packets[k];M=b.matrix;y0=first(b.row)-1
        if t
            s=p.slots;s0=b.slot
            @inbounds for j in 1:b.width;s[s0+j]=0.;end
            if b.plain
                _kernel_t!(s,s0,M,0,size(M,2),p.rowcoeff,y0)
            else
                z=p.scratch[w]
                _part_adjoint!(s,s0,b.f64,p.rowcoeff,y0,z)
                _part_adjoint!(s,s0,b.f32,p.rowcoeff,y0,z)
                _part_adjoint!(s,s0,b.f16,p.rowcoeff,y0,z)
            end
        elseif b.plain
            offset=0
            for cr in b.columns
                _kernel_n!(p.rowcoeff,y0,M,offset,length(cr),p.colcoeff,first(cr)-1)
                offset+=length(cr)
            end
        else
            z=p.scratch[w]
            _part_forward!(p.rowcoeff,y0,b.f64,p.colcoeff,z)
            _part_forward!(p.rowcoeff,y0,b.f32,p.colcoeff,z)
            _part_forward!(p.rowcoeff,y0,b.f16,p.colcoeff,z)
        end
    else
        b=p.nearpackets[-k];M=b.matrix
        if t
            g=p.scratch[w];offset=0
            for r in b.rows
                copyto!(g,offset+1,p.rowbuffer,first(r),length(r));offset+=length(r)
            end
            _kernel_t!(p.colbuffer,first(b.col)-1,M,0,size(M,2),g,0)
        else
            s=p.slots;s0=b.slot
            @inbounds for i in 1:size(M,1);s[s0+i]=0.;end
            _kernel_n!(s,s0,M,0,size(M,2),p.colbuffer,first(b.col)-1)
        end
    end
end
_arm!(p::H2PacketMatvecPlan)=(for (c,d) in zip(p.pending,p.jobdeps);c[]=d;end)
# Forward coupling packet, then the downward pass of its row subtree if this
# was the subtree's last packet (the completing worker runs it; no waiting).
# Entries `-j` run subtree `j`, which has no coupling packets, directly.
function _forward_task!(p,i,w,down::F) where {F}
    k=p.fwdtasks[i]
    k<0 && return down(-k)
    _interaction_task!(p,k,w,false)
    j=p.packetjob[k]
    Threads.atomic_sub!(p.pending[j],1)==1 && down(j)
    nothing
end
# Apply f(1,w),...,f(n,w) with up to p.workers tasks `w`; tasks claim work
# dynamically in the precomputed cost-descending order. The caller is worker 1.
function _task_loop(f::F,counter,n,w) where {F}
    while true
        i=Threads.atomic_add!(counter,1)+1
        i>n && return nothing
        f(i,w)
    end
end
function _run_tasks!(f::F,p::H2PacketMatvecPlan,n) where {F}
    w=min(p.workers,n)
    if w<=1
        for i in 1:n;f(i,1);end
    else
        counter=p.counter;counter[]=0
        @sync begin
            for k in 2:w;Threads.@spawn _task_loop(f,counter,n,k);end
            _task_loop(f,counter,n,1)
        end
    end
    nothing
end
function _packet_mul!(y,p::H2PacketMatvecPlan,x,alpha,beta,t)
    inputnodes,outputnodes=t ? (p.rows,p.cols) : (p.cols,p.rows)
    up,down=t ? (p.rowup,p.coldown) : (p.colup,p.rowdown)
    inputcoeff,outputcoeff=t ? (p.rowcoeff,p.colcoeff) : (p.colcoeff,p.rowcoeff)
    input,output=t ? (p.rowbuffer,p.colbuffer) : (p.colbuffer,p.rowbuffer)
    reduced=t ? outputcoeff : output
    ip,op=t ? (p.rowperm,p.colperm) : (p.colperm,p.rowperm)
    length(x)==length(input) && length(y)==length(output) || throw(DimensionMismatch("incompatible packet H2 matvec dimensions"))
    if iszero(alpha)
        iszero(beta) ? fill!(y,0.) : rmul!(y,beta)
        return y
    end
    @inbounds for i in eachindex(input);input[i]=x[ip[i]];end
    fill!(output,0.);fill!(outputcoeff,0.)
    nup=length(up.jobs)
    _run_tasks!((k,w)->k<=nup ? _up_job!(inputcoeff,inputnodes,input,up,k) : _interaction_task!(p,p.neartasks[k-nup],w,t),p,nup+length(p.neartasks))
    if t
        _run_tasks!((k,w)->_interaction_task!(p,p.tasks[k],w,t),p,length(p.tasks))
        _run_tasks!((k,w)->_down_job!(output,outputcoeff,reduced,p.slots,outputnodes,down,k),p,length(down.jobs))
    else
        _arm!(p)
        _run_tasks!((k,w)->_forward_task!(p,k,w,(j)->_down_job!(output,outputcoeff,reduced,p.slots,outputnodes,down,j)),p,length(p.fwdtasks))
    end
    @inbounds for i in eachindex(output)
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
    sum((sizeof(b.matrix) for b in p.nearpackets);init=0)+sum(_packet_bytes,p.packets;init=0)
end
function Base.copy(p::H2PacketMatvecPlan)
    H2PacketMatvecPlan(p.shape,p.rows,p.cols,p.packets,p.nearpackets,p.tasks,p.neartasks,p.fwdtasks,p.packetjob,p.jobdeps,p.rowup,p.colup,p.rowdown,p.coldown,p.workers,
        zeros(length(p.rowcoeff)),zeros(length(p.colcoeff)),zeros(length(p.rowbuffer)),zeros(length(p.colbuffer)),
        zeros(length(p.slots)),[zeros(length(s)) for s in p.scratch],p.rowperm,p.colperm,Threads.Atomic{Int}(0),
        [Threads.Atomic{Int}(0) for _ in p.pending],_MultiWorkspace())
end
Base.show(io::IO,p::H2PacketMatvecPlan)=print(io,"H2PacketMatvecPlan(",size(p,1)," × ",size(p,2),", ",p.workers," workers, ",storage_bytes(p)," numeric bytes)")
Base.show(io::IO,::MIME"text/plain",p::H2PacketMatvecPlan)=show(io,p)

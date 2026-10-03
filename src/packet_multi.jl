# Products with several right-hand sides. They use the same phases, tasks and
# write ownership as single-vector packet products, applied to column-major
# blocks of up to `_MULTI_BLOCK` vectors: each stored packet is streamed once
# per block and reused from cache by register-blocked kernels.
const _MULTI_BLOCK=16

# Register blocks of C matrix columns × V right-hand sides.
# Y[o+i+(v-1)ly] += Σ_c M[b+(c-1)m+i] X[q+c+(v-1)lx]
@generated function _nkb!(Y,o,ly,M,b,m,X,q,lx,::Val{C},::Val{V}) where {C,V}
    xs=[:($(Symbol(:x_,c,:_,v))=X[q+$c+$(v-1)*lx]) for c in 1:C for v in 1:V]
    ms=[:($(Symbol(:m_,c))=M[b+$(c-1)*m+i]) for c in 1:C]
    body=Expr[]
    for v in 1:V
        idx=:(o+$(v-1)*ly+i);ex=:(Y[$idx])
        for c in C:-1:1;ex=:(muladd($(Symbol(:m_,c)),$(Symbol(:x_,c,:_,v)),$ex));end
        push!(body,:(Y[$idx]=$ex))
    end
    quote
        @inbounds begin
            $(xs...)
            @simd for i in 1:m
                $(ms...)
                $(body...)
            end
        end
        nothing
    end
end
# S[s+c+(v-1)ls] += Σ_i M[b+(c-1)m+i] Y[o+i+(v-1)ly]
@generated function _tkb!(S,s,ls,M,b,m,Y,o,ly,::Val{C},::Val{V}) where {C,V}
    init=[:($(Symbol(:a_,c,:_,v))=0.) for c in 1:C for v in 1:V]
    ms=[:($(Symbol(:m_,c))=M[b+$(c-1)*m+i]) for c in 1:C]
    ys=[:($(Symbol(:y_,v))=Y[o+$(v-1)*ly+i]) for v in 1:V]
    fm=[:($(Symbol(:a_,c,:_,v))=muladd($(Symbol(:m_,c)),$(Symbol(:y_,v)),$(Symbol(:a_,c,:_,v)))) for c in 1:C for v in 1:V]
    st=[:(S[s+$c+$(v-1)*ls]+=$(Symbol(:a_,c,:_,v))) for c in 1:C for v in 1:V]
    quote
        $(init...)
        @inbounds @simd for i in 1:m
            $(ms...)
            $(ys...)
            $(fm...)
        end
        @inbounds begin
            $(st...)
        end
        nothing
    end
end
# Split the remaining vectors into register blocks of 4, 3 or 2 (e.g. 9=4+3+2).
@inline _vblock(r)=(r>=8 || r==4 || r==7) ? 4 : r>=5 ? 3 : r
@inline function _nk_call!(Y,o,ly,M,b,m,X,q,lx,c::Val,v)
    v==4 ? _nkb!(Y,o,ly,M,b,m,X,q,lx,c,Val(4)) : v==3 ? _nkb!(Y,o,ly,M,b,m,X,q,lx,c,Val(3)) :
    v==2 ? _nkb!(Y,o,ly,M,b,m,X,q,lx,c,Val(2)) : _nkb!(Y,o,ly,M,b,m,X,q,lx,c,Val(1))
end
@inline function _tk_call!(S,s,ls,M,b,m,Y,o,ly,c::Val,v)
    v==4 ? _tkb!(S,s,ls,M,b,m,Y,o,ly,c,Val(4)) : v==3 ? _tkb!(S,s,ls,M,b,m,Y,o,ly,c,Val(3)) :
    v==2 ? _tkb!(S,s,ls,M,b,m,Y,o,ly,c,Val(2)) : _tkb!(S,s,ls,M,b,m,Y,o,ly,c,Val(1))
end
# Y[y0+1:y0+m, v] += M[:, c0+1:c0+n] * X[x0+1:x0+n, v] for v=1:K (column strides ly, lx)
function _kernel_nk!(Y,y0,ly,M,c0,n,X,x0,lx,K)
    m=size(M,1);j=0
    while j+4<=n
        b=(c0+j)*m;v=0
        while v<K
            c=_vblock(K-v);_nk_call!(Y,y0+v*ly,ly,M,b,m,X,x0+j+v*lx,lx,Val(4),c);v+=c
        end
        j+=4
    end
    while j<n
        b=(c0+j)*m;v=0
        while v<K
            c=_vblock(K-v);_nk_call!(Y,y0+v*ly,ly,M,b,m,X,x0+j+v*lx,lx,Val(1),c);v+=c
        end
        j+=1
    end
    nothing
end
# S[s0+1:s0+n, v] += M[:, c0+1:c0+n]' * Y[y0+1:y0+m, v] for v=1:K
function _kernel_tk!(S,s0,ls,M,c0,n,Y,y0,ly,K)
    m=size(M,1);j=0
    while j+4<=n
        b=(c0+j)*m;v=0
        while v<K
            c=_vblock(K-v);_tk_call!(S,s0+j+v*ls,ls,M,b,m,Y,y0+v*ly,ly,Val(4),c);v+=c
        end
        j+=4
    end
    while j<n
        b=(c0+j)*m;v=0
        while v<K
            c=_vblock(K-v);_tk_call!(S,s0+j+v*ls,ls,M,b,m,Y,y0+v*ly,ly,Val(1),c);v+=c
        end
        j+=1
    end
    nothing
end
function _multi_workspace!(p::H2PacketMatvecPlan,k)
    w=p.multi
    if w.k<k
        w.rowcoeff=zeros(length(p.rowcoeff)*k);w.colcoeff=zeros(length(p.colcoeff)*k)
        w.rowbuffer=zeros(length(p.rowbuffer)*k);w.colbuffer=zeros(length(p.colbuffer)*k)
        w.slots=zeros(length(p.slots)*k);w.scratch=[zeros(length(s)*k) for s in p.scratch]
        w.k=k
    end
    w
end
function _up_job_k!(coeff,lc,nodes,x,lx,tree::_TreeSchedule,k,K)
    order=tree.order
    @inbounds for t in reverse(tree.jobs[k].nodes)
        n=nodes[order[t]];isempty(n.coeff) && continue
        c0=first(n.coeff)-1;len=length(n.coeff)
        if n.identity
            i0=first(n.indices)-1
            for v in 0:K-1, i in 1:len;coeff[c0+i+v*lc]=x[i0+i+v*lx];end
        elseif n.passthrough
            for j in n.children
                child=nodes[j];isempty(child.coeff) && continue
                d0=first(child.coeff)-1
                for v in 0:K-1, i in 1:length(child.coeff);coeff[c0+i+v*lc]=coeff[d0+i+v*lc];end
                c0+=length(child.coeff)
            end
        else
            for v in 0:K-1, i in 1:len;coeff[c0+i+v*lc]=0.;end
            if isempty(n.children)
                _kernel_tk!(coeff,c0,lc,n.V,0,size(n.V,2),x,first(n.indices)-1,lx,K)
            else
                for j in n.children
                    child=nodes[j];isempty(child.coeff) && continue
                    _kernel_tk!(coeff,c0,lc,child.E,0,size(child.E,2),coeff,first(child.coeff)-1,lc,K)
                end
            end
        end
    end
end
function _down_job_k!(y,ly,coeff,lc,dest,ld,slots,ls,nodes,tree::_TreeSchedule,k,K)
    job=tree.jobs[k];order=tree.order
    @inbounds for t in job.refs
        r=tree.refs[t]
        for v in 0:K-1
            d=r.target+v*ld;s=r.source+v*ls
            @simd for i in 1:r.length;dest[d+i]+=slots[s+i];end
        end
    end
    @inbounds for t in job.nodes
        n=nodes[order[t]];isempty(n.coeff) && continue
        c0=first(n.coeff)-1
        if n.identity
            i0=first(n.indices)-1
            for v in 0:K-1
                @simd for i in 1:length(n.coeff);y[i0+i+v*ly]+=coeff[c0+i+v*lc];end
            end
        elseif n.passthrough
            for j in n.children
                child=nodes[j];isempty(child.coeff) && continue
                d0=first(child.coeff)-1
                for v in 0:K-1
                    @simd for i in 1:length(child.coeff);coeff[d0+i+v*lc]+=coeff[c0+i+v*lc];end
                end
                c0+=length(child.coeff)
            end
        elseif isempty(n.children)
            _kernel_nk!(y,first(n.indices)-1,ly,n.V,0,size(n.V,2),coeff,c0,lc,K)
        else
            for j in n.children
                child=nodes[j];isempty(child.coeff) && continue
                _kernel_nk!(coeff,first(child.coeff)-1,lc,child.E,0,size(child.E,2),coeff,c0,lc,K)
            end
        end
    end
end
# Z[1:r, v] .*= c (column stride r)
@inline function _scale_rows_k!(Z,c,r,K)
    @inbounds for v in 0:K-1, i in 1:r;Z[i+v*r]*=c[i];end
    nothing
end
# Several right-hand sides: as `_part_forward!`, inner vectors of factored
# segments are r × K blocks (column stride r) in the worker scratch `Z`.
function _part_forward_k!(rowcoeff,y0,lrc,q::_PacketPart,colcoeff,lcc,Z,K)
    M=q.matrix;offset=0;direct=isempty(q.factors);scaled=!isempty(q.scales)
    @inbounds for (s,cr) in enumerate(q.columns)
        R=direct ? q.matrix : q.factors[s]
        if direct || size(R,1)==0
            _kernel_nk!(rowcoeff,y0,lrc,M,offset,length(cr),colcoeff,first(cr)-1,lcc,K)
            offset+=length(cr)
        else
            r=size(R,2)
            for i in 1:r*K;Z[i]=0.;end
            _kernel_tk!(Z,0,r,R,0,r,colcoeff,first(cr)-1,lcc,K)
            if scaled
                c=q.scales[s];isempty(c) || _scale_rows_k!(Z,c,r,K)
            end
            _kernel_nk!(rowcoeff,y0,lrc,M,offset,r,Z,0,r,K)
            offset+=r
        end
    end
    nothing
end
function _part_adjoint_k!(slots,s0,ls,q::_PacketPart,rowcoeff,y0,lrc,Z,K)
    M=q.matrix;offset=0;direct=isempty(q.factors);scaled=!isempty(q.scales)
    @inbounds for (s,cr) in enumerate(q.columns)
        o=s0+q.offsets[s]
        R=direct ? q.matrix : q.factors[s]
        if direct || size(R,1)==0
            _kernel_tk!(slots,o,ls,M,offset,length(cr),rowcoeff,y0,lrc,K)
            offset+=length(cr)
        else
            r=size(R,2)
            for i in 1:r*K;Z[i]=0.;end
            _kernel_tk!(Z,0,r,M,offset,r,rowcoeff,y0,lrc,K)
            if scaled
                c=q.scales[s];isempty(c) || _scale_rows_k!(Z,c,r,K)
            end
            _kernel_nk!(slots,o,ls,R,0,r,Z,0,r,K)
            offset+=r
        end
    end
    nothing
end
function _interaction_task_k!(p,ws,k,w,t,K)
    lrc=length(p.rowcoeff);lcc=length(p.colcoeff);lrb=length(p.rowbuffer);lcb=length(p.colbuffer);ls=length(p.slots)
    if k>0
        b=p.packets[k];M=b.matrix;y0=first(b.row)-1
        if t
            s=ws.slots
            @inbounds for v in 0:K-1, j in 1:b.width;s[b.slot+j+v*ls]=0.;end
            if b.plain
                _kernel_tk!(s,b.slot,ls,M,0,size(M,2),ws.rowcoeff,y0,lrc,K)
            else
                Z=ws.scratch[w]
                _part_adjoint_k!(s,b.slot,ls,b.f64,ws.rowcoeff,y0,lrc,Z,K)
                _part_adjoint_k!(s,b.slot,ls,b.f32,ws.rowcoeff,y0,lrc,Z,K)
                _part_adjoint_k!(s,b.slot,ls,b.f16,ws.rowcoeff,y0,lrc,Z,K)
            end
        elseif b.plain
            offset=0
            for cr in b.columns
                _kernel_nk!(ws.rowcoeff,y0,lrc,M,offset,length(cr),ws.colcoeff,first(cr)-1,lcc,K)
                offset+=length(cr)
            end
        else
            Z=ws.scratch[w]
            _part_forward_k!(ws.rowcoeff,y0,lrc,b.f64,ws.colcoeff,lcc,Z,K)
            _part_forward_k!(ws.rowcoeff,y0,lrc,b.f32,ws.colcoeff,lcc,Z,K)
            _part_forward_k!(ws.rowcoeff,y0,lrc,b.f16,ws.colcoeff,lcc,Z,K)
        end
    else
        b=p.nearpackets[-k];M=b.matrix
        if t
            g=ws.scratch[w];lg=size(M,1)
            for v in 0:K-1
                offset=0
                for r in b.rows
                    copyto!(g,offset+1+v*lg,ws.rowbuffer,first(r)+v*lrb,length(r));offset+=length(r)
                end
            end
            _kernel_tk!(ws.colbuffer,first(b.col)-1,lcb,M,0,size(M,2),g,0,lg,K)
        else
            s=ws.slots
            @inbounds for v in 0:K-1, i in 1:size(M,1);s[b.slot+i+v*ls]=0.;end
            _kernel_nk!(s,b.slot,ls,M,0,size(M,2),ws.colbuffer,first(b.col)-1,lcb,K)
        end
    end
end
function _forward_task_k!(p,ws,i,w,K,down::F) where {F}
    k=p.fwdtasks[i]
    k<0 && return down(-k)
    _interaction_task_k!(p,ws,k,w,false,K)
    j=p.packetjob[k]
    Threads.atomic_sub!(p.pending[j],1)==1 && down(j)
    nothing
end
# Columns c0+1:c0+K of Y and X; requires ws.k >= K.
function _packet_block!(Y,p::H2PacketMatvecPlan,X,alpha,beta,t,ws,c0,K)
    inputnodes,outputnodes=t ? (p.rows,p.cols) : (p.cols,p.rows)
    up,down=t ? (p.rowup,p.coldown) : (p.colup,p.rowdown)
    inputcoeff,outputcoeff=t ? (ws.rowcoeff,ws.colcoeff) : (ws.colcoeff,ws.rowcoeff)
    input,output=t ? (ws.rowbuffer,ws.colbuffer) : (ws.colbuffer,ws.rowbuffer)
    lic,loc=t ? (length(p.rowcoeff),length(p.colcoeff)) : (length(p.colcoeff),length(p.rowcoeff))
    li,lo=t ? (length(p.rowbuffer),length(p.colbuffer)) : (length(p.colbuffer),length(p.rowbuffer))
    reduced,lr=t ? (outputcoeff,loc) : (output,lo)
    ip,op=t ? (p.rowperm,p.colperm) : (p.colperm,p.rowperm)
    @inbounds for v in 1:K, i in 1:li;input[i+(v-1)*li]=X[ip[i],c0+v];end
    @inbounds for i in 1:lo*K;output[i]=0.;end
    @inbounds for i in 1:loc*K;outputcoeff[i]=0.;end
    nup=length(up.jobs)
    _run_tasks!((k,w)->k<=nup ? _up_job_k!(inputcoeff,lic,inputnodes,input,li,up,k,K) : _interaction_task_k!(p,ws,p.neartasks[k-nup],w,t,K),p,nup+length(p.neartasks))
    if t
        _run_tasks!((k,w)->_interaction_task_k!(p,ws,p.tasks[k],w,t,K),p,length(p.tasks))
        _run_tasks!((k,w)->_down_job_k!(output,lo,outputcoeff,loc,reduced,lr,ws.slots,length(p.slots),outputnodes,down,k,K),p,length(down.jobs))
    else
        _arm!(p)
        _run_tasks!((k,w)->_forward_task_k!(p,ws,k,w,K,(j)->_down_job_k!(output,lo,outputcoeff,loc,reduced,lr,ws.slots,length(p.slots),outputnodes,down,j,K)),p,length(p.fwdtasks))
    end
    @inbounds for v in 1:K, i in 1:lo
        j=op[i];o=output[i+(v-1)*lo]
        Y[j,c0+v]=iszero(beta) ? alpha*o : alpha*o+beta*Y[j,c0+v]
    end
    Y
end
function _packet_mul_k!(Y,p::H2PacketMatvecPlan,X,alpha,beta,t)
    mo,mi=t ? (size(p,2),size(p,1)) : size(p)
    size(X,1)==mi && size(Y,1)==mo && size(X,2)==size(Y,2) || throw(DimensionMismatch("incompatible packet H2 matrix-product dimensions"))
    k=size(X,2)
    if iszero(alpha)
        iszero(beta) ? fill!(Y,0.) : rmul!(Y,beta)
        return Y
    end
    k==0 && return Y
    k==1 && (_packet_mul!(view(Y,:,1),p,view(X,:,1),alpha,beta,t);return Y)
    blocks=cld(k,_MULTI_BLOCK);bs=cld(k,blocks)
    ws=_multi_workspace!(p,bs)
    for c0 in 0:bs:k-1
        _packet_block!(Y,p,X,alpha,beta,t,ws,c0,min(bs,k-c0))
    end
    Y
end
LinearAlgebra.mul!(Y::AbstractMatrix,p::H2PacketMatvecPlan,X::AbstractMatrix,alpha::Number=1,beta::Number=0)=_packet_mul_k!(Y,p,X,alpha,beta,false)
LinearAlgebra.mul!(Y::AbstractMatrix,p::TransposedPacketH2Plan,X::AbstractMatrix,alpha::Number=1,beta::Number=0)=_packet_mul_k!(Y,parent(p),X,alpha,beta,true)
Base.:*(p::Union{H2PacketMatvecPlan,TransposedPacketH2Plan},X::AbstractMatrix)=mul!(zeros(size(p,1),size(X,2)),p,X)
"""
    multi_workspace_bytes(plan, k)
    multi_workspace_bytes(plan)

With `k`: scratch bytes that products of `plan` with `k` right-hand sides use
(blocks of at most $(_MULTI_BLOCK) vectors); the workspace is allocated on
first use, kept by the plan and reused, and grows to the widest block used.
Without `k`: bytes of the workspace the plan currently keeps. `copy(plan)`
starts without one; [`release_multi_workspace!`](@ref) frees it.
"""
function multi_workspace_bytes(p::H2PacketMatvecPlan,k::Integer)
    k<=1 && return 0
    bs=cld(k,cld(k,_MULTI_BLOCK))
    8bs*(length(p.rowcoeff)+length(p.colcoeff)+length(p.rowbuffer)+length(p.colbuffer)+length(p.slots)+sum(length,p.scratch;init=0))
end
function multi_workspace_bytes(p::H2PacketMatvecPlan)
    w=p.multi
    sizeof(w.rowcoeff)+sizeof(w.colcoeff)+sizeof(w.rowbuffer)+sizeof(w.colbuffer)+sizeof(w.slots)+sum(sizeof,w.scratch;init=0)
end
"""
    release_multi_workspace!(plan) -> bytes

Drop the scratch that products with several right-hand sides keep in `plan`
and return its size in bytes. The next such product allocates a workspace for
its own block width, so this also shrinks a workspace that an earlier, wider
product enlarged. Numerical data and single-vector scratch are unaffected.
Like products, it must not run concurrently with another call on `plan`.
"""
function release_multi_workspace!(p::H2PacketMatvecPlan)
    bytes=multi_workspace_bytes(p)
    w=p.multi
    w.k=0;w.rowcoeff=Float64[];w.colcoeff=Float64[];w.rowbuffer=Float64[];w.colbuffer=Float64[]
    w.slots=Float64[];w.scratch=Vector{Float64}[]
    bytes
end

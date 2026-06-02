#!/usr/bin/env python3
"""Visualize Qwen3.5-0.8B benchmark (Phase 1 speed + Phase 2 breakdown)."""

import csv, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUTDIR = Path(__file__).resolve().parent
SPEED_CSV = OUTDIR / "speed.csv"
BD_CSV = OUTDIR / "breakdown_coarse.csv"
BREAK_LABELS = ["hybrid_linear_attn_block_pct","full_attn_block_pct","mlp_block_pct",
                "norm_pct","lm_head_pct","other_pct"]
BREAK_COLORS = {"hybrid_linear_attn_block_pct":"#2196F3","full_attn_block_pct":"#4CAF50",
                "mlp_block_pct":"#FF9800","norm_pct":"#9C27B0","lm_head_pct":"#F44336",
                "all_linear_pct":"#607D8B","other_pct":"#BDBDBD"}

def _f(v,d=0.0):
    try: return float(v)
    except: return d

def load_speed(path):
    rows=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            bs=int(r["batch_size"]); it=int(r["input_tokens"]); ot=int(r["output_tokens"])
            pf=_f(r["prefill_ms"]); dt=_f(r["decode_per_token_ms"])
            tps=_f(r["tokens_per_sec"]); oom=r.get("status","").strip()=="OOM"
            rows.append({"batch_size":bs,"input_tokens":it,"output_tokens":ot,
                         "prefill_ms":pf,"decode_per_token_ms":dt,"tokens_per_sec":tps,"oom":oom})
    return rows

def load_bd(path):
    rows=[]
    with open(path) as f:
        for r in csv.DictReader(f):
            if r.get("status","").strip()=="OOM": continue
            bs=int(r["batch_size"]); it=int(r["input_tokens"]); ot=int(r["output_tokens"])
            pref={k.replace("prefill_",""):_f(r[k]) for k in r if k.startswith("prefill_") and k.endswith("_pct")}
            dec={k.replace("decode_",""):_f(r[k]) for k in r if k.startswith("decode_") and k.endswith("_pct")}
            rows.append({"batch_size":bs,"input_tokens":it,"output_tokens":ot,
                         "prefill_breakdown":pref,"decode_breakdown":dec})
    return rows

# ── Speed plots ──

def plot_prefill_vs_input(rows,ax):
    for bs,color in zip(sorted(set(r["batch_size"] for r in rows)),
                        plt.cm.viridis(np.linspace(0.1,0.9,len(set(r["batch_size"] for r in rows))))):
        pts=sorted(((r["input_tokens"],r["prefill_ms"]) for r in rows
                    if r["batch_size"]==bs and not r["oom"] and r["prefill_ms"]>0))
        if not pts: continue
        xs,ys=zip(*pts); ax.plot(xs,ys,"o-",color=color,label=f"bs={bs}",markersize=5,linewidth=1.2)
    ax.set_xlabel("Input Tokens"); ax.set_ylabel("Prefill Latency (ms)")
    ax.set_title("Prefill Latency vs Input Length"); ax.legend(fontsize=7,ncol=2)
    ax.grid(True,alpha=0.3); ax.set_xscale("log",base=2)
    ax.set_xticks(sorted(set(r["input_tokens"] for r in rows)),
                  [str(t) for t in sorted(set(r["input_tokens"] for r in rows))],rotation=30,fontsize=7)

def plot_decode_vs_batch(rows,ax):
    for il,color in zip(sorted(set(r["input_tokens"] for r in rows)),
                        plt.cm.plasma(np.linspace(0.1,0.9,len(set(r["input_tokens"] for r in rows))))):
        pts=sorted(((r["batch_size"],r["decode_per_token_ms"]) for r in rows
                    if r["input_tokens"]==il and not r["oom"] and r["decode_per_token_ms"]>0))
        if not pts: continue
        xs,ys=zip(*pts); ax.plot(xs,ys,"s-",color=color,label=f"in={il}",markersize=5,linewidth=1.2)
    ax.set_xlabel("Batch Size"); ax.set_ylabel("Decode per Token (ms)")
    ax.set_title("Decode Latency vs Batch Size"); ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3)

def plot_throughput_vs_batch(rows,ax):
    for il,color in zip(sorted(set(r["input_tokens"] for r in rows)),
                        plt.cm.plasma(np.linspace(0.1,0.9,len(set(r["input_tokens"] for r in rows))))):
        pts=sorted(((r["batch_size"],r["tokens_per_sec"]) for r in rows
                    if r["input_tokens"]==il and not r["oom"] and r["tokens_per_sec"]>0))
        if not pts: continue
        xs,ys=zip(*pts); ax.plot(xs,ys,"D-",color=color,label=f"in={il}",markersize=5,linewidth=1.2)
    ax.set_xlabel("Batch Size"); ax.set_ylabel("Tokens / sec")
    ax.set_title("Decode Throughput vs Batch Size"); ax.legend(fontsize=7,ncol=2); ax.grid(True,alpha=0.3)

def plot_oom_map(rows,ax):
    bss=sorted(set(r["batch_size"] for r in rows)); ils=sorted(set(r["input_tokens"] for r in rows))
    mat=np.zeros((len(bss),len(ils)))
    for r in rows: mat[bss.index(r["batch_size"]),ils.index(r["input_tokens"])]=1 if r["oom"] else 0
    ax.imshow(mat,cmap=plt.cm.RdYlGn.reversed(),aspect="auto",vmin=0,vmax=1)
    ax.set_xticks(range(len(ils)),ils,rotation=30,fontsize=8)
    ax.set_yticks(range(len(bss)),bss,fontsize=8)
    ax.set_xlabel("Input Tokens"); ax.set_ylabel("Batch Size")
    ax.set_title("OOM Map (red=OOM, green=OK)")
    for bi in range(len(bss)):
        for ii in range(len(ils)):
            ax.text(ii,bi,"OOM" if mat[bi,ii] else "OK",ha="center",va="center",fontsize=7,
                    color="white" if mat[bi,ii] else "black")

# ── Breakdown plots ──

def plot_prefill_breakdown_vs_input(bd,ax):
    rows=sorted([r for r in bd if r["batch_size"]==1],key=lambda r:r["input_tokens"])
    if not rows: return
    xs=np.arange(len(rows)); bottom=np.zeros(len(rows))
    for key in BREAK_LABELS:
        vals=[r["prefill_breakdown"].get(key.replace("_pct",""),0) for r in rows]
        ax.bar(xs,vals,bottom=bottom,label=key.replace("_pct",""),color=BREAK_COLORS[key],width=0.6)
        bottom+=vals
    ax.set_xticks(xs,[f"in={r['input_tokens']}" for r in rows],rotation=30,fontsize=8)
    ax.set_ylabel("% of Prefill Time"); ax.set_title("Prefill Breakdown vs Input Tokens (bs=1)")
    ax.legend(fontsize=7,ncol=3)

def plot_decode_breakdown_vs_batch(bd,ax):
    rows=sorted([r for r in bd if r["input_tokens"]==128],key=lambda r:r["batch_size"])
    if not rows: return
    xs=np.arange(len(rows)); bottom=np.zeros(len(rows))
    for key in BREAK_LABELS:
        vals=[r["decode_breakdown"].get(key.replace("_pct",""),0) for r in rows]
        ax.bar(xs,vals,bottom=bottom,label=key.replace("_pct",""),color=BREAK_COLORS[key],width=0.6)
        bottom+=vals
    ax.set_xticks(xs,[f"bs={r['batch_size']}" for r in rows],fontsize=8)
    ax.set_ylabel("% of Decode Time"); ax.set_title("Decode Breakdown vs Batch Size (input=128)")
    ax.legend(fontsize=7,ncol=3)

def plot_prefill_trend(bd,ax):
    rows=sorted([r for r in bd if r["batch_size"]==1],key=lambda r:r["input_tokens"])
    if not rows: return
    xs=[r["input_tokens"] for r in rows]
    for c in ["hybrid_linear_attn_block","full_attn_block","mlp_block","all_linear"]:
        ys=[r["prefill_breakdown"].get(c,0) for r in rows]
        ax.plot(xs,ys,"o-",label=c,markersize=5,linewidth=1.5)
    ax.set_xlabel("Input Tokens"); ax.set_ylabel("% of Prefill Time")
    ax.set_title("Prefill Component Trends vs Input Length (bs=1)")
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3); ax.set_xscale("log",base=2)
    ax.set_xticks(xs,[str(x) for x in xs],rotation=30,fontsize=7)

def plot_decode_trend(bd,ax):
    rows=sorted([r for r in bd if r["batch_size"]==1],key=lambda r:r["input_tokens"])
    if not rows: return
    xs=[r["input_tokens"] for r in rows]
    for c in ["hybrid_linear_attn_block","full_attn_block","mlp_block","all_linear"]:
        ys=[r["decode_breakdown"].get(c,0) for r in rows]
        ax.plot(xs,ys,"s-",label=c,markersize=5,linewidth=1.5)
    ax.set_xlabel("Input Tokens"); ax.set_ylabel("% of Decode Time")
    ax.set_title("Decode Component Trends vs Input Length (bs=1)")
    ax.legend(fontsize=8); ax.grid(True,alpha=0.3); ax.set_xscale("log",base=2)
    ax.set_xticks(xs,[str(x) for x in xs],rotation=30,fontsize=7)

# ── Main ──

def main():
    speed=load_speed(SPEED_CSV); bd=load_bd(BD_CSV)
    ok=[r for r in speed if not r["oom"]]; oom=[r for r in speed if r["oom"]]
    print(f"Speed: {len(speed)} rows ({len(ok)} OK, {len(oom)} OOM)")
    print(f"Breakdown: {len(bd)} rows")

    # Figure 1: Speed overview
    fig1,axes1=plt.subplots(2,2,figsize=(14,10))
    plot_prefill_vs_input(speed,axes1[0,0]); plot_decode_vs_batch(speed,axes1[0,1])
    plot_throughput_vs_batch(speed,axes1[1,0]); plot_oom_map(speed,axes1[1,1])
    fig1.suptitle("Qwen3.5-0.8B Speed Benchmark (Phase 1)",fontsize=14,fontweight="bold")
    fig1.tight_layout(); fig1.savefig(OUTDIR/"speed_analysis.png",dpi=150,bbox_inches="tight")
    print("Saved speed_analysis.png")

    # Figure 2: Prefill scaling
    fig2,ax2=plt.subplots(figsize=(10,5))
    bs1=sorted([r for r in ok if r["batch_size"]==1],key=lambda r:r["input_tokens"])
    xs=np.array([r["input_tokens"] for r in bs1],dtype=float)
    ys=np.array([r["prefill_ms"] for r in bs1],dtype=float)
    ax2.plot(xs,ys,"o-",color="steelblue",markersize=8,label="Prefill (bs=1)")
    ax2.plot(xs,xs*ys[-1]/xs[-1],"--",color="gray",alpha=0.6,label="O(n) reference")
    ax2.plot(xs,xs**0.6*ys[-1]/xs[-1]**0.6,":",color="red",alpha=0.6,label=r"$O(n^{0.6})$ reference")
    ax2.set_xlabel("Input Tokens"); ax2.set_ylabel("Prefill Latency (ms)")
    ax2.set_title("Prefill Scaling (bs=1) — sub-linear confirms parallel SSM scan")
    ax2.legend(); ax2.grid(True,alpha=0.3); ax2.set_xscale("log",base=2); ax2.set_yscale("log",base=2)
    ax2.set_xticks(xs,[str(int(x)) for x in xs])
    fig2.savefig(OUTDIR/"prefill_scaling.png",dpi=150,bbox_inches="tight")
    print("Saved prefill_scaling.png")

    # Figure 3: Decode scaling
    fig3,ax3=plt.subplots(figsize=(10,5))
    in128=sorted([r for r in ok if r["input_tokens"]==128],key=lambda r:r["batch_size"])
    bv=[r["batch_size"] for r in in128]; tv=[r["tokens_per_sec"] for r in in128]
    ax3.plot(bv,tv,"o-",color="steelblue",markersize=8,label="Actual throughput")
    ax3.plot(bv,[tv[0]*b for b in bv],"--",color="gray",alpha=0.6,label="Ideal linear scaling")
    ax3.set_xlabel("Batch Size"); ax3.set_ylabel("Tokens / sec")
    ax3.set_title("Decode Throughput Scaling (input=128)"); ax3.legend(); ax3.grid(True,alpha=0.3)
    fig3.savefig(OUTDIR/"decode_scaling.png",dpi=150,bbox_inches="tight")
    print("Saved decode_scaling.png")

    if not bd: print("No breakdown data."); return

    # Figure 4: Breakdown overview
    fig4,axes4=plt.subplots(2,2,figsize=(14,10))
    plot_prefill_breakdown_vs_input(bd,axes4[0,0]); plot_decode_breakdown_vs_batch(bd,axes4[0,1])
    plot_prefill_trend(bd,axes4[1,0]); plot_decode_trend(bd,axes4[1,1])
    fig4.suptitle("Qwen3.5-0.8B Coarse Breakdown (Phase 2)",fontsize=14,fontweight="bold")
    fig4.tight_layout(); fig4.savefig(OUTDIR/"breakdown_analysis.png",dpi=150,bbox_inches="tight")
    print("Saved breakdown_analysis.png")

    # Figure 5: Prefill vs Decode bar comparison
    fig5,ax5=plt.subplots(figsize=(10,5))
    r0=[r for r in bd if r["batch_size"]==1 and r["input_tokens"]==128]
    if r0:
        x=np.arange(5); w=0.35
        comps=["hybrid_linear_attn_block","full_attn_block","mlp_block","norm","other"]
        pf_vals=[r0[0]["prefill_breakdown"].get(c,0) for c in comps]
        de_vals=[r0[0]["decode_breakdown"].get(c,0) for c in comps]
        ax5.bar(x-w/2,pf_vals,w,label="Prefill",color="#2196F3",alpha=0.8)
        ax5.bar(x+w/2,de_vals,w,label="Decode",color="#FF9800",alpha=0.8)
        ax5.set_xticks(x,[c.replace("_","\n") for c in comps],fontsize=8)
        ax5.set_ylabel("% of Time"); ax5.set_title("Prefill vs Decode Time Distribution (bs=1, in=128)")
        ax5.legend()
    fig5.tight_layout(); fig5.savefig(OUTDIR/"prefill_vs_decode_bar.png",dpi=150,bbox_inches="tight")
    print("Saved prefill_vs_decode_bar.png")

    print("Done.")

if __name__=="__main__":
    main()

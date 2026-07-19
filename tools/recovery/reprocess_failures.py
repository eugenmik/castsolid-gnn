"""Re-process no_bijection stems with KDTree nearest matching (robust to the
float noise from the XDMF roundtrip). Greedy top-k assignment guarantees a
bijection for near-identical point sets. Updates perms + a second report."""
import os, glob, csv, warnings
import numpy as np
warnings.filterwarnings("ignore")
from mpi4py import MPI
from dolfinx import io, fem
import meshio
from scipy.spatial import cKDTree

MSHROOT="/home/eugen/Foundry_GNN/data"
RAW="/home/eugen/Media/MIGRATION/Debian/casting_gnn/data/raw"
OUT="/home/eugen/Foundry_GNN/data/features/perms"
REPORT="/home/eugen/Foundry_GNN/runs/recovery_report.csv"
OUT2="/home/eugen/Foundry_GNN/runs/recovery_report_fix.csv"
TMP=f"/home/eugen/.claude/jobs/430075b7/tmp/_fix_{os.getpid()}.xdmf"
T_SOL=1120.0

def find_msh(s):
    for d in ("msh_none","msh_half","msh_all"):
        h=glob.glob(f"{MSHROOT}/{d}/*/{s}.msh")
        if h: return h[0]
def find_npz(s):
    h=glob.glob(f"{RAW}/*/{s}.npz"); return h[0] if h else None

def parse_msh(path):
    lines=open(path).read().splitlines(); i=0; phys={}; ids=[]; xyz=[]
    while i<len(lines):
        s=lines[i].strip()
        if s=="$PhysicalNames":
            c=int(lines[i+1])
            for k in range(c):
                dim,tag,nm=lines[i+2+k].split(maxsplit=2); phys[int(tag)]=nm.strip().strip('"')
            i+=2+c; continue
        if s=="$Nodes":
            c=int(lines[i+1])
            for k in range(c):
                p=lines[i+2+k].split(); ids.append(int(p[0])); xyz.append((float(p[1]),float(p[2]),float(p[3])))
            i+=2+c; continue
        if s=="$Elements":
            c=int(lines[i+1]); id2={n:j for j,n in enumerate(ids)}; cool=set(); adi=set()
            for k in range(c):
                p=[int(x) for x in lines[i+2+k].split()]; et,nt=p[1],p[2]; tags=p[3:3+nt]; conn=p[3+nt:]
                if et==2 and nt>=1:
                    nm=phys.get(tags[0])
                    if nm=="Cooling": cool.update(id2[v] for v in conn)
                    elif nm=="Adiabatic": adi.update(id2[v] for v in conn)
            N=len(ids); cm=np.zeros(N,bool); am=np.zeros(N,bool); cm[list(cool)]=True; am[list(adi)]=True
            return np.array(xyz),cm,am
        i+=1

def greedy_match(dof, gxyz):
    N=dof.shape[0]; tree=cKDTree(gxyz)
    k=min(6,N); d,cand=tree.query(dof,k=k)
    if k==1: cand=cand[:,None]; d=d[:,None]
    order=np.argsort(d[:,0])          # assign closest-first
    g=-np.ones(N,dtype=np.int64); used=np.zeros(N,bool)
    ok=True
    for i in order:
        placed=False
        for c in cand[i]:
            if not used[c]:
                g[i]=c; used[c]=True; placed=True; break
        if not placed: ok=False; break
    return (g if ok and (g>=0).all() and len(np.unique(g))==N else None), float(d[:,0].max())

def compute(stem):
    mp=find_msh(stem)
    if mp is None: return dict(stem=stem,status="no_msh")
    xyz,cool,adi=parse_msh(mp); inter=~(cool|adi); N=xyz.shape[0]
    m=meshio.read(mp); tet=[cb.data for cb in m.cells if cb.type=="tetra"]
    if not tet: return dict(stem=stem,status="no_tet")
    meshio.write(TMP, meshio.Mesh(points=m.points, cells=[("tetra",tet[0])]))
    with io.XDMFFile(MPI.COMM_SELF,TMP,"r") as xf: dom=xf.read_mesh(name="Grid")
    V=fem.functionspace(dom,("Lagrange",1)); dof=V.tabulate_dof_coordinates()[:,:3]
    if dof.shape[0]!=N: return dict(stem=stem,status=f"N_mismatch")
    g,maxd=greedy_match(dof,xyz)
    if g is None: return dict(stem=stem,status="no_bijection2",delta_after=round(maxd,4))
    sn=np.load(find_npz(stem))["snapshots_nodal"]; mid=sn.shape[0]//2
    tg=np.empty_like(sn); tg[:,g]=sn
    da=float(tg[mid][inter].mean()-tg[mid][cool].mean()) if cool.any() and inter.any() else float('nan')
    status="valid" if (da>0 or np.isnan(da)) else "phys_fail"
    if status=="valid": np.save(os.path.join(OUT,f"{stem}.npy"), g.astype(np.int32))
    return dict(stem=stem,status=status,delta_after=round(da,2),maxd=round(maxd,5))

fails=[r[0] for r in csv.reader(open(REPORT)) if len(r)>1 and r[1]=="no_bijection"]
print(f"reprocessing {len(fails)} failures", flush=True)
counts={}
with open(OUT2,"w",newline="") as f:
    w=csv.DictWriter(f,fieldnames=["stem","status","delta_after","maxd"]); w.writeheader()
    for i,s in enumerate(fails):
        try: r=compute(s)
        except Exception as e: r=dict(stem=s,status=f"err_{type(e).__name__}")
        w.writerow({k:r.get(k,"") for k in ["stem","status","delta_after","maxd"]})
        counts[r["status"]]=counts.get(r["status"],0)+1
        if (i+1)%200==0: f.flush(); print(f"  {i+1}/{len(fails)} {counts}",flush=True)
print("DONE",counts,flush=True)
if os.path.exists(TMP): os.remove(TMP)

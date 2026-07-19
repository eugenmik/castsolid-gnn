"""Compute DOF->gmsh node permutation per casting via dolfinx and validate
physics (interior hotter than cooling after reorder). Saves compact int32
perm per stem + a CSV report. Run in the fenicsx env.

perm g: temps_gmsh[:, g] = temps_dof  (g[dof] = gmsh_node_index)
"""
import os, glob, warnings, csv, sys
import numpy as np
warnings.filterwarnings("ignore")
from mpi4py import MPI
from dolfinx import io, fem
import meshio

RAW="/home/eugen/Media/MIGRATION/Debian/casting_gnn/data/raw"
MSHROOT="/home/eugen/Foundry_GNN/data"
PTDIR="/home/eugen/Foundry_GNN/data/pt"
OUT="/home/eugen/Foundry_GNN/data/features/perms"
REPORT="/home/eugen/Foundry_GNN/runs/recovery_report.csv"
TMP=f"/home/eugen/.claude/jobs/430075b7/tmp/_perm_{os.getpid()}.xdmf"
T_SOL=1120.0
os.makedirs(OUT, exist_ok=True)

def find_msh(stem):
    for s in ("msh_none","msh_half","msh_all"):
        h=glob.glob(os.path.join(MSHROOT,s,"*",stem+".msh"))
        if h: return h[0]
def find_npz(stem):
    h=glob.glob(os.path.join(RAW,"*",stem+".npz")); return h[0] if h else None

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

def compute(stem):
    mp=find_msh(stem)
    if mp is None: return dict(stem=stem, status="no_msh")
    xyz,cool,adi=parse_msh(mp); inter=~(cool|adi); N=xyz.shape[0]
    m=meshio.read(mp); tet=[cb.data for cb in m.cells if cb.type=="tetra"]
    if not tet: return dict(stem=stem, status="no_tet")
    meshio.write(TMP, meshio.Mesh(points=m.points, cells=[("tetra",tet[0])]))
    with io.XDMFFile(MPI.COMM_SELF, TMP, "r") as xf: dom=xf.read_mesh(name="Grid")
    V=fem.functionspace(dom,("Lagrange",1)); dof=V.tabulate_dof_coordinates()[:,:3]
    if dof.shape[0]!=N: return dict(stem=stem, status=f"N_mismatch_{dof.shape[0]}_{N}")
    gk={tuple(np.round(p,4)):j for j,p in enumerate(xyz)}
    g=np.array([gk.get(tuple(np.round(p,4)),-1) for p in dof])
    if (g<0).any() or len(np.unique(g))!=N:
        return dict(stem=stem, status="no_bijection")
    # validate physics on raw temps
    npzp=find_npz(stem)
    sn=np.load(npzp)["snapshots_nodal"]
    mid=sn.shape[0]//2
    tg=np.empty_like(sn); tg[:,g]=sn
    db=float(sn[mid][inter].mean()-sn[mid][cool].mean()) if cool.any() and inter.any() else float('nan')
    da=float(tg[mid][inter].mean()-tg[mid][cool].mean()) if cool.any() and inter.any() else float('nan')
    status="valid" if (da>0 or np.isnan(da)) else "phys_fail"
    np.save(os.path.join(OUT, f"{stem}.npy"), g.astype(np.int32))
    return dict(stem=stem, status=status, delta_before=round(db,2), delta_after=round(da,2))

def main():
    stems=sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(PTDIR,"*.pt")))
    print(f"stems: {len(stems)}", flush=True)
    rows=[]; counts={}
    with open(REPORT,"w",newline="") as f:
        w=csv.DictWriter(f, fieldnames=["stem","status","delta_before","delta_after"]); w.writeheader()
        for i,s in enumerate(stems):
            try: r=compute(s)
            except Exception as e: r=dict(stem=s, status=f"err_{type(e).__name__}")
            w.writerow({k:r.get(k,"") for k in ["stem","status","delta_before","delta_after"]})
            counts[r["status"]]=counts.get(r["status"],0)+1
            if (i+1)%500==0:
                f.flush(); print(f"  {i+1}/{len(stems)}  {counts}", flush=True)
    print("DONE", counts, flush=True)
    if os.path.exists(TMP): os.remove(TMP)

if __name__=="__main__": main()

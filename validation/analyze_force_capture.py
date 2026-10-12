"""Validate a complete force-ramp capture and summarize its diagnostic state."""
import csv, math, json, sys
from pathlib import Path
src=Path(sys.argv[1]); frames={}; load={}; energy={}; quats={}; broken={}; count={}; ramp=None; header=None
initial_positions={}; final_positions={}; peak_displacement=0.0; fixed_drift=0.0
for r in csv.reader(src.open()):
 if r[0]=='H':
  header=r; nx,ny,_=map(int,r[3:6]); anchors={0,nx-1,nx*(ny-1),nx*ny-1}
 elif r[0]=='R': ramp=list(map(float,r[1:4]))
 elif r[0]=='F': frames[int(r[1])]=list(map(float,r[2:]))
 elif r[0]=='L': load[int(r[1])]=list(map(float,r[2:]))
 elif r[0]=='E': energy[int(r[1])]=list(map(float,r[2:]))
 elif r[0]=='Q': quats[int(r[1])]=list(map(float,r[2:]))
 elif r[0]=='B': broken.setdefault(int(r[1]),[]).append(tuple(map(int,r[2:4])))
 elif r[0]=='V':
  n=int(r[1]); count[n]=count.get(n,0)+1
  assert all(math.isfinite(float(x)) for x in r[3:7]), 'nonfinite voxel state'
  i=int(r[2]); p=tuple(map(float,r[3:6])); final_positions[i]=p
  if n==0: initial_positions[i]=p
  displacement=math.sqrt(sum((a-b)**2 for a,b in zip(p,initial_positions[i])))
  peak_displacement=max(peak_displacement,displacement)
  if i in anchors: fixed_drift=max(fixed_drift,displacement)
keys=sorted(frames); last=keys[-1]; dt=float(header[2]); nx,ny,nz=map(int,header[3:6]); size=nx*ny*nz
assert all(count.get(n)==size and n in quats and n in load and n in energy for n in keys)
assert all(abs(sum(x*x for x in q)-1)<2e-5 for q in quats.values())
assert all(math.isfinite(x) for v in frames.values() for x in v)
assert all(load[b][0]>=load[a][0] for a,b in zip(keys,keys[1:]))
previous_broken=set()
for n in keys:
 current=set(broken.get(n,[]))
 assert len(current)==len(broken.get(n,[]))==int(frames[n][9]), 'broken bond count mismatch'
 assert previous_broken.issubset(current), 'this fixture has no bond healing'
 for i,j in current:
  assert 0<=i<j<size, 'invalid broken endpoint'
  assert ((j-i==1 and i%nx<nx-1) or
          (j-i==nx and (i//nx)%ny<ny-1) or
          (j-i==nx*ny and i//(nx*ny)<nz-1)), 'broken bond is not an original face neighbor'
 previous_broken=current
errors=[]
for n in keys:
 expected=ramp[0]*max(0,min(1,((n-.5)*dt-ramp[2])/ramp[1])) if n else 0
 errors.append(abs(expected-load[n][0]))
assert max(errors)<0.001, 'force ramp differs from requested load'
assert fixed_drift<1e-7, 'fixed support moved'
parent=list(range(size))
def find(i):
 while parent[i]!=i:
  parent[i]=parent[parent[i]]; i=parent[i]
 return i
missing={tuple(sorted(b)) for b in broken.get(last,[])}
for i in range(size):
 x=i%nx; y=(i//nx)%ny; z=i//(nx*ny)
 for ok,j in [(x<nx-1,i+1),(y<ny-1,i+nx),(z<nz-1,i+nx*ny)]:
  if ok and (i,j) not in missing: parent[find(i)]=find(j)
sizes={}
for i in range(size):
 root=find(i); sizes[root]=sizes.get(root,0)+1
sizes=sorted(sizes.values(),reverse=True)
f=frames[last]
assert len(sizes)==int(f[10]), 'saved cluster count differs from reconstructed connectivity'
def budget(n):
 f=frames[n]
 return sum(f[i] for i in [12,13,15,16,17,18,19])+energy[n][1]
gaps={n:budget(n)-budget(keys[0])-load[n][1] for n in keys}
peak_gap_frame=max(keys,key=lambda n:abs(gaps[n]))
from collections import Counter
summary={'physical_seconds':last*dt,'post_contact_seconds':last*dt-ramp[2], 'snapshots':len(keys),
 'force_peak_N':ramp[0],'ramp_duration_seconds':ramp[1],'hold_seconds':last*dt-ramp[2]-ramp[1],
 'max_recorded_applied_force_N':max(v[0] for v in load.values()),
 'force_error_max_N':max(errors),'drive_work_estimate_J':load[last][1],
 'broken_bonds':int(f[9]),'components':len(sizes),'component_sizes':sizes,
 'max_recorded_components':max(int(v[10]) for v in frames.values()),
 'component_size_histogram':dict(sorted(Counter(sizes).items())),
 'largest_component_voxels':sizes[0],'singleton_components':sizes.count(1),
 'peak_recorded_voxel_speed_m_s':max(v[11] for v in frames.values()),
 'peak_recorded_stress_ratio':max(v[20] for v in frames.values()),
 'peak_recorded_voxel_displacement_m':peak_displacement,
 'final_max_voxel_displacement_m':max(math.sqrt(sum((a-b)**2 for a,b in zip(p,initial_positions[i]))) for i,p in final_positions.items()),
 'max_fixed_support_drift_m':fixed_drift,
 'final_tool_speed_m_s':math.sqrt(sum(x*x for x in f[3:6])),
 'initial_accounted_energy_J':budget(keys[0]),'final_accounted_energy_J':budget(last),
 'final_energy_reservoirs_J':dict(zip(['wood_kinetic','tool_kinetic','pending_release','surface_work','damage_heat','plastic_heat','stored_plastic','elastic'],[f[12],f[13],f[15],f[16],f[17],f[18],f[19],energy[last][1]])),
 'energy_gap_after_drive_work_estimate_J':budget(last)-budget(keys[0])-load[last][1],
 'max_abs_recorded_energy_gap_J':abs(gaps[peak_gap_frame]),
 'max_abs_energy_gap_at_physical_seconds':peak_gap_frame*dt,
 'final_PCG_relative_residual':energy[last][2],
 'peak_recorded_PCG_relative_residual':max(v[2] for v in energy.values()),
 'work_note':'COM force times net displacement per outer step, including contact projection; diagnostic estimate, not an exact substep energy ledger.'}
out=src.with_name(src.stem+'-measurements.json'); out.write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps({k:v for k,v in summary.items() if k!='component_sizes'},indent=2))

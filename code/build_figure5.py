"""Figure 5 assets from the main-text reduced model preset; no fitting or legacy overwrite.
Run: uv run python code/build_figure5.py
"""
from pathlib import Path
from dataclasses import asdict
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
import phase_model as model
import dual_oscillator_minimal as old
import manuscript_fig_2x2 as legacy

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'figure' / 'figure5'
M, E = legacy.M_COLOR, legacy.E_COLOR
CACHE = {}


def peaks(P, params=model.TUNED, genotype='WT'):
    key = (float(P), genotype)
    if key not in CACHE:
        result = model.periodic_summary(P, params, genotype)
        if not result['locked']:
            raise RuntimeError(f'Unsettled orbit: {key}: {result}')
        CACHE[key] = result
    return CACHE[key]['peaks']


def save(fig, name):
    for ext in ('pdf', 'svg', 'png'):
        fig.savefig(OUT / f'{name}.{ext}', dpi=400, bbox_inches=None)
    plt.close(fig)


def schematic(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.set_axis_off(); ax.set_aspect('equal')
    for x, label, color in ((2, 'M', M), (8, 'E', E)):
        ax.add_patch(Circle((x, 5), 1, fill=False, ec=color, lw=1.4))
        ax.text(x, 5, label, color=color, ha='center', va='center', fontsize=12)
        ax.annotate('', (x, 6.1), (x, 8.1), arrowprops=dict(arrowstyle='->', lw=1))
    ax.text(5, 8.6, 'Shared light reference', ha='center', fontsize=7)
    ax.add_patch(FancyArrowPatch((3.1, 5), (6.9, 5), arrowstyle='<->', mutation_scale=9, lw=1))
    ax.text(5, 4.1, 'Synchronization', ha='center', fontsize=6)
    ax.text(5, 3.1, 'PDF-dependent drive', ha='center', fontsize=7)
    ax.text(2, 1.9, 'Advance', ha='center', fontsize=9, color=M)
    ax.text(8, 1.9, 'Delay', ha='center', fontsize=9, color=E)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    legacy._style()
    legacy.peak_zt = peaks
    legacy._draw_schematic = schematic
    legacy.build_figure(params=model.TUNED, out_path=str(OUT / 'model_summary_2x2.pdf'))
    # The quantitative panels are also supplied separately for manuscript assembly.
    Ps = sorted(old.EXPERIMENTAL_TARGETS)
    grid = np.arange(min(Ps), max(Ps)+.01, .2)
    pred = np.array([peaks(P) for P in grid])
    for kind in ('peak_times', 'phase_gap', 'genotypes'):
        fig, ax = plt.subplots(figsize=(3.3, 2.7), layout='constrained')
        if kind == 'peak_times':
            for c, color, label in ((0, M, 'M'), (1, E, 'E')):
                ax.plot(grid, pred[:, c], color=color, label=label)
                ax.errorbar(Ps, [old.EXPERIMENTAL_TARGETS[P][c] for P in Ps],
                            yerr=[old.EXPERIMENTAL_PHASE_SDS[P][c] for P in Ps],
                            fmt='o', ms=3, color=color, lw=.6, capsize=1.5)
            ax.set(xlabel='Daylength (h)', ylabel='Peak time (ZT)', ylim=(4,24), xticks=[4,8,12,16,20])
            ax.legend(fontsize=8)
        elif kind == 'phase_gap':
            ax.plot(grid, old.folded(pred[:,1]-pred[:,0]), color=legacy.WT_COLOR)
            ax.errorbar(Ps, [old.folded(old.EXPERIMENTAL_TARGETS[P][1]-old.EXPERIMENTAL_TARGETS[P][0]) for P in Ps],
                        yerr=[old.EXPERIMENTAL_GAP_SDS[P] for P in Ps], fmt='o', ms=3,
                        color=legacy.DATA_GRAY, lw=.6, capsize=1.5)
            ax.set(xlabel='Daylength (h)', ylabel='M–E phase gap (h)', ylim=(-.5,6), xticks=[4,8,12,16,20])
        else:
            genos=['WT','pdf0','pdfrM','pdfrE']
            targets={'WT':old.EXPERIMENTAL_TARGETS[12], **old.GENOTYPE_TARGETS_12}
            for c, color, label in ((0,M,'M'),(1,E,'E')):
                # Horizontal separation exposes coincident peaks without changing their times.
                x=np.arange(4)+(c-.5)*.16
                ax.scatter(x-.16,[targets[g][c] for g in genos],facecolors='none',edgecolors=color,s=23)
                ax.scatter(x+.16,[peaks(12,genotype=g)[c] for g in genos],color=color,marker='s',s=15,label=label)
            ax.set(xticks=range(4),xticklabels=['WT',r'$pdf^0$','pdfr-M','pdfr-E'],ylabel='Peak time at 12:12 (ZT)',ylim=(11.5,18.5))
        ax.text(.03,.97,'Filled: model; open: data' if kind=='genotypes' else 'Line: model; points: data',transform=ax.transAxes,va='top',fontsize=7)
        for side in ('top','right'): ax.spines[side].set_visible(False)
        save(fig,kind)
    fig, axes = plt.subplots(2,2,figsize=(5.4,4.2),layout='constrained')
    profile_rows=[]
    for ax,(title,P,g) in zip(axes.flat, [('PDF loss',12,'pdf0'),('Spring / fall',12,'WT'),('Winter',10,'WT'),('Summer',16,'WT')]):
        t, phi=model.simulate(P,genotype=g,dt=.02)
        keep=t>=t[-1]-24
        zt=t[keep]-(t[-1]-24)
        wave=old.tim_proxy(phi[keep]); wave/=wave.max(axis=0)
        ax.axvspan(P,24,color='.88',lw=0)
        ax.plot(zt,wave[:,1],color=E,lw=2.2,label='E')
        ax.plot(zt,wave[:,0],color=M,lw=1.5,ls='--' if g=='pdf0' else '-',label='M')
        ax.set(title=f'{title} ({P}L:{24-P}D)',xlim=(0,24),ylim=(0,1.12),xticks=[0,6,12,18,24],yticks=[],xlabel='ZT (h)')
        ax.tick_params(labelsize=9); ax.title.set_fontsize(10)
        for side in ('top','right','left'): ax.spines[side].set_visible(False)
        for z,m,e in zip(zt,*wave.T): profile_rows.append([title,P,g,z,m,e])
    axes[0,0].legend(fontsize=8,loc='upper left')
    save(fig,'seasonal_profiles')
    with (OUT/'seasonal_profiles.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['panel','light_h','genotype','ZT_h','M_normalized','E_normalized']); w.writerows(profile_rows)
    with (OUT/'model_peaks.csv').open('w') as f:
        w=csv.writer(f); w.writerow(['light_h','genotype','M_ZT','E_ZT','gap_h','locked'])
        for (P,g),r in sorted(CACHE.items()):
            m,e=r['peaks']; w.writerow([P,g,m,e,old.folded(e-m),r['locked']])
    rmse=float(np.sqrt(np.mean([(np.array(peaks(P))-old.EXPERIMENTAL_TARGETS[P])**2 for P in Ps])))
    manifest={'model':model.__version__,'preset':'original compact main-text preset','parameters':asdict(model.TUNED),'paired_WT_RMSE_h':rmse,
              'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['code/phase_model.py','code/build_figure5.py','code/manuscript_fig_2x2.py']}}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Figure 5 reduced model assets: {OUT}; paired WT RMSE = {rmse:.3f} h')

if __name__=='__main__': main()

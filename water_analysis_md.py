import os
import shutil 
import numpy as np
import subprocess as sp
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
import argparse
from schrodinger.application.desmond.packages import analysis
from schrodinger.application.desmond.packages import topo
from schrodinger.application.desmond.packages import traj_util
from schrodinger.structure import create_new_structure
from schrodinger.application.desmond import cmj
from schrodinger.application.desmond import cms
from schrodinger.application.desmond import constants
from schrodinger.application.desmond import envir
from schrodinger.application.desmond import struc
from schrodinger.application.desmond import util
from schrodinger.application.desmond.cns_io import write_cns
from schrodinger.application.desmond.mxmd import mxmd_system_builder as msb
from schrodinger.structutils.analyze import evaluate_asl
from schrodinger import structure



def normalize_probe_occupancy_grid(grid):
    '''
    Convert the grid counts to z-scores by normalizing the grid. Since the grid
    contains probe occupancies, this matrix is mostly sparse (with ~1% of non-
    zero values) we need to omit all the zero-containing values from calculating
    of mean and standard deviation.
    '''
    normgrid = np.zeros(grid.shape, dtype='float16')
    mask = grid != 0.0
    normgrid[mask] = grid[mask] - grid[mask].mean()
    normgrid /= grid[mask].std()
    return normgrid

def water_analysis(jobname,cms_fname):
    grid_spacing =0.5
    align_asl = '(protein and atom.ptype CA)'
    #read in cms file 
    msys_model, cms_model, tr = traj_util.read_cms_and_traj(cms_fname)
    #get cosolvent structure 
    solvent_ct = struc.component_structures(cms_model).solvent
    
    box_length = solvent_ct.property.get('r_chorus_box_cz')
    
    solvent_aids=cms_model.get_fullsys_ct_atom_index_range(cms_model.comp_ct.index(solvent_ct))

    solvent_aids_noh = [
        aid for aid in solvent_aids
        if cms_model.fsys_ct.atom[aid].atomic_number != 1
    ]
    solvent_mols = solvent_ct.mol_total

    solvent_probe='water'
    # extract reference structure and use it as a reference to align
    # trajectory and cms_model
    ref_mae = struc.get_reference_ct(cms_model.fsys_ct)
    ref_pos = ref_mae.extract(evaluate_asl(ref_mae, align_asl)).getXYZ()
    center_pos = np.mean(ref_pos, axis=0)
    ref_gids = topo.asl2gids(cms_model,
                            align_asl,
                            include_pseudoatoms=False)
    tr = topo.superimpose(msys_model, ref_gids, tr, ref_pos)
    cms_model = topo.superimpose_cms(msys_model, ref_gids, cms_model,
                                    ref_pos)



    grid_spacing = [grid_spacing] * 3
    box_length = [box_length] * 3

    vmw = analysis.VolumeMapper(cms_model,
                            aids=solvent_aids_noh,
                            spacing=grid_spacing,
                            length=box_length,
                            center=center_pos,
                            normalize=False)

    grid = analysis.analyze(tr,vmw)
    # reduce precision
    grid = grid.astype('uint16')
    normgrid=normalize_probe_occupancy_grid(grid)
    # Write .cns and .raw files. The later file will be used by the
    # cleanup stage to generate aggregate data for each probe type.
    _, cms_fname = os.path.split(cms_fname)
    out_cns_fname = f'{jobname}-water-out.cns'
    out_raw_fname = f'{jobname}-water-out.raw'
    out_mae_fname = f'{jobname}-water-out.mae'
    out_probes_fname = f'{jobname}-water-probes.mae'
    cms_fname = os.path.join(os.getcwd(), cms_fname)
    write_cns(normgrid, box_length, grid_spacing, out_cns_fname,center=[center_pos[0],center_pos[1],center_pos[2]])
    # Save probe molecules from 20 frames into a CT for later use.
    #probes_ct = create_new_structure()
    solvent_probes_ct = create_new_structure()

    _pct = struc.delete_ffio_ff(solvent_ct)

    solvent_gids = topo.aids2gids(cms_model,
                                solvent_aids,
                                include_pseudoatoms=False)            
    nframes = len(tr)
    fr_interval = 1 if nframes < 20 else nframes // 20
    for fr in tr[::fr_interval]:
        _pct.setXYZ(fr.pos(solvent_gids))
        y = _pct.copy()
        for at in y.atom:
            at.property["s_user_ensemble_frame"] = "%s_%d" %(jobname, fr._index)
        solvent_probes_ct = solvent_probes_ct.merge(y)


    solvent_probes_ct.title = _pct.title

    
    solvent_probes_ct.write(out_probes_fname)


    ct = struc.delete_ffio_ff(
    struc.component_structures(cms_model).solute)

    
    ct.write(out_mae_fname)
    


    with open(out_raw_fname, 'wb') as fh:
        np.save(fh, grid)
    

    return



if __name__== '__main__':

    parser = argparse.ArgumentParser(description='water hotspot',usage='tbd',formatter_class=argparse.RawTextHelpFormatter)
    #argparse usage example: python md_workflow.py GIPR --analyze 30
    #all arguments with "--" are OPTIONAL, with exception to pname and jname. options: run only prep, run only MD, run only analysis, run md + analysis, run prep + md + analysis, \
    parser.add_argument('--cms',nargs=1,type=str,required=True,help='job name from the MD run')
    parser.add_argument('--outname',nargs=1,type=str,required=True,help='add the name of the probes you want to do the analysis for, separated by spaces. ex: --probes acetaldehyde pyrimidine')
    #parser.add_argument('--replicates',nargs=1,type=str,required=True,help='num reps')
    
    args = parser.parse_args()

    #extract user input
    cms = args.cms[0] 
    outname = args.outname[0]
    #replicates = int(args.replicates[0])

    #for i in range(0,replicates):
    water_analysis(outname,cms)

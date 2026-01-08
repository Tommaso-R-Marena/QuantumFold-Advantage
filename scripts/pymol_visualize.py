#!/usr/bin/env python3
"""PyMOL visualization script for protein structures.

Usage:
    pymol scripts/pymol_visualize.py -- --pdb structure.pdb
    
    Or from within PyMOL:
    run scripts/pymol_visualize.py
    load_and_style('structure.pdb')
"""

import argparse
from pathlib import Path
import sys

try:
    from pymol import cmd, stored
except ImportError:
    print("PyMOL not found. This script must be run with PyMOL.")
    print("Install: conda install -c conda-forge pymol-open-source")
    sys.exit(1)


def load_and_style(pdb_path: str, name: str = None):
    """Load PDB and apply publication-quality styling.
    
    Args:
        pdb_path: Path to PDB file
        name: Object name in PyMOL (defaults to filename)
    """
    if name is None:
        name = Path(pdb_path).stem
    
    # Load structure
    cmd.load(pdb_path, name)
    
    # Basic styling
    cmd.hide('everything', name)
    cmd.show('cartoon', name)
    cmd.color('cyan', name)
    
    # Color by secondary structure
    cmd.color('red', f'{name} and ss h')  # Helices
    cmd.color('yellow', f'{name} and ss s')  # Sheets
    cmd.color('green', f'{name} and ss l+')  # Loops
    
    # Set cartoon style
    cmd.set('cartoon_fancy_helices', 1)
    cmd.set('cartoon_smooth_loops', 1)
    cmd.set('cartoon_loop_radius', 0.3)
    
    # Lighting and rendering
    cmd.set('ray_trace_mode', 1)
    cmd.set('ray_shadows', 1)
    cmd.set('depth_cue', 1)
    cmd.set('specular', 1)
    cmd.set('ambient', 0.4)
    
    # Background
    cmd.bg_color('white')
    
    # Center and zoom
    cmd.center(name)
    cmd.zoom(name, 5)
    
    print(f"Loaded {name} with publication-quality styling")


def compare_structures(pdb1: str, pdb2: str, name1: str = 'structure1', name2: str = 'structure2'):
    """Load and compare two structures.
    
    Args:
        pdb1: First PDB file
        pdb2: Second PDB file
        name1: Name for first structure
        name2: Name for second structure
    """
    # Load structures
    cmd.load(pdb1, name1)
    cmd.load(pdb2, name2)
    
    # Align
    alignment = cmd.align(name1, name2)
    rmsd = alignment[0]
    
    print(f"RMSD after alignment: {rmsd:.2f} Å")
    
    # Style first structure
    cmd.hide('everything', name1)
    cmd.show('cartoon', name1)
    cmd.color('cyan', name1)
    cmd.set('cartoon_transparency', 0.5, name1)
    
    # Style second structure
    cmd.hide('everything', name2)
    cmd.show('cartoon', name2)
    cmd.color('magenta', name2)
    cmd.set('cartoon_transparency', 0.5, name2)
    
    # Center view
    cmd.center(name1)
    cmd.zoom('all', 5)
    
    return rmsd


def show_confidence(pdb_path: str, confidence_values: list = None):
    """Color structure by confidence scores.
    
    Args:
        pdb_path: Path to PDB file
        confidence_values: List of per-residue confidence scores (0-1)
    """
    name = Path(pdb_path).stem
    cmd.load(pdb_path, name)
    
    if confidence_values is None:
        # Generate mock confidence (decreasing from N to C terminus)
        import numpy as np
        n_residues = cmd.count_atoms(f'{name} and name CA')
        confidence_values = np.linspace(0.9, 0.6, n_residues)
    
    # Store confidence values
    stored.confidence = confidence_values
    
    # Assign B-factors (PyMOL uses these for coloring)
    cmd.alter(f'{name} and name CA', 'b=stored.confidence.pop(0)')
    
    # Color by B-factor (confidence)
    cmd.hide('everything', name)
    cmd.show('cartoon', name)
    cmd.spectrum('b', 'red_yellow_green', name)
    
    # Add scale bar
    cmd.set('cartoon_putty', 1, name)
    cmd.set('cartoon_putty_transform', 1, name)
    cmd.set('cartoon_putty_scale_min', 0.6)
    cmd.set('cartoon_putty_scale_max', 1.0)
    
    cmd.center(name)
    cmd.zoom(name, 5)
    
    print(f"Colored {name} by confidence (red=low, green=high)")


def export_image(output_path: str, width: int = 2400, height: int = 2400, dpi: int = 300):
    """Export high-quality image.
    
    Args:
        output_path: Output file path
        width: Image width in pixels
        height: Image height in pixels
        dpi: DPI for raster formats
    """
    # Set rendering parameters
    cmd.set('ray_trace_mode', 1)
    cmd.set('ray_shadows', 1)
    cmd.set('antialias', 2)
    
    # Render and save
    cmd.png(output_path, width=width, height=height, dpi=dpi, ray=1)
    print(f"Image saved to {output_path}")


def create_rotation_movie(
    pdb_path: str,
    output_path: str = 'rotation.mp4',
    frames: int = 360,
    fps: int = 30
):
    """Create rotation movie of structure.
    
    Args:
        pdb_path: Path to PDB file
        output_path: Output movie path
        frames: Number of frames
        fps: Frames per second
    """
    name = Path(pdb_path).stem
    load_and_style(pdb_path, name)
    
    # Set up movie
    cmd.mset(f'1 x{frames}')
    
    # Rotate 360 degrees
    cmd.movie.produce(output_path, mode='draw', encoder='ffmpeg', fps=fps)
    
    for frame in range(frames):
        cmd.frame(frame + 1)
        cmd.rotate('y', 360.0 / frames)
        cmd.refresh()
    
    print(f"Movie saved to {output_path}")


def highlight_residues(pdb_path: str, residue_indices: list, color: str = 'red'):
    """Highlight specific residues.
    
    Args:
        pdb_path: Path to PDB file
        residue_indices: List of residue indices to highlight
        color: Color for highlighted residues
    """
    name = Path(pdb_path).stem
    load_and_style(pdb_path, name)
    
    # Create selection
    selection = '+'.join([str(i) for i in residue_indices])
    cmd.select('highlighted', f'{name} and resi {selection}')
    
    # Show as spheres
    cmd.show('spheres', 'highlighted')
    cmd.color(color, 'highlighted')
    cmd.set('sphere_scale', 1.5, 'highlighted')
    
    # Label
    cmd.label('highlighted and name CA', '"Res %s" % resi')
    
    print(f"Highlighted {len(residue_indices)} residues in {color}")


def main():
    """Command-line interface."""
    parser = argparse.ArgumentParser(description='PyMOL visualization for QuantumFold')
    parser.add_argument('--pdb', required=True, help='PDB file path')
    parser.add_argument('--compare', help='Second PDB file for comparison')
    parser.add_argument('--output', help='Output image path')
    parser.add_argument('--movie', action='store_true', help='Create rotation movie')
    parser.add_argument('--highlight', nargs='+', type=int, help='Residues to highlight')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_structures(args.pdb, args.compare)
    elif args.highlight:
        highlight_residues(args.pdb, args.highlight)
    else:
        load_and_style(args.pdb)
    
    if args.output:
        export_image(args.output)
    
    if args.movie:
        movie_path = Path(args.pdb).stem + '_rotation.mp4'
        create_rotation_movie(args.pdb, movie_path)


if __name__ == '__main__':
    # Check if running in PyMOL or as script
    if 'pymol' in sys.modules:
        print("Running in PyMOL. Functions available:")
        print("  - load_and_style(pdb_path)")
        print("  - compare_structures(pdb1, pdb2)")
        print("  - show_confidence(pdb_path)")
        print("  - export_image(output_path)")
    else:
        main()

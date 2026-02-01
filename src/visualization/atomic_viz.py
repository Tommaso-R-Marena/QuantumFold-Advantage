"""World-class atomic-level visualization for protein structures.

Provides:
- Interactive 3D molecular visualization with py3Dmol and nglview
- Quantum circuit visualization
- Attention mechanism heatmaps
- Dynamic trajectory rendering
- Contact maps with secondary structure
- Ramachandran plots
- Publication-quality figures
"""

import base64
import io
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.colors import LinearSegmentedColormap
from scipy.spatial.distance import pdist, squareform

try:
    import py3Dmol

    PY3DMOL_AVAILABLE = True
except ImportError:
    PY3DMOL_AVAILABLE = False

try:
    import nglview as nv

    NGLVIEW_AVAILABLE = True
except ImportError:
    NGLVIEW_AVAILABLE = False


class ProteinVisualizer:
    """High-quality protein structure visualization."""

    def __init__(self, style: str = "publication"):
        """Initialize visualizer.

        Args:
            style: Visualization style ('publication', 'presentation', 'web')
        """
        self.style = style
        self._setup_style()

    def _setup_style(self):
        """Configure matplotlib for publication-quality figures."""
        if self.style == "publication":
            plt.rcParams.update(
                {
                    "font.size": 11,
                    "font.family": "serif",
                    "font.serif": ["Computer Modern Roman"],
                    "text.usetex": False,  # Set True if LaTeX available
                    "figure.dpi": 300,
                    "savefig.dpi": 300,
                    "figure.figsize": (8, 6),
                    "axes.linewidth": 1.5,
                    "grid.linewidth": 0.5,
                    "lines.linewidth": 2,
                    "patch.linewidth": 1,
                    "xtick.major.width": 1.5,
                    "ytick.major.width": 1.5,
                }
            )

    def visualize_3d_structure(
        self,
        coords: np.ndarray,
        sequence: str,
        confidence: Optional[np.ndarray] = None,
        secondary_structure: Optional[str] = None,
        width: int = 800,
        height: int = 600,
        style: str = "cartoon",
        color_by: str = "confidence",
    ) -> str:
        """Create interactive 3D visualization with py3Dmol.

        Args:
            coords: Protein coordinates (N, 3)
            sequence: Amino acid sequence
            confidence: Per-residue confidence scores (pLDDT)
            secondary_structure: DSSP secondary structure string
            width: Viewer width
            height: Viewer height
            style: Representation ('cartoon', 'stick', 'sphere', 'line')
            color_by: Color scheme ('confidence', 'secondary_structure', 'rainbow')

        Returns:
            HTML string for embedding in notebook
        """
        if not PY3DMOL_AVAILABLE:
            return "<p>py3Dmol not available. Install with: pip install py3Dmol</p>"

        # Convert to PDB format
        pdb_string = self._coords_to_pdb(coords, sequence, confidence)

        # Create viewer
        view = py3Dmol.view(width=width, height=height)
        view.addModel(pdb_string, "pdb")

        # Apply style
        if color_by == "confidence" and confidence is not None:
            # Color by confidence (pLDDT-style)
            for i, conf in enumerate(confidence):
                color = self._plddt_to_color(conf)
                view.setStyle({"resi": i + 1}, {style: {"color": color}})
        elif color_by == "secondary_structure" and secondary_structure:
            # Color by secondary structure
            ss_colors = {
                "H": "#FF0080",  # Helix - pink
                "E": "#FFC800",  # Sheet - yellow
                "C": "#00BFFF",  # Coil - cyan
            }
            for i, ss in enumerate(secondary_structure):
                color = ss_colors.get(ss, "#CCCCCC")
                view.setStyle({"resi": i + 1}, {style: {"color": color}})
        elif color_by == "rainbow":
            view.setStyle({}, {style: {"colorscheme": "chain"}})
        else:
            view.setStyle({}, {style: {"color": "spectrum"}})

        # Add surface if cartoon
        if style == "cartoon":
            view.addSurface(py3Dmol.VDW, {"opacity": 0.7, "color": "white"})

        view.zoomTo()
        view.spin(True)

        return view._make_html()

    def plot_ramachandran(
        self,
        coords: np.ndarray,
        sequence: str,
        secondary_structure: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10),
    ) -> plt.Figure:
        """Generate Ramachandran plot with secondary structure highlighting.

        Args:
            coords: CA coordinates (N, 3) or backbone (N, 4, 3)
            sequence: Amino acid sequence
            secondary_structure: DSSP string for coloring
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Compute phi/psi angles
        phi, psi = self._compute_dihedral_angles(coords)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot allowed regions (background)
        self._plot_ramachandran_background(ax)

        # Plot points
        if secondary_structure:
            # Color by secondary structure
            colors = {"H": "#FF0080", "E": "#FFC800", "C": "#00BFFF"}
            for ss_type, color in colors.items():
                mask = np.array([s == ss_type for s in secondary_structure[1:-1]])
                if mask.any():
                    ax.scatter(
                        phi[mask],
                        psi[mask],
                        c=color,
                        s=50,
                        alpha=0.7,
                        label=ss_type,
                        edgecolors="black",
                        linewidth=0.5,
                    )
            ax.legend(title="Secondary Structure", loc="upper right")
        else:
            # Color by sequence position
            scatter = ax.scatter(
                phi,
                psi,
                c=np.arange(len(phi)),
                cmap="viridis",
                s=50,
                alpha=0.7,
                edgecolors="black",
                linewidth=0.5,
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Residue Index", rotation=270, labelpad=20)

        ax.set_xlabel("φ (degrees)", fontsize=14)
        ax.set_ylabel("ψ (degrees)", fontsize=14)
        ax.set_title("Ramachandran Plot", fontsize=16, fontweight="bold")
        ax.set_xlim(-180, 180)
        ax.set_ylim(-180, 180)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_contact_map(
        self,
        coords: np.ndarray,
        sequence: str,
        threshold: float = 8.0,
        secondary_structure: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
    ) -> plt.Figure:
        """Generate contact map with secondary structure annotations.

        Args:
            coords: CA coordinates (N, 3)
            sequence: Amino acid sequence
            threshold: Distance threshold for contact (Angstroms)
            secondary_structure: DSSP string for annotations
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Compute pairwise distances
        distances = squareform(pdist(coords))
        contacts = distances < threshold

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot contact map
        im = ax.imshow(
            contacts, cmap="RdBu_r", aspect="equal", origin="lower", interpolation="nearest"
        )

        # Add secondary structure bars
        if secondary_structure:
            self._add_secondary_structure_bars(ax, secondary_structure, len(sequence))

        # Styling
        ax.set_xlabel("Residue Index", fontsize=14)
        ax.set_ylabel("Residue Index", fontsize=14)
        ax.set_title(f"Contact Map (threshold = {threshold} Å)", fontsize=16, fontweight="bold")

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Contact", rotation=270, labelpad=20)

        plt.tight_layout()
        return fig

    def plot_attention_heatmap(
        self,
        attention_weights: np.ndarray,
        sequence: str,
        layer_idx: int = 0,
        head_idx: int = 0,
        figsize: Tuple[int, int] = (14, 12),
    ) -> plt.Figure:
        """Visualize attention weights as heatmap.

        Args:
            attention_weights: Attention matrix (heads, N, N) or (N, N)
            sequence: Amino acid sequence
            layer_idx: Layer index for title
            head_idx: Attention head index
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        # Handle multi-head attention
        if attention_weights.ndim == 3:
            attn = attention_weights[head_idx]
        else:
            attn = attention_weights

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot heatmap
        sns.heatmap(
            attn,
            cmap="YlOrRd",
            square=True,
            cbar_kws={"label": "Attention Weight"},
            ax=ax,
            vmin=0,
            vmax=attn.max(),
        )

        # Add sequence labels (every 10 residues)
        step = max(1, len(sequence) // 20)
        ticks = np.arange(0, len(sequence), step)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels([f"{sequence[i]}{i+1}" for i in ticks], rotation=45, ha="right")
        ax.set_yticklabels([f"{sequence[i]}{i+1}" for i in ticks])

        ax.set_xlabel("Target Residue", fontsize=14)
        ax.set_ylabel("Query Residue", fontsize=14)
        ax.set_title(
            f"Attention Heatmap (Layer {layer_idx}, Head {head_idx})",
            fontsize=16,
            fontweight="bold",
        )

        plt.tight_layout()
        return fig

    def plot_quantum_circuit(
        self,
        num_qubits: int,
        circuit_depth: int,
        gate_sequence: List[str],
        figsize: Tuple[int, int] = (14, 8),
    ) -> plt.Figure:
        """Visualize quantum circuit architecture.

        Args:
            num_qubits: Number of qubits
            circuit_depth: Circuit depth
            gate_sequence: List of gate types
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Draw qubit lines
        for i in range(num_qubits):
            ax.plot([0, circuit_depth + 1], [i, i], "k-", linewidth=2)
            ax.text(-0.5, i, f"q{i}", fontsize=12, ha="right", va="center")

        # Gate colors
        gate_colors = {
            "RX": "#FF6B6B",
            "RY": "#4ECDC4",
            "RZ": "#45B7D1",
            "CNOT": "#FFA07A",
            "Hadamard": "#98D8C8",
            "CZ": "#F7DC6F",
        }

        # Draw gates
        gate_idx = 0
        for depth in range(1, circuit_depth + 1):
            for qubit in range(num_qubits):
                if gate_idx < len(gate_sequence):
                    gate = gate_sequence[gate_idx]
                    color = gate_colors.get(gate, "#CCCCCC")

                    # Single qubit gate
                    if gate in ["RX", "RY", "RZ", "Hadamard"]:
                        circle = mpatches.Circle(
                            (depth, qubit), 0.2, facecolor=color, edgecolor="black", linewidth=2
                        )
                        ax.add_patch(circle)
                        ax.text(
                            depth,
                            qubit,
                            gate[:1],
                            fontsize=10,
                            ha="center",
                            va="center",
                            fontweight="bold",
                        )

                    # Two qubit gate
                    elif gate in ["CNOT", "CZ"] and qubit < num_qubits - 1:
                        # Control
                        circle = mpatches.Circle(
                            (depth, qubit), 0.1, facecolor="black", edgecolor="black"
                        )
                        ax.add_patch(circle)
                        # Target
                        circle = mpatches.Circle(
                            (depth, qubit + 1), 0.2, facecolor=color, edgecolor="black", linewidth=2
                        )
                        ax.add_patch(circle)
                        # Connection
                        ax.plot([depth, depth], [qubit, qubit + 1], "k-", linewidth=2)
                        ax.text(
                            depth,
                            qubit + 1,
                            "⊕" if gate == "CNOT" else "Z",
                            fontsize=12,
                            ha="center",
                            va="center",
                            fontweight="bold",
                        )

                    gate_idx += 1

        ax.set_xlim(-1, circuit_depth + 2)
        ax.set_ylim(-0.5, num_qubits - 0.5)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title("Quantum Circuit Architecture", fontsize=16, fontweight="bold")

        # Add legend
        legend_elements = [
            mpatches.Patch(facecolor=color, edgecolor="black", label=gate)
            for gate, color in gate_colors.items()
        ]
        ax.legend(handles=legend_elements, loc="upper right", fontsize=10)

        plt.tight_layout()
        return fig

    def create_trajectory_animation(
        self,
        trajectory: np.ndarray,
        sequence: str,
        output_path: str = "trajectory.gif",
        confidence: Optional[np.ndarray] = None,
        fps: int = 10,
    ) -> str:
        """Create animated GIF of structure refinement trajectory.

        Args:
            trajectory: Coordinates over time (T, N, 3)
            sequence: Amino acid sequence
            output_path: Path to save GIF
            confidence: Per-residue confidence
            fps: Frames per second

        Returns:
            Path to saved animation
        """
        from matplotlib.animation import FuncAnimation, PillowWriter

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection="3d")

        def update(frame):
            ax.clear()
            coords = trajectory[frame]

            # Plot backbone
            ax.plot(coords[:, 0], coords[:, 1], coords[:, 2], "o-", markersize=4)

            # Color by confidence if available
            if confidence is not None:
                colors = [self._plddt_to_rgb(c) for c in confidence]
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], c=colors, s=50, alpha=0.8)

            ax.set_xlabel("X (Å)")
            ax.set_ylabel("Y (Å)")
            ax.set_zlabel("Z (Å)")
            ax.set_title(f"Refinement Step {frame + 1}/{len(trajectory)}")

            # Set consistent limits
            all_coords = trajectory.reshape(-1, 3)
            margin = 5
            ax.set_xlim(all_coords[:, 0].min() - margin, all_coords[:, 0].max() + margin)
            ax.set_ylim(all_coords[:, 1].min() - margin, all_coords[:, 1].max() + margin)
            ax.set_zlim(all_coords[:, 2].min() - margin, all_coords[:, 2].max() + margin)

        anim = FuncAnimation(fig, update, frames=len(trajectory), interval=1000 // fps)
        writer = PillowWriter(fps=fps)
        anim.save(output_path, writer=writer)
        plt.close()

        return output_path

    # Helper methods

    def _coords_to_pdb(
        self, coords: np.ndarray, sequence: str, confidence: Optional[np.ndarray] = None
    ) -> str:
        """Convert coordinates to PDB format string."""
        pdb_lines = []
        pdb_lines.append("HEADER    PREDICTED STRUCTURE")

        for i, (aa, coord) in enumerate(zip(sequence, coords)):
            if confidence is not None:
                b_factor = confidence[i] * 100  # pLDDT-style
            else:
                b_factor = 100.0

            pdb_line = (
                f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    "
                f"{coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
                f"  1.00{b_factor:6.2f}           C"
            )
            pdb_lines.append(pdb_line)

        pdb_lines.append("END")
        return "\n".join(pdb_lines)

    def _plddt_to_color(self, plddt: float) -> str:
        """Convert pLDDT score to color (AlphaFold scheme)."""
        if plddt > 90:
            return "#0053D6"  # Very high (blue)
        elif plddt > 70:
            return "#65CBF3"  # Confident (light blue)
        elif plddt > 50:
            return "#FFDB13"  # Low (yellow)
        else:
            return "#FF7D45"  # Very low (orange)

    def _plddt_to_rgb(self, plddt: float) -> Tuple[float, float, float]:
        """Convert pLDDT to RGB tuple."""
        color_hex = self._plddt_to_color(plddt)
        return tuple(int(color_hex[i : i + 2], 16) / 255 for i in (1, 3, 5))

    def _compute_dihedral_angles(self, coords: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute phi/psi dihedral angles from CA coordinates."""
        # Simplified calculation for CA-only
        # For full backbone, need N, CA, C atoms
        n = len(coords)
        phi = np.zeros(n - 2)
        psi = np.zeros(n - 2)

        for i in range(1, n - 1):
            # Approximate using CA positions
            v1 = coords[i] - coords[i - 1]
            v2 = coords[i + 1] - coords[i]

            # Pseudo-dihedral
            angle = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
            phi[i - 1] = np.degrees(angle)
            psi[i - 1] = np.degrees(angle)  # Simplified

        return phi, psi

    def _plot_ramachandran_background(self, ax):
        """Add allowed regions to Ramachandran plot."""
        # Favored regions (simplified)
        # Alpha helix
        alpha_region = mpatches.Rectangle(
            (-90, -60), 40, 80, facecolor="#E8F4F8", edgecolor="#B0D4E3", alpha=0.5, linewidth=2
        )
        ax.add_patch(alpha_region)

        # Beta sheet
        beta_region = mpatches.Rectangle(
            (-180, 100), 100, 60, facecolor="#FFF4E6", edgecolor="#FFD699", alpha=0.5, linewidth=2
        )
        ax.add_patch(beta_region)

    def _add_secondary_structure_bars(self, ax, ss_string: str, length: int):
        """Add secondary structure color bars to contact map."""
        ss_colors = {"H": "#FF0080", "E": "#FFC800", "C": "#00BFFF"}

        for axis in ["x", "y"]:
            for i, ss in enumerate(ss_string):
                color = ss_colors.get(ss, "#CCCCCC")
                if axis == "x":
                    ax.add_patch(
                        mpatches.Rectangle(
                            (i, -length * 0.02),
                            1,
                            length * 0.015,
                            facecolor=color,
                            edgecolor="none",
                        )
                    )
                else:
                    ax.add_patch(
                        mpatches.Rectangle(
                            (-length * 0.02, i),
                            length * 0.015,
                            1,
                            facecolor=color,
                            edgecolor="none",
                        )
                    )

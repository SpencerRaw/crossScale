"""
AA MD simulation of short amphiphilic peptides in implicit solvent.
Generates short trajectories for feature extraction.

Usage:
    python -m modules.A_triScale.aa_md.run_md

Outputs:
    data/trajectories/    — DCD trajectory files
    data/topology.pdb     — System topology (with hydrogens)
"""

import time
import numpy as np
from pathlib import Path

# ——— paths ————————————————————————————————————————————————
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = PROJECT_ROOT / "data" / "trajectories"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ——— simulation parameters ——————————————————————————————————
PEPTIDE_SEQUENCE = "ACE-A-F-A-F-A-F-A-K-NME"  # amphiphilic: Ala/Phe/Lys
N_COPIES = 2
N_TRAJECTORIES = 8
TRAJ_LENGTH_NS = 10.0
TIMESTEP_FS = 2.0
REPORT_INTERVAL_PS = 10.0
TEMPERATURE_K = 300.0
FRICTION_COEFF = 1.0
SALT_CONC_M = 0.15

# ——— forcefield —————————————————————————————————————————————
FORCEFIELD_FILES = ["amber14/protein.ff14SB.xml", "implicit/gbn2.xml"]


def _make_pdb_string(sequence_str, n_copies):
    """
    Generate a minimal PDB string for the peptide with extended geometry.

    This is the cleanest way to get OpenMM to recognize atom types:
    write a proper PDB file and let Modeller add hydrogens via the
    forcefield template matching.

    Returns: (pdb_string, positions_nm)
    """
    sequence_raw = sequence_str.replace("ACE-", "").replace("-NME", "")
    res_codes = sequence_raw.split("-")

    res_map = {"A": "ALA", "F": "PHE", "K": "LYS"}

    # Ideal backbone bond lengths (nm) for extended chain
    BOND_N_CA = 0.147
    BOND_CA_C = 0.153
    BOND_C_N = 0.132

    atoms = []    # (serial, name, resName, chainID, resSeq, x, y, z)
    positions_nm = []

    atom_serial = 0

    for copy_idx in range(n_copies):
        chain_id = chr(ord("A") + copy_idx)
        offset_x = copy_idx * 4.0  # nm between copies

        # Start from origin for each chain
        pos = np.array([offset_x, 0.0, 0.0])
        res_seq = 1

        # ── ACE cap (CH3-CO-) ──
        ace_atoms = [
            ("CH3", -0.150, 0.000, 0.000),   # methyl carbon
            ("C",    0.000, 0.000, 0.000),   # carbonyl carbon
            ("O",    0.000, 0.123, 0.000),   # carbonyl oxygen
        ]
        for aname, dx, dy, dz in ace_atoms:
            atom_serial += 1
            x, y, z = pos[0] + dx, pos[1] + dy, pos[2] + dz
            atoms.append((atom_serial, aname, "ACE", chain_id, res_seq, x, y, z))
            positions_nm.append([x, y, z])

        # Position for first residue N (after ACE-C)
        ace_c_pos = pos + np.array([0.0, 0.0, 0.0])
        backbone_n_pos = ace_c_pos + np.array([BOND_C_N, 0.0, 0.0])

        res_seq += 1

        # ── Standard residues ──
        for ri, res_code in enumerate(res_codes):
            res3 = res_map.get(res_code, "ALA")

            # N
            atom_serial += 1
            pos_n = backbone_n_pos.copy()
            atoms.append((atom_serial, "N", res3, chain_id, res_seq,
                          pos_n[0], pos_n[1], pos_n[2]))
            positions_nm.append(pos_n.tolist())

            # CA (at N-CA distance along x)
            ca_pos = pos_n + np.array([BOND_N_CA, 0.0, 0.0])
            atom_serial += 1
            atoms.append((atom_serial, "CA", res3, chain_id, res_seq,
                          ca_pos[0], ca_pos[1], ca_pos[2]))
            positions_nm.append(ca_pos.tolist())

            # C
            c_pos = ca_pos + np.array([BOND_CA_C, 0.0, 0.0])
            atom_serial += 1
            atoms.append((atom_serial, "C", res3, chain_id, res_seq,
                          c_pos[0], c_pos[1], c_pos[2]))
            positions_nm.append(c_pos.tolist())

            # O (carbonyl oxygen — out of backbone plane)
            o_pos = c_pos + np.array([0.0, BOND_CA_C * 0.8, 0.0])
            atom_serial += 1
            atoms.append((atom_serial, "O", res3, chain_id, res_seq,
                          o_pos[0], o_pos[1], o_pos[2]))
            positions_nm.append(o_pos.tolist())

            # CB
            cb_pos = ca_pos + np.array([-0.05, -0.05, 0.07])
            atom_serial += 1
            atoms.append((atom_serial, "CB", res3, chain_id, res_seq,
                          cb_pos[0], cb_pos[1], cb_pos[2]))
            positions_nm.append(cb_pos.tolist())

            # Sidechain heavy atoms for PHE
            if res3 == "PHE":
                extra = [
                    ("CG",  0.05, -0.05, 0.10),
                    ("CD1", 0.05,  0.05, 0.15),
                    ("CD2", 0.05, -0.15, 0.15),
                    ("CE1", 0.05,  0.05, 0.20),
                    ("CE2", 0.05, -0.15, 0.20),
                    ("CZ",  0.05, -0.05, 0.25),
                ]
                for aname, dx, dy, dz in extra:
                    atom_serial += 1
                    x, y, z = cb_pos[0] + dx, cb_pos[1] + dy, cb_pos[2] + dz
                    atoms.append((atom_serial, aname, res3, chain_id, res_seq, x, y, z))
                    positions_nm.append([x, y, z])

            # Sidechain heavy atoms for LYS
            elif res3 == "LYS":
                extra = [
                    ("CG",  0.05, -0.05, 0.10),
                    ("CD",  0.05, -0.05, 0.15),
                    ("CE",  0.05, -0.05, 0.20),
                    ("NZ",  0.05, -0.05, 0.25),
                ]
                for aname, dx, dy, dz in extra:
                    atom_serial += 1
                    x, y, z = cb_pos[0] + dx, cb_pos[1] + dy, cb_pos[2] + dz
                    atoms.append((atom_serial, aname, res3, chain_id, res_seq, x, y, z))
                    positions_nm.append([x, y, z])

            # Next residue N position
            backbone_n_pos = c_pos + np.array([BOND_C_N, 0.0, 0.0])
            res_seq += 1

        # ── NME cap (-NH-CH3) ──
        atom_serial += 1
        atoms.append((atom_serial, "N", "NME", chain_id, res_seq,
                      backbone_n_pos[0], backbone_n_pos[1], backbone_n_pos[2]))
        positions_nm.append(backbone_n_pos.tolist())

        nme_ch3_pos = backbone_n_pos + np.array([BOND_N_CA, 0.0, 0.0])
        atom_serial += 1
        atoms.append((atom_serial, "CH3", "NME", chain_id, res_seq,
                      nme_ch3_pos[0], nme_ch3_pos[1], nme_ch3_pos[2]))
        positions_nm.append(nme_ch3_pos.tolist())

    # Build PDB string
    pdb_lines = []
    for (serial, aname, resn, chain, resseq, x, y, z) in atoms:
        # PDB ATOM record: columns 1-6 "ATOM  ", 7-11 serial, 13-16 atom name,
        # 17 altLoc, 18-20 resName, 22 chainID, 23-26 resSeq,
        # 31-38 x, 39-46 y, 47-54 z
        pdb_lines.append(
            f"ATOM  {serial:5d} {aname:4s} {resn:3s} {chain}{resseq:4d}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}"
        )
    pdb_lines.append("TER")
    pdb_lines.append("END")

    return "\n".join(pdb_lines), np.array(positions_nm)


def _build_system():
    """
    Build the peptide system using OpenMM Modeller for robust hydrogen
    placement via forcefield template matching.
    """
    import openmm as mm
    from openmm import app, unit

    print("Generating peptide PDB …")
    pdb_string, positions_nm = _make_pdb_string(PEPTIDE_SEQUENCE, N_COPIES)

    # Save the heavy-atom PDB for reference
    heavy_pdb_path = DATA_DIR.parent / "topology_heavy.pdb"
    with open(heavy_pdb_path, "w") as f:
        f.write(pdb_string)

    # Parse PDB
    from io import StringIO
    pdb_file = app.PDBFile(StringIO(pdb_string))

    print(f"Heavy-atom topology: {pdb_file.topology.getNumAtoms()} atoms")

    # Use Modeller to add hydrogens using forcefield templates
    forcefield = app.ForceField(*FORCEFIELD_FILES)
    modeller = app.Modeller(pdb_file.topology, pdb_file.positions)
    print("Adding hydrogens via Modeller …")

    try:
        modeller.addHydrogens(forcefield)
    except Exception as e:
        print(f"  addHydrogens failed: {e}")
        print("  Trying with protein-only hydrogen addition …")
        # Fallback: add hydrogens using a simpler variant
        variants = [
            ["amber14/protein.ff14SB.xml"],
            ["amber99sb.xml"],
        ]
        for variant in variants:
            try:
                ff2 = app.ForceField(*variant)
                modeller.addHydrogens(ff2)
                print(f"  Success with {variant}")
                break
            except Exception:
                continue
        else:
            print("  WARNING: Could not add hydrogens. "
                  "Proceeding with heavy atoms only (no constraints).")

    print(f"Final topology: {modeller.topology.getNumAtoms()} atoms "
          f"(including hydrogens)")

    # Create system (with or without hydrogen constraints)
    has_hydrogens = modeller.topology.getNumAtoms() > pdb_file.topology.getNumAtoms()

    if has_hydrogens:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * unit.nanometer,
            constraints=app.HBonds,
        )
    else:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.CutoffNonPeriodic,
            nonbondedCutoff=2.0 * unit.nanometer,
        )

    topology = modeller.topology
    final_positions = np.array(
        modeller.positions.value_in_unit(unit.nanometer)
    )

    # Save topology with hydrogens
    topo_path = str(DATA_DIR.parent / "topology.pdb")
    with open(topo_path, "w") as f:
        app.PDBFile.writeFile(
            topology,
            modeller.positions,
            f,
        )
    print(f"Topology saved → {topo_path}")

    return system, topology, final_positions


def run_single_trajectory(traj_id, system, topology, positions_nm, output_dir):
    """Run one MD trajectory."""
    import openmm as mm
    from openmm import app, unit

    print(f"  Trajectory {traj_id + 1}/{N_TRAJECTORIES} …")

    integrator = mm.LangevinMiddleIntegrator(
        TEMPERATURE_K * unit.kelvin,
        FRICTION_COEFF / unit.picosecond,
        TIMESTEP_FS * unit.femtoseconds,
    )

    # Platform selection
    try:
        platform = mm.Platform.getPlatformByName("CUDA")
        print("    [CUDA]")
    except Exception:
        try:
            platform = mm.Platform.getPlatformByName("OpenCL")
            print("    [OpenCL]")
        except Exception:
            platform = mm.Platform.getPlatformByName("CPU")
            print("    [CPU — slow!]")

    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions_nm * unit.nanometer)
    simulation.context.setVelocitiesToTemperature(
        TEMPERATURE_K * unit.kelvin, 42 + traj_id
    )

    # Energy minimization
    print("    Minimizing energy …")
    simulation.minimizeEnergy(maxIterations=1000)

    # Check initial energy is reasonable
    state = simulation.context.getState(getEnergy=True)
    pot_energy = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    print(f"    Initial PE: {pot_energy:.0f} kJ/mol")
    if abs(pot_energy) > 1e6:
        print("    ✗ CRITICAL: Potential energy is unreasonably high!")
        print("    This indicates bad initial geometry or forcefield mismatch.")
        print("    Skip this trajectory, continue with others.")
        return

    # Equilibration: 50ps NVT
    n_steps_eq = int(50000 / TIMESTEP_FS)
    simulation.step(n_steps_eq)

    # Production
    n_steps_prod = int(TRAJ_LENGTH_NS * 1e6 / TIMESTEP_FS)
    report_interval = int(REPORT_INTERVAL_PS * 1000 / TIMESTEP_FS)
    n_frames = n_steps_prod // report_interval

    dcd_path = str(output_dir / f"traj_{traj_id:02d}.dcd")
    simulation.reporters.append(app.DCDReporter(dcd_path, report_interval))

    data_path = str(output_dir / f"energy_{traj_id:02d}.csv")
    simulation.reporters.append(
        app.StateDataReporter(
            data_path, report_interval,
            step=True, potentialEnergy=True, temperature=True, speed=True,
        )
    )

    chk_path = str(output_dir / f"chk_{traj_id:02d}.xml")
    simulation.reporters.append(
        app.CheckpointReporter(chk_path, n_steps_prod // 2)
    )

    print(f"    Running {TRAJ_LENGTH_NS:.0f} ns "
          f"({n_steps_prod} steps, saving {n_frames} frames) …")
    t0 = time.time()
    simulation.step(n_steps_prod)
    elapsed = time.time() - t0
    ns_per_day = (TRAJ_LENGTH_NS / max(elapsed, 1)) * 86400
    print(f"    Done in {elapsed:.0f}s → ~{ns_per_day:.0f} ns/day")

    # Save final state PDB
    state = simulation.context.getState(getPositions=True)
    pdb_path = str(output_dir / f"final_{traj_id:02d}.pdb")
    with open(pdb_path, "w") as f:
        app.PDBFile.writeFile(simulation.topology, state.getPositions(), f)


def run_all():
    """Main entry: build system and run all trajectories."""
    print("=" * 60)
    print("Module A — AA MD: Peptide Simulation")
    print(f"Peptide: {PEPTIDE_SEQUENCE}")
    print(f"Copies: {N_COPIES}")
    print(f"Trajectories: {N_TRAJECTORIES} × {TRAJ_LENGTH_NS:.0f} ns")
    print(f"Output: {DATA_DIR}")
    print("=" * 60)

    # Build system
    system, topology, positions_nm = _build_system()

    # Run trajectories
    for traj_id in range(N_TRAJECTORIES):
        try:
            run_single_trajectory(
                traj_id, system, topology, positions_nm, DATA_DIR
            )
        except Exception as e:
            print(f"  ✗ Trajectory {traj_id + 1} failed: {e}")
            print(f"  Continuing with remaining trajectories …")

    # Check results
    dcd_files = sorted(DATA_DIR.glob("traj_*.dcd"))
    print(f"\nCompleted: {len(dcd_files)}/{N_TRAJECTORIES} trajectories")
    print(f"Data in: {DATA_DIR}")


if __name__ == "__main__":
    run_all()

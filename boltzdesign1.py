import os
import sys
import argparse
import yaml
import json
import shutil
import pickle
import glob
import numpy as np
import random
import logging
import subprocess
import pandas as pd
from pathlib import Path
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
sys.path.append(f'{os.getcwd()}/boltzdesign')
sys.path.append(f'{os.getcwd()}/boltz/src')

from boltzdesign_utils import *
from ligandmpnn_utils import *
from alphafold_utils import *
from input_utils import *
from utils import *
import torch


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def setup_gpu_environment(gpu_id):
    """Setup GPU environment variables"""
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="BoltzDesign: Protein Design Pipeline with Aptamer Support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Design protein binder for DNA target (Original mode)
  python boltzdesign.py --target_name 5zmc --target_type dna --pdb_target_ids C,D --target_mols SAM --binder_id A
  
  # Design RNA aptamer for protein target (NEW: Aptamer mode)
  python boltzdesign.py --design_mode aptamer --aptamer_type RNA --target_protein_seq "MKLLVVVGGVGSGKTTLLRQLAKEFG" --aptamer_length 40
  
  # Design DNA aptamer for protein target (NEW: Aptamer mode)  
  python boltzdesign.py --design_mode aptamer --aptamer_type DNA --target_protein_seq "MKLLVVVGGVGSGKTTLLRQLAKEFG" --aptamer_length 50
  
  # Design RNA aptamer for small molecule target (NEW: Ligand Aptamer mode)
  python boltzdesign.py --design_mode aptamer --aptamer_type RNA --target_ligand_smiles "N[C@@H](Cc1ccc(O)cc1)C(=O)O" --aptamer_length 30
  
  # Design DNA aptamer for drug molecule target (NEW: Ligand Aptamer mode)
  python boltzdesign.py --design_mode aptamer --aptamer_type DNA --target_ligand_smiles "CC(C)N(C(=O)CCC)C(C)c1ccccc1" --aptamer_length 35
        """
    )
    
    # Required arguments (在适配体模式下可选)
    parser.add_argument('--target_name', type=str, default='target',
                        help='Target name/PDB code (e.g., 5zmc) or custom name for aptamer mode')
    # Target configuration
    parser.add_argument('--target_type', type=str, choices=['protein', 'rna', 'dna', 'small_molecule', 'metal'],
                        default='protein', help='Type of target molecule')
    parser.add_argument('--input_type', type=str, choices=['pdb', 'custom'], default='pdb',
                        help='Input type: pdb code or custom input')
    parser.add_argument('--pdb_path', type=str, default='',
                        help='Path to a local PDB file (if specify use custom pdb, else fetch from RCSB)')
    parser.add_argument('--pdb_target_ids', type=str, default='',
                        help='Target PDB IDs (comma-separated, e.g., "C,D")')
    parser.add_argument('--target_mols', type=str, default='',
                        help='Target molecules for small molecules (comma-separated, e.g., "SAM,FAD")')
    parser.add_argument('--custom_target_input', type=str, default='',
                        help='Custom target sequences/ligand(smiles)/dna/rna/metals (comma-separated, e.g., "ATAT,GCGC", "[O-]C(=O)C(N)CC[S+](C)CC3OC(n2cnc1c(ncnc12)N)C(O)C3O", "ZN")')
    parser.add_argument('--custom_target_ids', type=str, default='',
                        help='Custom target IDs (comma-separated, e.g., "A,B")')
    parser.add_argument('--binder_id', type=str, default='A',
                        help='Binder chain ID')
    parser.add_argument('--use_msa', type=str2bool, default=False,
                        help='Use MSA (if False, runs in single-sequence mode)')
    parser.add_argument('--msa_max_seqs', type=int, default=4096,
                        help='Maximum MSA sequences')
    parser.add_argument('--suffix', type=str, default='0',
                        help='Suffix for the output directory')
    
    # 新增: 适配体设计参数
    parser.add_argument('--design_mode', type=str, choices=['protein', 'aptamer', 'predict_structure'], 
                       default='protein', help='Design mode: protein binder (original), aptamer design (new), or predict_structure (structure prediction only)')
    parser.add_argument('--aptamer_type', type=str, choices=['RNA', 'DNA'], 
                       default='RNA', help='Type of aptamer to design (RNA or DNA)')
    parser.add_argument('--target_protein_seq', type=str, default='',
                       help='Target protein sequence for aptamer binding (required in aptamer mode)')
    parser.add_argument('--target_ligand_smiles', type=str, default='',
                       help='Target ligand SMILES string for aptamer binding (alternative to protein)')
    parser.add_argument('--aptamer_length', type=int, default=40,
                       help='Length of aptamer to design (20-80 recommended)')
    parser.add_argument('--aptamer_chain', type=str, default='A',
                       help='Chain ID for the designed aptamer')
    parser.add_argument('--target_protein_chains', type=str, default='B',
                       help='Chain IDs for target protein (comma-separated if multiple)')
    parser.add_argument('--target_ligand_chains', type=str, default='B',
                       help='Chain IDs for target ligand (comma-separated if multiple)')
    
    # 结构预测参数
    parser.add_argument('--predict_structure_only', type=str2bool, default=False,
                       help='Only predict structure from existing aptamer YAML file')
    parser.add_argument('--input_yaml', type=str, default='',
                       help='Input YAML file for structure prediction only mode')
    parser.add_argument('--save_structures', type=str2bool, default=True,
                       help='Save final structures in CIF and PDB formats')
    parser.add_argument('--output_format', type=str, choices=['cif', 'pdb', 'both'], default='both',
                       help='Output format for structure files')
    parser.add_argument('--structure_output_dir', type=str, default='',
                       help='Custom directory for structure outputs (default: same as aptamer output)')
    
    # Modifications
    parser.add_argument('--modifications', type=str, default='',
                        help='Modifications (comma-separated, e.g., "SEP,SEP")')
    parser.add_argument('--modifications_wt', type=str, default='',
                        help='Modifications (comma-separated, e.g., "S,S")')
    parser.add_argument('--modifications_positions', type=str, default='',
                        help='Modification positions (comma-separated, matching order)')
    parser.add_argument('--modification_target', type=str, default='',
                        help='Target ID for modifications (e.g., "A")')
    
    # Constraints
    parser.add_argument('--constraint_target', type=str, default='',
                        help='Target ID for constraints (e.g., "A")')
    parser.add_argument('--contact_residues', type=str, default='',
                        help='Contact residues for constraints (comma-separated, e.g., "99,100,109")')

    # Design parameters
    parser.add_argument('--length_min', type=int, default=100,
                        help='Minimum binder length')
    parser.add_argument('--length_max', type=int, default=150,
                        help='Maximum binder length')
    parser.add_argument('--optimizer_type', type=str, choices=['SGD', 'AdamW'], default='SGD',
                        help='Optimizer type')
    
    # Iteration parameters
    parser.add_argument('--pre_iteration', type=int, default=30,
                        help='Pre-iteration steps')
    parser.add_argument('--soft_iteration', type=int, default=75,
                        help='Soft iteration steps')
    parser.add_argument('--temp_iteration', type=int, default=50,
                        help='Temperature iteration steps')
    parser.add_argument('--hard_iteration', type=int, default=5,
                        help='Hard iteration steps')
    parser.add_argument('--semi_greedy_steps', type=int, default=2,
                        help='Semi-greedy steps')
    parser.add_argument('--recycling_steps', type=int, default=0,
                        help='Recycling steps')
    
    # Advanced configuration
    parser.add_argument('--use_default_config', type=str2bool, default=True,
                        help='Use default configuration (recommended)')
    parser.add_argument('--mask_ligand', type=str2bool, default=False,
                        help='Mask target for warm-up stage')
    parser.add_argument('--optimize_contact_per_binder_pos', type=str2bool, default=False,
                        help='Optimize interface contact per binder position')
    parser.add_argument('--distogram_only', type=str2bool, default=True,
                        help='Only use distogram for optimization')
    parser.add_argument('--design_algorithm', type=str, choices=['3stages', '3stages_extra'], 
                        default='3stages', help='Design algorithm')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='Learning rate for optimization')
    parser.add_argument('--learning_rate_pre', type=float, default=0.1, 
                        help='Learning rate for pre iterations (warm-up stage)')
    parser.add_argument('--e_soft', type=float, default=0.8,
                        help='Softmax temperature for 3stages')
    parser.add_argument('--e_soft_1', type=float, default=0.8,
                        help='Initial softmax temperature for 3stages_extra')
    parser.add_argument('--e_soft_2', type=float, default=1.0,
                        help='Additional softmax temperature for 3stages_extra')
    
    # Interaction parameters
    parser.add_argument('--inter_chain_cutoff', type=int, default=20,
                        help='Inter-chain distance cutoff')
    parser.add_argument('--intra_chain_cutoff', type=int, default=14,
                        help='Intra-chain distance cutoff')
    parser.add_argument('--num_inter_contacts', type=int, default=1,
                        help='Number of inter-chain contacts')
    parser.add_argument('--num_intra_contacts', type=int, default=2,
                        help='Number of intra-chain contacts')
    

    # loss parameters
    parser.add_argument('--con_loss', type=float, default=1.0,
                        help='Contact loss weight')
    parser.add_argument('--i_con_loss', type=float, default=1.0,
                        help='Inter-chain contact loss weight')
    parser.add_argument('--plddt_loss', type=float, default=0.1,
                        help='pLDDT loss weight')
    parser.add_argument('--pae_loss', type=float, default=0.4,
                        help='PAE loss weight')
    parser.add_argument('--i_pae_loss', type=float, default=0.1,
                        help='Inter-chain PAE loss weight')
    parser.add_argument('--rg_loss', type=float, default=0.0,
                        help='Radius of gyration loss weight')
    parser.add_argument('--helix_loss_max', type=float, default=0.0,
                        help='Maximum helix loss weights')
    parser.add_argument('--helix_loss_min', type=float, default=-0.3,
                        help='Minimum helix loss weights')

    
    # LigandMPNN parameters
    parser.add_argument('--num_designs', type=int, default=2,
                        help='Number of designs per PDB for LigandMPNN')
    parser.add_argument('--cutoff', type=int, default=4,
                        help='Cutoff distance for interface residues (Angstroms)')
    parser.add_argument('--i_ptm_cutoff', type=float, default=0.5,
                        help='iPTM cutoff for redesign')
    parser.add_argument('--complex_plddt_cutoff', type=float, default=0.7,
                        help='Complex pLDDT cutoff for high confidence designs')
    
    # System configuration
    parser.add_argument('--gpu_id', type=int, default=0,
                        help='GPU ID to use')
    parser.add_argument('--design_samples', type=int, default=1,
                        help='Number of design samples')
    parser.add_argument('--work_dir', type=str, default=None,
                        help='Working directory (default: current directory)')
    parser.add_argument('--high_iptm', type=str2bool, default=True,
                        help='Disable high iPTM designs')
    # Paths
    parser.add_argument('--boltz_checkpoint', type=str,
        default='/home/zihao/.boltz/boltz1_conf.ckpt',
        help='Path to Boltz checkpoint')
    parser.add_argument('--ccd_path', type=str,
        default='/home/zihao/.boltz/ccd.pkl',
        help='Path to CCD file')
    parser.add_argument('--alphafold_dir', type=str,
        default='~/alphafold3',
        help='AlphaFold directory')
    parser.add_argument('--af3_docker_name', type=str,
        default='alphafold3',
        help='Docker name')
    parser.add_argument('--af3_database_settings', type=str,
        default='~/alphafold3/alphafold3_data_save',
        help='AlphaFold3 database settings')
    parser.add_argument('--af3_hmmer_path', type=str,
        default='/home/jupyter-yehlin/.conda/envs/alphafold3_venv',
        help='AlphaFold3 hmmer path, required for RNA MSA generation')
    # Control flags
    parser.add_argument('--run_boltz_design', type=str2bool, default=True,
                        help='Run Boltz design step')
    parser.add_argument('--run_ligandmpnn', type=str2bool, default=True,
                        help='Run LigandMPNN redesign step')
    parser.add_argument('--run_alphafold', type=str2bool, default=True,
                        help='Run AlphaFold validation step')
    parser.add_argument('--run_rosetta', type=str2bool, default=True,
                        help='Run Rosetta energy calculation (protein targets only)')
    parser.add_argument('--redo_boltz_predict', type=str2bool, default=False,
                        help='Redo Boltz prediction')


    ## Visualization
    parser.add_argument('--show_animation', type=str2bool, default=True,
                        help='Show animation')
    parser.add_argument('--save_trajectory', type=str2bool, default=False,
                        help='Save trajectory')
    return parser.parse_args()


class YamlConfig:
    """Configuration class for managing directories"""
    def __init__(self, main_dir: str = None):
        if main_dir is None:
            self.MAIN_DIR = Path.cwd() / 'inputs'
        else:
            self.MAIN_DIR = Path(main_dir)
        self.PDB_DIR = self.MAIN_DIR / 'PDB'
        self.MSA_DIR = self.MAIN_DIR / 'MSA'
        self.YAML_DIR = self.MAIN_DIR / 'yaml'
    
    def setup_directories(self):
        """Create necessary directories if they don't exist."""
        for directory in [self.MAIN_DIR, self.PDB_DIR, self.MSA_DIR, self.YAML_DIR]:
            directory.mkdir(parents=True, exist_ok=True)


def load_boltz_model(args, device):
    """Load Boltz model"""
    predict_args = {
        "recycling_steps": args.recycling_steps,
        "sampling_steps": 200,
        "diffusion_samples": 1,
        "write_confidence_summary": True,
        "write_full_pae": False,
        "write_full_pde": False,
    }
    
    boltz_model = get_boltz_model(args.boltz_checkpoint, predict_args, device)
    boltz_model.train()
    return boltz_model, predict_args

def load_design_config(target_type, work_dir):
    """
    Load design configuration based on target type.
    Modified so that config files are always loaded from the script's directory,
    instead of using work_dir/boltzdesign/configs.
    """
    # Determine the directory where this script (boltzdesign.py) lives:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # The configs directory is under script_dir/boltzdesign/configs/
    config_dir = os.path.join(script_dir, 'boltzdesign', 'configs')
    
    if target_type=='small_molecule':
        config_path = os.path.join(config_dir, "default_sm_config.yaml")
    elif target_type=='metal':
        config_path = os.path.join(config_dir, "default_metal_config.yaml")
    elif target_type=='dna' or target_type=='rna':
        config_path = os.path.join(config_dir, "default_na_config.yaml")

    elif target_type=='protein':
        config_path = os.path.join(config_dir, "default_ppi_config.yaml")
    else:
        raise ValueError(f"Unknown target type: {target_type}")
    
    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_explicit_args():
    # Get all command-line arguments (excluding the script name)
    explicit_args = set()
    for arg in sys.argv[1:]:
        if arg.startswith('--'):
            # Handle --arg=value and --arg value
            if '=' in arg:
                explicit_args.add(arg.split('=')[0].lstrip('-').replace('-', '_'))
            else:
                explicit_args.add(arg.lstrip('-').replace('-', '_'))
    return explicit_args

def update_config_with_args(config, args):
    """Update configuration with command line arguments"""
    # Always update these basic parameters regardless of use_default_config
    basic_params = {
    'binder_chain': args.binder_id,
    'non_protein_target': args.target_type != 'protein',
    'pocket_conditioning': bool(args.contact_residues),
    }

    # Update basic parameters
    explicit_args = get_explicit_args()
    config.update(basic_params)
    
    # For advanced parameters, only update those that are explicitly set by the user
    # (i.e., different from their default values in argparse)
    parser = argparse.ArgumentParser()
    _, defaults = parser.parse_known_args([])  # Get default values
    
    advanced_params = {
        'mask_ligand': args.mask_ligand,
        'optimize_contact_per_binder_pos': args.optimize_contact_per_binder_pos,
        'distogram_only': args.distogram_only,
        'design_algorithm': args.design_algorithm,
        'learning_rate': args.learning_rate,
        'learning_rate_pre': args.learning_rate_pre,
        'e_soft': args.e_soft,
        'e_soft_1': args.e_soft_1,
        'e_soft_2': args.e_soft_2,
        'length_min': args.length_min,
        'length_max': args.length_max,
        'inter_chain_cutoff': args.inter_chain_cutoff,
        'intra_chain_cutoff': args.intra_chain_cutoff,
        'num_inter_contacts': args.num_inter_contacts,
        'num_intra_contacts': args.num_intra_contacts,
        'helix_loss_max': args.helix_loss_max,
        'helix_loss_min': args.helix_loss_min,
        'optimizer_type': args.optimizer_type,
        'pre_iteration': args.pre_iteration,
        'soft_iteration': args.soft_iteration,
        'temp_iteration': args.temp_iteration,
        'hard_iteration': args.hard_iteration,
        'semi_greedy_steps': args.semi_greedy_steps,
        'msa_max_seqs': args.msa_max_seqs,
        'recycling_steps': args.recycling_steps,
    }

    for param_name, param_value in advanced_params.items():
        if param_name in explicit_args:
            print(f"Updating {param_name} to {param_value}")
            config[param_name] = param_value
    return config
    
def run_boltz_design_step(args, config, boltz_model, yaml_dir, main_dir, version_name):
    """Run the Boltz design step"""
    print("Starting Boltz design step...")
    
    loss_scales = {
        'con_loss': args.con_loss,
        'i_con_loss': args.i_con_loss,
        'plddt_loss': args.plddt_loss,
        'pae_loss': args.pae_loss,
        'i_pae_loss': args.i_pae_loss,
        'rg_loss': args.rg_loss,
    }
    
    boltz_path = shutil.which("boltz")
    if boltz_path is None:
        raise FileNotFoundError("The 'boltz' command was not found in the system PATH.")
    
    run_boltz_design(
        boltz_path=boltz_path,
        main_dir=main_dir,
        yaml_dir=os.path.dirname(yaml_dir),
        boltz_model=boltz_model,
        ccd_path=args.ccd_path,
        design_samples=args.design_samples,
        version_name=version_name,
        config=config,
        loss_scales=loss_scales,
        show_animation=args.show_animation,
        save_trajectory=args.save_trajectory,
        redo_boltz_predict=args.redo_boltz_predict,
    )
    
    print("Boltz design step completed!")

def run_ligandmpnn_step(args, main_dir, version_name, ligandmpnn_dir, yaml_dir, work_dir):
    """Run the LigandMPNN redesign step"""
    print("Starting LigandMPNN redesign step...")
    # Setup LigandMPNN config
    yaml_path = f"{work_dir}/LigandMPNN/run_ligandmpnn_logits_config.yaml"
    with open(yaml_path, "r") as f:
        mpnn_config = yaml.safe_load(f)
    
    for key, value in mpnn_config.items():
        if isinstance(value, str) and "${CWD}" in value:
            mpnn_config[key] = value.replace("${CWD}", work_dir)
    
    if not Path(mpnn_config["checkpoint_soluble_mpnn"]).exists():
        raise FileNotFoundError("LigandMPNN checkpoint file not found!")
    
    with open(yaml_path, "w") as f:
        yaml.dump(mpnn_config, f, default_flow_style=False)
    
    # Setup directories
    boltzdesign_dir = f"{main_dir}/{version_name}/results_final"
    pdb_save_dir = f"{main_dir}/{version_name}/pdb"
    
    lmpnn_redesigned_dir = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned')
    lmpnn_redesigned_fa_dir = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_fa')
    lmpnn_redesigned_yaml_dir = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_yaml')
    
    os.makedirs(ligandmpnn_dir, exist_ok=True)
    # Convert CIF to PDB and run LigandMPNN
    convert_cif_files_to_pdb(boltzdesign_dir, pdb_save_dir, high_iptm=args.high_iptm, i_ptm_cutoff=args.i_ptm_cutoff)

    if not any(f.endswith('.pdb') for f in os.listdir(pdb_save_dir)):
        print("No successful designs from BoltzDesign")
        sys.exit(1)
    
    run_ligandmpnn_redesign(
        ligandmpnn_dir, pdb_save_dir, shutil.which("boltz"),
        os.path.dirname(yaml_dir), yaml_path, top_k=args.num_designs, cutoff=args.cutoff,
        non_protein_target=args.target_type != 'protein', binder_chain=args.binder_id,
        target_chains="all", out_dir=lmpnn_redesigned_fa_dir,
        lmpnn_yaml_dir=lmpnn_redesigned_yaml_dir, results_final_dir=lmpnn_redesigned_dir
    )
    
    # Filter high confidence designs
    filter_high_confidence_designs(args, ligandmpnn_dir, lmpnn_redesigned_dir, lmpnn_redesigned_yaml_dir)
    
    print("LigandMPNN redesign step completed!")
    return ligandmpnn_dir

def filter_high_confidence_designs(args, ligandmpnn_dir, lmpnn_redesigned_dir, lmpnn_redesigned_yaml_dir):
    """Filter and save high confidence designs"""
    print("Filtering high confidence designs...")
    
    yaml_dir_success_designs_dir = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_high_iptm')
    yaml_dir_success_boltz_yaml = os.path.join(yaml_dir_success_designs_dir, 'yaml')
    yaml_dir_success_boltz_cif = os.path.join(yaml_dir_success_designs_dir, 'cif')
    
    os.makedirs(yaml_dir_success_boltz_yaml, exist_ok=True)
    os.makedirs(yaml_dir_success_boltz_cif, exist_ok=True)
    
    successful_designs = 0
    
    # Process designs
    for root in os.listdir(lmpnn_redesigned_dir):
        root_path = os.path.join(lmpnn_redesigned_dir, root, 'predictions')
        if not os.path.isdir(root_path):
            continue
        
        for subdir in os.listdir(root_path):
            json_path = os.path.join(root_path, subdir, f'confidence_{subdir}_model_0.json')
            yaml_path = os.path.join(lmpnn_redesigned_yaml_dir, f'{subdir}.yaml')
            cif_path = os.path.join(lmpnn_redesigned_dir, f'boltz_results_{subdir}', 'predictions', subdir, f'{subdir}_model_0.cif')
            
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                design_name = json_path.split('/')[-2]
                length = int(subdir[subdir.find('length') + 6:subdir.find('_model')])
                iptm = data.get('iptm', 0)
                complex_plddt = data.get('complex_plddt', 0)
                
                print(f"{design_name} length: {length} complex_plddt: {complex_plddt:.2f} iptm: {iptm:.2f}")
                
                if iptm > args.i_ptm_cutoff and complex_plddt > args.complex_plddt_cutoff:
                    shutil.copy(yaml_path, os.path.join(yaml_dir_success_boltz_yaml, f'{subdir}.yaml'))
                    shutil.copy(cif_path, os.path.join(yaml_dir_success_boltz_cif, f'{subdir}.cif'))
                    print(f"✅ {design_name} copied")
                    successful_designs += 1
            
            except (KeyError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Skipping {subdir}: {e}")
                continue
    
    if successful_designs == 0:
        print("Error: No LigandMPNN/ProteinMPNN redesigned designs passed the confidence thresholds")
        sys.exit(1)


def calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, binder_chain):
    """Calculate RMSD between holo and apo structures and update confidence CSV.
    
    Args:
        af_pdb_dir (str): Directory containing holo PDB files
        af_pdb_dir_apo (str): Directory containing apo PDB files
    """
    confidence_csv_path = af_pdb_dir + '/high_iptm_confidence_scores.csv'
    if os.path.exists(confidence_csv_path):
        df_confidence_csv = pd.read_csv(confidence_csv_path)
        for pdb_name in os.listdir(af_pdb_dir):
            if pdb_name.endswith('.pdb'):
                pdb_path = os.path.join(af_pdb_dir, pdb_name)
                pdb_path_apo = os.path.join(af_pdb_dir_apo, pdb_name)
                xyz_holo, _ = get_CA_and_sequence(pdb_path, chain_id=binder_chain)
                xyz_apo, _ = get_CA_and_sequence(pdb_path_apo, chain_id='A')
                rmsd = np_rmsd(np.array(xyz_holo), np.array(xyz_apo))
                df_confidence_csv.loc[df_confidence_csv['file'] == pdb_name.split('.pdb')[0]+'.cif', 'rmsd'] = rmsd
                print(f"{pdb_path} rmsd: {rmsd}")
        df_confidence_csv.to_csv(confidence_csv_path, index=False)
        
        
def run_alphafold_step(args, ligandmpnn_dir, work_dir, mod_to_wt_aa):
    """Run AlphaFold validation step"""
    print("Starting AlphaFold validation step...")

    alphafold_dir = os.path.expanduser(args.alphafold_dir)
    afdb_dir = os.path.expanduser(args.af3_database_settings)
    hmmer_path = os.path.expanduser(args.af3_hmmer_path)
    print("alphafold_dir", alphafold_dir)
    print("afdb_dir", afdb_dir)
    print("hmmer_path", hmmer_path)
    
    # Create AlphaFold directories
    af_input_dir = f'{ligandmpnn_dir}/02_design_json_af3'
    af_output_dir = f'{ligandmpnn_dir}/02_design_final_af3'
    af_input_apo_dir = f'{ligandmpnn_dir}/02_design_json_af3_apo'
    af_output_apo_dir = f'{ligandmpnn_dir}/02_design_final_af3_apo'
    
    for dir_path in [af_input_dir, af_output_dir, af_input_apo_dir, af_output_apo_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Process YAML files
    yaml_dir_success_boltz_yaml = os.path.join(ligandmpnn_dir, '01_lmpnn_redesigned_high_iptm', 'yaml')
    
    process_yaml_files(
        yaml_dir_success_boltz_yaml,
        af_input_dir,
        af_input_apo_dir,
        target_type=args.target_type,
        binder_chain=args.binder_id,
        mod_to_wt_aa=mod_to_wt_aa,
        afdb_dir=afdb_dir,
        hmmer_path=hmmer_path
    )
    # Run AlphaFold on holo state
    subprocess.run([
        f'{work_dir}/boltzdesign/alphafold.sh',
        af_input_dir,
        af_output_dir,
        str(args.gpu_id),
        alphafold_dir,
        args.af3_docker_name
    ], check=True)
    
    # Run AlphaFold on apo state
    subprocess.run([
        f'{work_dir}/boltzdesign/alphafold.sh',
        af_input_apo_dir,
        af_output_apo_dir,
        str(args.gpu_id),
        alphafold_dir,
        args.af3_docker_name
    ], check=True)
    
    print("AlphaFold validation step completed!")

    af_pdb_dir = f"{ligandmpnn_dir}/03_af_pdb_success"
    af_pdb_dir_apo = f"{ligandmpnn_dir}/03_af_pdb_apo"
    
    convert_cif_files_to_pdb(af_output_dir, af_pdb_dir, af_dir=True, high_iptm=args.high_iptm)
    if not any(f.endswith('.pdb') for f in os.listdir(af_pdb_dir)):
        print("No successful designs from AlphaFold")
        sys.exit(1)
    convert_cif_files_to_pdb(af_output_apo_dir, af_pdb_dir_apo, af_dir=True)
    calculate_holo_apo_rmsd(af_pdb_dir, af_pdb_dir_apo, args.binder_id)

    return af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo

def run_rosetta_step(args, ligandmpnn_dir, af_output_dir, af_output_apo_dir, af_pdb_dir, af_pdb_dir_apo):
    """Run Rosetta energy calculation (protein targets only)"""
    if args.target_type != 'protein':
        print("Skipping Rosetta step (not a protein target)")
        return
    
    print("Starting Rosetta energy calculation...")
    af_pdb_rosetta_success_dir = f"{ligandmpnn_dir}/af_pdb_rosetta_success"
    from pyrosetta_utils import measure_rosetta_energy
    measure_rosetta_energy(
        af_pdb_dir, af_pdb_dir_apo, af_pdb_rosetta_success_dir,
        binder_holo_chain=args.binder_id, binder_apo_chain='A'
    )
    
    print("Rosetta energy calculation completed!")

def setup_environment():
    """Setup environment and parse arguments"""
    args = parse_arguments()
    work_dir = args.work_dir or os.getcwd()
    os.chdir(work_dir)
    setup_gpu_environment(args.gpu_id)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return args

def get_target_ids(args):
    """Get target IDs from either PDB or custom input"""
    target_ids = args.pdb_target_ids if args.input_type == "pdb" else args.custom_target_ids
    
    if (args.contact_residues or args.modifications) and not target_ids:
        input_type = "PDB" if args.input_type == "pdb" else "Custom"
        raise ValueError(f"{input_type} target IDs must be provided when using contacts or modifications")
        sys.exit(1)

    return [str(x.strip()) for x in target_ids.split(",")] if target_ids else []

def assign_chain_ids(target_ids_list, binder_chain='A'):
    """Maps target IDs to unique chain IDs, skipping binder_chain."""
    letters = [c for c in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ' if c != binder_chain]
    return {id: letters[i] for i, id in enumerate(target_ids_list)}


def initialize_pipeline(args):
    """Initialize models and configurations"""
    work_dir = args.work_dir or os.getcwd()
    boltz_model, _ = load_boltz_model(args, torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))
    
    config_obj = YamlConfig(main_dir=f'{work_dir}/inputs/{args.target_type}_{args.target_name}_{args.suffix}')
    config_obj.setup_directories()
    return boltz_model, config_obj

def generate_yaml_config(args, config_obj):
    """Generate YAML configuration based on input type"""
    if args.contact_residues or args.modifications:
        target_ids_list = get_target_ids(args)
        target_id_map = assign_chain_ids(target_ids_list, args.binder_id)
        print(f"Mapped target IDs: {list(target_id_map.values())}")
        constraints, modifications = process_design_constraints(target_id_map, args.modifications, args.modifications_positions, args.modification_target, args.contact_residues, args.constraint_target, args.binder_id)
    else:
        constraints, modifications = None, None
    target = []
    if args.input_type == "pdb":
        pdb_target_ids = [str(x.strip()) for x in args.pdb_target_ids.split(",")] if args.pdb_target_ids else None
        target_mols = [str(x.strip()) for x in args.target_mols.split(",")] if args.target_mols else None
        if args.pdb_path:
            pdb_path = Path(args.pdb_path)
            print("load local pdb from", pdb_path)
            if not pdb_path.is_file():
                raise FileNotFoundError(f"Could not find local PDB: {args.pdb_path}")
        else:
            print("fetch pdb from RCSB")
            download_pdb(args.target_name, config_obj.PDB_DIR)
            pdb_path = config_obj.PDB_DIR / f"{args.target_name}.pdb"

        if args.target_type in ['rna', 'dna']:
            nucleotide_dict = get_nucleotide_from_pdb(pdb_path)
            for target_id in pdb_target_ids:
                target.append(nucleotide_dict[target_id]['seq'])
        elif args.target_type == 'small_molecule':
            ligand_dict = get_ligand_from_pdb(args.target_name)
            for target_mol in target_mols:
                print(target_mol, ligand_dict.keys())
                target.append(ligand_dict[target_mol])
        elif args.target_type == 'protein':
            chain_sequences = get_chains_sequence(pdb_path)
            for target_id in pdb_target_ids:
                target.append(chain_sequences[target_id])
        else:
            raise ValueError(f"Unsupported target type: {args.target_type}")
    else:
        target_inputs = [str(x.strip()) for x in args.custom_target_input.split(",")] if args.custom_target_input else []
        target = target_inputs or [args.target_name]

    return generate_yaml_for_target_binder(
        args.target_name, 
        args.target_type,
        target,
        config=config_obj,
        binder_id=args.binder_id,
        constraints=constraints,
        modifications=modifications['data'] if modifications else None,
        modification_target=modifications['target'] if modifications else None,
        use_msa=args.use_msa
    )

def setup_pipeline_config(args):
    """Setup pipeline configuration"""
    work_dir = args.work_dir or os.getcwd()
    config = load_design_config(args.target_type, work_dir)
    return update_config_with_args(config, args)

def setup_output_directories(args):
    """Setup output directories"""
    work_dir = args.work_dir or os.getcwd()
    main_dir = f'{work_dir}/outputs'
    os.makedirs(main_dir, exist_ok=True)
    return {
        'main_dir': main_dir,
        'version': f'{args.target_type}_{args.target_name}_{args.suffix}'
    }
def modification_to_wt_aa(modifications, modifications_wt):
    """Convert modifications to WT AA"""
    if not modifications:
        return None, None
    mod_to_wt_aa = {}
    for mod, wt in zip(modifications.split(','), modifications_wt.split(',')):
        mod_to_wt_aa[mod] = wt
    return mod_to_wt_aa

def run_pipeline_steps(args, config, boltz_model, yaml_dir, output_dir):
    """Run the pipeline steps based on arguments"""
    results = {'ligandmpnn_dir': f"{output_dir['main_dir']}/{output_dir['version']}/ligandmpnn_cutoff_{args.cutoff}", 'af_output_dir': None, 'af_output_apo_dir': None, 'af_pdb_dir': None, 'af_pdb_dir_apo': None}
    
    if args.run_boltz_design:
        run_boltz_design_step(args, config, boltz_model, yaml_dir, 
                            output_dir['main_dir'], output_dir['version'])

    if args.run_ligandmpnn:
        run_ligandmpnn_step(
            args, output_dir['main_dir'], output_dir['version'], 
            results['ligandmpnn_dir'], yaml_dir, args.work_dir or os.getcwd()
        )
    if args.run_alphafold:
        mod_to_wt_aa = modification_to_wt_aa(args.modifications, args.modifications_wt)
        results['af_output_dir'], results['af_output_apo_dir'], results['af_pdb_dir'], results['af_pdb_dir_apo'] = run_alphafold_step(
            args, results['ligandmpnn_dir'], args.work_dir or os.getcwd(), mod_to_wt_aa
        )
    
    if args.run_rosetta:
        run_rosetta_step(args, results['ligandmpnn_dir'], 
                        results['af_output_dir'], results['af_output_apo_dir'], results['af_pdb_dir'], results['af_pdb_dir_apo'])
    
    return results

def main():
    """Main function for running the BoltzDesign pipeline"""
    args = setup_environment()
    
    # 检查设计模式并执行相应的流程
    if args.design_mode == 'aptamer':
        run_aptamer_design_pipeline(args)
        return
    elif args.design_mode == 'predict_structure' or args.predict_structure_only:
        run_aptamer_structure_prediction_only(args)
        return
    
    # 原有的蛋白质设计流程
    boltz_model, config_obj = initialize_pipeline(args)
    yaml_dict, yaml_dir = generate_yaml_config(args, config_obj)

    print("Generated YAML configuration:")
    for key, value in yaml_dict.items():
        if isinstance(value, list):
            print(f"  {key}:")
            for item in value:
                print(f"    - {item}")
        else:
            print(f"  {key}: {value}")
    
    # Setup pipeline configuration
    config = setup_pipeline_config(args)
    output_dir = setup_output_directories(args)
    
    # Run pipeline steps
    print("config:")
    items = list(config.items())
    max_key_len = max(len(key) for key, _ in items)
    max_val_len = max(len(str(val)) for _, val in items)
    
    # Print header
    print("  " + "=" * (max_key_len + max_val_len + 5))
    
    # Print items in two columns
    for i in range(0, len(items), 2):
        key1, value1 = items[i]
        if i+1 < len(items):
            key2, value2 = items[i+1]
            print(f"  {key1:<{max_key_len}}: {str(value1):<{max_val_len}}    "
                  f"{key2:<{max_key_len}}: {value2}")
        else:
            print(f"  {key1:<{max_key_len}}: {value1}")
    
    print("  " + "=" * (max_key_len + max_val_len + 5))
    results = run_pipeline_steps(args, config, boltz_model, yaml_dir, output_dir)
    
    print("Pipeline completed successfully!")


def run_aptamer_design_pipeline(args):
    """运行适配体设计流程 - 实现角色互换的核心逻辑"""
    print("🧬" + "="*80)
    print(f"🚀 启动 {args.aptamer_type} 适配体设计流程")
    print("🧬" + "="*80)
    
    # 验证必要参数
    if not args.target_protein_seq and not args.target_ligand_smiles:
        print("❌ 错误: 适配体设计模式需要目标蛋白质序列 (--target_protein_seq) 或小分子SMILES (--target_ligand_smiles)")
        return
    
    if args.target_protein_seq and args.target_ligand_smiles:
        print("❌ 错误: 请选择一个目标类型：蛋白质序列或小分子SMILES，不能同时指定两者")
        return
    
    if args.aptamer_length < 20 or args.aptamer_length > 80:
        print("⚠️  警告: 适配体长度建议在20-80之间")
    
    # 设置环境
    setup_gpu_environment(args.gpu_id)
    print(f"🖥️  使用GPU: {args.gpu_id}")
    
    # 初始化模型
    sys.path.append(f'{os.getcwd()}/boltzdesign')
    from boltzdesign_utils import get_boltz_model
    from aptamer_design_utils import AptamerDesignConfig, create_aptamer_yaml, save_aptamer_yaml
    
    # 加载模型
    cache_dir = os.path.expanduser("~/.boltz")
    checkpoint_path = '/home/zihao/.boltz/boltz1_conf.ckpt'
    if not os.path.exists(checkpoint_path):
        print(f"❌ 错误: 找不到Boltz模型权重文件: {checkpoint_path}")
        print("请运行: python -c 'from boltz.main import download; from pathlib import Path; download(Path(\"~/.boltz\").expanduser())'")
        return
    
    device = f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
    boltz_model = get_boltz_model(checkpoint_path, device=device)
    print(f"✅ Boltz模型加载成功")
    
    # 创建适配体配置
    if args.target_protein_seq:
        target_chains = args.target_protein_chains.split(',') if args.target_protein_chains else ['B']
        target_type = 'protein'
    else:
        target_chains = args.target_ligand_chains.split(',') if args.target_ligand_chains else ['B']
        target_type = 'ligand'
    
    aptamer_config = AptamerDesignConfig(
        aptamer_type=args.aptamer_type,
        aptamer_chain=args.aptamer_chain, 
        target_chains=target_chains,
        target_type=target_type
    )
    print(f"🔧 适配体配置: {args.aptamer_type} 适配体链 {args.aptamer_chain} -> 蛋白质链 {target_chains}")
    
    # 创建适配体设计YAML (根据目标类型选择)
    if args.target_protein_seq:
        from aptamer_design_utils import create_aptamer_yaml
        yaml_content = create_aptamer_yaml(args.target_protein_seq, aptamer_config, f"{args.aptamer_type.lower()}_aptamer_{args.target_name}")
        target_info = f"蛋白质序列: {args.target_protein_seq[:30]}{'...' if len(args.target_protein_seq) > 30 else ''}"
    else:
        from aptamer_design_utils import create_ligand_aptamer_yaml
        yaml_content = create_ligand_aptamer_yaml(args.target_ligand_smiles, aptamer_config, f"{args.aptamer_type.lower()}_ligand_aptamer_{args.target_name}")
        target_info = f"小分子SMILES: {args.target_ligand_smiles[:50]}{'...' if len(args.target_ligand_smiles) > 50 else ''}"
    
    # 保存YAML文件
    work_dir = args.work_dir or os.getcwd()
    output_dir = f'{work_dir}/outputs/aptamer_{args.aptamer_type.lower()}_{args.target_name}_{args.suffix}'
    os.makedirs(output_dir, exist_ok=True)
    yaml_path = save_aptamer_yaml(yaml_content, f"{output_dir}/aptamer_design.yaml")
    print(f"📝 YAML配置已保存: {yaml_path}")
    
    # 加载CCD库
    import pickle
    ccd_path = '/home/zihao/.boltz/ccd.pkl'
    with open(ccd_path, 'rb') as f:
        ccd_lib = pickle.load(f)
    
    # 加载适配体配置
    config_path = f'boltzdesign/configs/aptamer_{args.aptamer_type.lower()}_config.yaml'
    with open(config_path, 'r') as f:
        design_config = yaml.safe_load(f)
    
    # 更新配置参数 (移除与函数参数冲突的键)
    conflicting_keys = ['length', 'aptamer_type', 'aptamer_chain', 'target_chains', 'length_min', 'length_max', 'gc_content_target', 'gc_content_weight']
    for key in conflicting_keys:
        design_config.pop(key, None)
    chain_to_number = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4}
    
    print("🚀 开始适配体设计优化...")
    print(f"🎯 目标: 设计长度为 {args.aptamer_length} 的 {args.aptamer_type} 适配体")
    print(f"🧪 目标信息: {target_info}")
    
    try:
        # 运行适配体设计 (核心的角色互换逻辑)
        from boltzdesign_utils import boltz_hallucination
        
        result = boltz_hallucination(
            boltz_model=boltz_model,
            yaml_path=yaml_path,
            ccd_lib=ccd_lib,
            length=args.aptamer_length,
            binder_chain=aptamer_config.aptamer_chain,
            aptamer_config=aptamer_config,  # 新增: 激活适配体模式
            chain_to_number=chain_to_number,
            **design_config
        )
        
        print("✅ 适配体设计完成!")
        print(f"📁 结果保存在: {output_dir}")
        
        # 提取最终序列
        from aptamer_design_utils import validate_aptamer_design
        
        # 从YAML中读取最终序列
        with open(yaml_path, 'r') as f:
            final_yaml = yaml.safe_load(f)
        
        final_sequence = None
        for seq_entry in final_yaml['sequences']:
            if args.aptamer_type.lower() in seq_entry:
                final_sequence = seq_entry[args.aptamer_type.lower()]['sequence']
                break
        
        if final_sequence:
            validation = validate_aptamer_design(final_sequence, args.aptamer_type)
            print(f"🎯 最终适配体序列: {final_sequence}")
            print(f"📊 序列长度: {validation['length']}")
            print(f"📊 GC含量: {validation['gc_content']:.2%}")
            if args.target_protein_seq:
                print(f"🧪 目标蛋白质序列: {args.target_protein_seq}")
                print(f"🔗 蛋白质长度: {len(args.target_protein_seq)} 氨基酸")
            else:
                print(f"🧪 目标小分子SMILES: {args.target_ligand_smiles}")
                print(f"🔗 小分子复杂度: {len(args.target_ligand_smiles)} 字符")
            
            # 运行结构预测和保存
            if args.save_structures:
                print("\n🏗️ 开始生成最终三维结构...")
                try:
                    structure_results = run_aptamer_structure_prediction(
                        args, yaml_path, output_dir, None, final_sequence
                    )
                    print(f"✅ 结构预测完成! 文件保存在: {structure_results['structure_dir']}")
                    
                    # 显示生成的文件
                    for file_type, file_path in structure_results['files'].items():
                        if file_path and os.path.exists(file_path):
                            file_size = os.path.getsize(file_path) / 1024  # KB
                            print(f"📁 {file_type.upper()}: {file_path} ({file_size:.1f} KB)")
                    
                    # 显示结构质量信息
                    if 'confidence' in structure_results:
                        conf = structure_results['confidence']
                        print(f"📊 结构置信度 (pLDDT): {conf['avg_plddt']:.2f}")
                        if conf['avg_plddt'] > 70:
                            print("✅ 结构质量: 高置信度")
                        elif conf['avg_plddt'] > 50:
                            print("⚠️  结构质量: 中等置信度")
                        else:
                            print("❌ 结构质量: 低置信度")
                        
                except Exception as e:
                    print(f"⚠️ 结构预测过程中出现错误: {str(e)}")
                    print("💡 序列设计已完成，但结构生成失败")
        else:
            print("⚠️  警告: 无法提取最终序列")
        
    except Exception as e:
        print(f"❌ 适配体设计过程中出现错误: {str(e)}")
        import traceback
        print("详细错误信息:")
        traceback.print_exc()

def run_aptamer_structure_prediction(args, yaml_path, output_dir, boltz_model, sequence=None):
    """
    运行适配体结构预测
    
    Args:
        args: 命令行参数
        yaml_path: YAML配置文件路径
        output_dir: 输出目录
        boltz_model: 已加载的Boltz模型
        sequence: 适配体序列(可选)
    
    Returns:
        dict: 包含结构文件路径和置信度信息的字典
    """
    import sys
    sys.path.append('/home/yifan/boltz-for-RNA-DNA/boltz/src')
    from boltz.data.write.mmcif import to_mmcif
    from boltz.data.write.pdb import to_pdb
    from boltz.data.tokenize.boltz import BoltzTokenizer
    from boltz.data.feature.featurizer import BoltzFeaturizer
    from boltz.data.parse.schema import parse_boltz_schema
    
    # 设置结构输出目录
    if args.structure_output_dir:
        structure_dir = args.structure_output_dir
    else:
        structure_dir = os.path.join(output_dir, "structures")
    
    predictions_dir = os.path.join(structure_dir, "predictions")
    os.makedirs(predictions_dir, exist_ok=True)
    
    print(f"🔬 运行适配体结构预测...")
    print(f"📝 输入YAML: {yaml_path}")
    print(f"📁 输出目录: {structure_dir}")
    
    try:
        # 使用boltz命令行工具进行结构预测
        print("🚀 开始Boltz结构预测...")
        
        # 构建boltz命令
        boltz_cmd = [
            "boltz", "predict", yaml_path,
            "--out_dir", structure_dir,
            "--recycling_steps", str(args.recycling_steps),
            "--output_format", "mmcif"  # 先生成mmcif格式
        ]
        
        # 运行boltz预测
        result = subprocess.run(boltz_cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Boltz命令执行失败:")
            print(f"STDOUT: {result.stdout}")
            print(f"STDERR: {result.stderr}")
            raise RuntimeError(f"Boltz prediction failed with return code {result.returncode}")
        
        print("✅ Boltz结构预测完成")
        
        # 查找生成的文件
        results = {
            'structure_dir': structure_dir,
            'files': {},
            'confidence': {}
        }
        
        # 查找CIF文件
        cif_files = glob.glob(os.path.join(structure_dir, "**", "*.cif"), recursive=True)
        if cif_files:
            cif_path = cif_files[0]  # 取第一个文件
            results['files']['cif'] = cif_path
            print(f"💾 找到CIF文件: {cif_path}")
            
            # 如果需要PDB格式，进行转换
            if args.output_format in ['pdb', 'both']:
                pdb_path = cif_path.replace('.cif', '.pdb')
                try:
                    from boltzdesign.utils import convert_cif_to_pdb
                    if convert_cif_to_pdb(cif_path, pdb_path):
                        results['files']['pdb'] = pdb_path
                        print(f"💾 PDB文件已转换: {pdb_path}")
                except Exception as e:
                    print(f"⚠️ PDB转换失败: {e}")
        
        # 查找置信度文件
        confidence_files = glob.glob(os.path.join(structure_dir, "**", "confidence_*.json"), recursive=True)
        if confidence_files:
            confidence_path = confidence_files[0]
            results['files']['confidence'] = confidence_path
            
            # 读取置信度信息
            try:
                with open(confidence_path, 'r') as f:
                    confidence_data = json.load(f)
                
                if 'complex_plddt' in confidence_data:
                    results['confidence'] = {
                        'avg_plddt': confidence_data['complex_plddt'],
                        'plddt_scores': confidence_data.get('plddt_scores', [])
                    }
                    print(f"📊 找到置信度文件: {confidence_path}")
                    print(f"📊 平均pLDDT: {confidence_data['complex_plddt']:.2f}")
            except Exception as e:
                print(f"⚠️ 读取置信度文件失败: {e}")
        
        # 生成文件名
        target_name = args.target_name or "aptamer"
        aptamer_name = f"aptamer_{args.aptamer_type.lower()}_{target_name}"
        
        # 保存序列和类型信息到NPZ文件
        coords_path = os.path.join(structure_dir, f"{aptamer_name}_info.npz")
        np.savez_compressed(
            coords_path,
            sequence=sequence if sequence else "",
            aptamer_type=args.aptamer_type,
            target_name=target_name,
            yaml_path=yaml_path
        )
        results['files']['info'] = coords_path
        print(f"📁 信息文件已保存: {coords_path}")
        
        return results
        
    except Exception as e:
        print(f"❌ 结构预测失败: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


def run_aptamer_structure_prediction_only(args):
    """
    独立的适配体结构预测模式
    从现有的YAML文件预测结构
    """
    print("🏗️" + "="*80)
    print("🚀 启动适配体结构预测模式")
    print("🏗️" + "="*80)
    
    # 验证输入文件
    if args.input_yaml:
        yaml_path = args.input_yaml
    else:
        # 尝试自动推断YAML文件路径
        if args.target_name and args.aptamer_type:
            work_dir = args.work_dir or os.getcwd()
            yaml_path = f'{work_dir}/outputs/aptamer_{args.aptamer_type.lower()}_{args.target_name}_{args.suffix}/aptamer_design.yaml'
        else:
            print("❌ 错误: 请提供 --input_yaml 参数或者 --target_name 和 --aptamer_type 参数")
            return
    
    if not os.path.exists(yaml_path):
        print(f"❌ 错误: 找不到YAML文件: {yaml_path}")
        return
    
    print(f"📝 输入YAML文件: {yaml_path}")
    
    # 读取YAML文件获取序列信息
    try:
        with open(yaml_path, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        # 提取适配体序列
        aptamer_sequence = None
        for seq_entry in yaml_data.get('sequences', []):
            if args.aptamer_type.lower() in seq_entry:
                aptamer_sequence = seq_entry[args.aptamer_type.lower()].get('sequence', '')
                break
        
        if aptamer_sequence and 'N' not in aptamer_sequence:
            print(f"🧬 检测到适配体序列: {aptamer_sequence}")
            print(f"📏 序列长度: {len(aptamer_sequence)}")
        else:
            print("⚠️  警告: 未找到有效的适配体序列或序列包含未确定的核苷酸(N)")
            aptamer_sequence = None
            
    except Exception as e:
        print(f"❌ 读取YAML文件失败: {e}")
        return
    
    # 设置GPU环境
    setup_gpu_environment(args.gpu_id)
    print(f"🖥️  使用GPU: {args.gpu_id}")
    
    # 检查boltz命令是否可用
    try:
        result = subprocess.run(["boltz", "--help"], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ 错误: boltz命令不可用，请确保已正确安装Boltz")
            return
        print("✅ Boltz命令行工具检查通过")
        
    except FileNotFoundError:
        print("❌ 错误: 找不到boltz命令，请确保已正确安装Boltz并添加到PATH")
        return
    except Exception as e:
        print(f"❌ Boltz工具检查失败: {e}")
        return
    
    # 确定输出目录
    output_dir = os.path.dirname(yaml_path)
    if args.structure_output_dir:
        output_dir = args.structure_output_dir
    
    # 运行结构预测
    try:
        print("\n🔬 开始结构预测...")
        structure_results = run_aptamer_structure_prediction(
            args, yaml_path, output_dir, None, aptamer_sequence
        )
        
        print(f"\n✅ 结构预测完成!")
        print(f"📁 结果保存在: {structure_results['structure_dir']}")
        
        # 显示生成的文件
        print("\n📋 生成的文件:")
        for file_type, file_path in structure_results['files'].items():
            if file_path and os.path.exists(file_path):
                file_size = os.path.getsize(file_path) / 1024  # KB
                print(f"  📄 {file_type.upper()}: {os.path.basename(file_path)} ({file_size:.1f} KB)")
        
        # 显示结构质量信息
        if 'confidence' in structure_results:
            conf = structure_results['confidence']
            print(f"\n📊 结构质量评估:")
            print(f"  🎯 平均pLDDT: {conf['avg_plddt']:.2f}")
            if conf['avg_plddt'] > 70:
                print("  ✅ 结构质量: 高置信度 (>70)")
            elif conf['avg_plddt'] > 50:
                print("  ⚠️  结构质量: 中等置信度 (50-70)")
            else:
                print("  ❌ 结构质量: 低置信度 (<50)")
        
        print(f"\n🎉 结构预测流程完成!")
        
    except Exception as e:
        print(f"❌ 结构预测过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()


def show_aptamer_design_help():
    """显示适配体设计的使用帮助"""
    print("""
🧬 适配体设计模式使用指南:

1. RNA适配体设计 (蛋白质目标):
   python boltzdesign1.py --design_mode aptamer --aptamer_type RNA \\
       --target_protein_seq "MKLLVVVGGVGSGKTTLLRQLAKEFG" \\
       --aptamer_length 40 --target_name myprotein --gpu_id 0

2. DNA适配体设计 (蛋白质目标):
   python boltzdesign1.py --design_mode aptamer --aptamer_type DNA \\
       --target_protein_seq "MKLLVVVGGVGSGKTTLLRQLAKEFG" \\
       --aptamer_length 50 --target_name myprotein --gpu_id 0

3. RNA适配体设计 (小分子目标):
   python boltzdesign1.py --design_mode aptamer --aptamer_type RNA \\
       --target_ligand_smiles "N[C@@H](Cc1ccc(O)cc1)C(=O)O" \\
       --aptamer_length 30 --target_name tyrosine --gpu_id 0

4. DNA适配体设计 (小分子目标):
   python boltzdesign1.py --design_mode aptamer --aptamer_type DNA \\
       --target_ligand_smiles "CC(C)N(C(=O)CCC)C(C)c1ccccc1" \\
       --aptamer_length 35 --target_name drug --gpu_id 0

5. 独立结构预测模式:
   python boltzdesign1.py --design_mode predict_structure \\
       --input_yaml "outputs/aptamer_dna_tyrosine_balanced_0/aptamer_design.yaml" \\
       --output_format both --gpu_id 0

参数说明:
- --design_mode: protein(原始), aptamer(适配体设计), predict_structure(仅结构预测)
- --aptamer_type: RNA 或 DNA
- --target_protein_seq: 目标蛋白质序列 (与SMILES二选一)
- --target_ligand_smiles: 目标小分子SMILES字符串 (与蛋白质序列二选一)
- --aptamer_length: 适配体长度 (推荐20-80)
- --save_structures: 是否保存结构文件 (默认True)
- --output_format: 输出格式 cif/pdb/both (默认both)
- --input_yaml: 结构预测模式的输入YAML文件
""")

if __name__ == "__main__":
    main()

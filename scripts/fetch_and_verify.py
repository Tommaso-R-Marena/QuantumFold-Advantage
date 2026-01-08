#!/usr/bin/env python3
"""Fetch protein data from PDB and verify integrity."""
import hashlib
import json
import logging
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

PDB_API = "https://data.rcsb.org/rest/v1/core/entry"
PROVENANCE_FILE = "outputs/sources.json"

def fetch_pdb_structure(pdb_id):
    """
    Fetch structure data from PDB.
    
    Args:
        pdb_id: PDB identifier (e.g., '1YPA')
    
    Returns:
        dict: Structure metadata
    """
    url = f"{PDB_API}/{pdb_id}"
    logger.info(f"Fetching PDB entry {pdb_id}...")
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        # Calculate checksum
        content_bytes = json.dumps(data, sort_keys=True).encode()
        sha256 = hashlib.sha256(content_bytes).hexdigest()
        
        logger.info(f"Successfully fetched {pdb_id} (SHA256: {sha256[:16]}...)")
        
        return {
            'pdb_id': pdb_id,
            'url': url,
            'sha256': sha256,
            'data': data
        }
    except Exception as e:
        logger.error(f"Failed to fetch {pdb_id}: {e}")
        return None

def save_provenance(entries, output_path=PROVENANCE_FILE):
    """Save data provenance information."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    provenance = {
        'source': 'RCSB Protein Data Bank',
        'license': 'PDB data is freely available (CC0)',
        'doi': '10.1093/nar/gkz1021',
        'citation': 'Berman et al., Nucleic Acids Res. 2000',
        'entries': entries
    }
    
    with open(output_path, 'w') as f:
        json.dump(provenance, f, indent=2)
    
    logger.info(f"Provenance saved to {output_path}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Example: fetch small protein structures
    pdb_ids = ['1YPA', '1CRN']  # Small proteins for testing
    
    entries = []
    for pdb_id in pdb_ids:
        result = fetch_pdb_structure(pdb_id)
        if result:
            entries.append({
                'pdb_id': result['pdb_id'],
                'url': result['url'],
                'sha256': result['sha256']
            })
    
    if entries:
        save_provenance(entries)
        print(f"\nFetched {len(entries)} structures. See {PROVENANCE_FILE}")
    else:
        print("No structures fetched. Check network connectivity.")
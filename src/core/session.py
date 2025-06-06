"""
Evaluation Session Management

Handles session creation, metadata tracking, and result persistence.
"""

import json
import hashlib
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import uuid


class EvaluationSession:
    """Manages an evaluation session with full metadata tracking."""
    
    def __init__(self, session_id: Optional[str] = None):
        """Initialize a new evaluation session."""
        self.session_id = session_id or self._generate_session_id()
        self.start_time = datetime.now()
        self.metadata = {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': None,
            'status': 'running',
            'evaluators': {},
            'datasets': {},
            'config': {},
            'results': {},
            'git_info': self._get_git_info(),
            'environment': self._get_environment_info()
        }
        self.results_dir = Path(f"src/results/{self.session_id}")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
    def _generate_session_id(self) -> str:
        """Generate a unique session ID with timestamp."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}"
    
    def _get_git_info(self) -> Dict[str, str]:
        """Get current git commit hash and status."""
        try:
            # Get project root (go up from src/core/)
            project_root = Path(__file__).parent.parent.parent
            
            # Get current commit hash
            commit_hash = subprocess.check_output(
                ['git', 'rev-parse', 'HEAD'], 
                cwd=project_root,
                text=True
            ).strip()
            
            # Get commit message
            commit_message = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%B'],
                cwd=project_root,
                text=True
            ).strip()
            
            # Get commit author and date
            commit_author = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%an <%ae>'],
                cwd=project_root,
                text=True
            ).strip()
            
            commit_date = subprocess.check_output(
                ['git', 'log', '-1', '--pretty=%ai'],
                cwd=project_root,
                text=True
            ).strip()
            
            # Get current branch
            try:
                branch = subprocess.check_output(
                    ['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                    cwd=project_root,
                    text=True
                ).strip()
            except:
                branch = 'unknown'
            
            # Check if working directory is clean
            status = subprocess.check_output(
                ['git', 'status', '--porcelain'],
                cwd=project_root,
                text=True
            ).strip()
            
            return {
                'commit_hash': commit_hash,
                'commit_message': commit_message,
                'commit_author': commit_author,
                'commit_date': commit_date,
                'branch': branch,
                'is_clean': len(status) == 0,
                'status': status if status else 'clean',
                'uncommitted_files': status.split('\n') if status else []
            }
        except Exception as e:
            return {
                'commit_hash': 'unknown',
                'is_clean': False,
                'error': str(e)
            }
    
    def _get_environment_info(self) -> Dict[str, Any]:
        """Get environment information."""
        import platform
        import sys
        
        return {
            'python_version': sys.version,
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor()
        }
    
    def register_evaluator(self, evaluator_name: str, evaluator_path: str, config: Dict[str, Any]):
        """Register an evaluator used in this session."""
        # Calculate file hash
        file_hash = self._calculate_file_hash(evaluator_path)
        
        self.metadata['evaluators'][evaluator_name] = {
            'path': evaluator_path,
            'file_hash': file_hash,
            'config': config,
            'registered_at': datetime.now().isoformat()
        }
        self._save_metadata()
    
    def register_dataset(self, dataset_name: str, dataset_path: str, metadata: Optional[Dict[str, Any]] = None):
        """Register a dataset used in this session."""
        file_hash = self._calculate_file_hash(dataset_path)
        
        # Get basic dataset info
        dataset_info = {
            'path': dataset_path,
            'file_hash': file_hash,
            'registered_at': datetime.now().isoformat()
        }
        
        # Add metadata if provided
        if metadata:
            dataset_info.update(metadata)
        
        # Try to get row count if it's a CSV
        if dataset_path.endswith('.csv'):
            try:
                import csv
                with open(dataset_path, 'r') as f:
                    row_count = sum(1 for _ in csv.reader(f)) - 1  # Subtract header
                dataset_info['row_count'] = row_count
            except:
                pass
        
        self.metadata['datasets'][dataset_name] = dataset_info
        self._save_metadata()
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        try:
            with open(file_path, "rb") as f:
                for byte_block in iter(lambda: f.read(4096), b""):
                    sha256_hash.update(byte_block)
            return sha256_hash.hexdigest()
        except Exception as e:
            return f"error: {str(e)}"
    
    def log_result(self, evaluator_name: str, dataset_name: str, results: Dict[str, Any]):
        """Log evaluation results."""
        result_key = f"{evaluator_name}_{dataset_name}"
        
        self.metadata['results'][result_key] = {
            'evaluator': evaluator_name,
            'dataset': dataset_name,
            'timestamp': datetime.now().isoformat(),
            'summary': results.get('summary', {}),
            'details_file': f"{result_key}_details.json"
        }
        
        # Save detailed results to separate file
        details_path = self.results_dir / f"{result_key}_details.json"
        with open(details_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        self._save_metadata()
    
    def update_config(self, config: Dict[str, Any]):
        """Update session configuration."""
        self.metadata['config'].update(config)
        self._save_metadata()
    
    def finish(self):
        """Mark session as finished."""
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['status'] = 'completed'
        self.metadata['duration_seconds'] = (datetime.now() - self.start_time).total_seconds()
        self._save_metadata()
    
    def _save_metadata(self):
        """Save session metadata to file."""
        metadata_path = self.results_dir / "session_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get session summary."""
        return {
            'session_id': self.session_id,
            'status': self.metadata['status'],
            'duration': self.metadata.get('duration_seconds', 'ongoing'),
            'evaluators': list(self.metadata['evaluators'].keys()),
            'datasets': list(self.metadata['datasets'].keys()),
            'results': len(self.metadata['results'])
        }
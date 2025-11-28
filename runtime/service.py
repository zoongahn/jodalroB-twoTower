"""
STEP1 Runtime Service

Main entry point for the Two-Tower prediction runtime.
This module provides the interface for the platform to use the Two-Tower model.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import torch
import logging

# 프로젝트 루트를 Python path에 추가 (임시)
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Database import (공용 패키지)
from database.database_connector import DatabaseConnector

# Runtime 내부 imports
from .model.loader import ModelLoader
from .model.predictor import TwoTowerPredictor
from .preprocess.torchrec_adapter import TorchRecAdapter

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances (싱글톤 패턴)
_db_connector: Optional[DatabaseConnector] = None
_model_loader: Optional[ModelLoader] = None
_predictor: Optional[TwoTowerPredictor] = None
_adapter: Optional[TorchRecAdapter] = None
_config: Dict[str, Any] = {}


def initialize(
    checkpoint_path: str,
    vector_db_path: Optional[str] = None,
    metadata_path: Optional[str] = None,
    device: str = "cuda",
    db_config: Optional[Dict[str, Any]] = None
) -> bool:
    """
    Initialize the STEP1 runtime

    Args:
        checkpoint_path: Path to the model checkpoint file
        vector_db_path: Optional path to vector database (for faster inference)
        metadata_path: Optional path to metadata.csv (if not in default location)
        device: Device to use ("cuda" or "cpu")
        db_config: Optional database configuration dict

    Returns:
        True if initialization successful
    """
    global _db_connector, _model_loader, _predictor, _adapter, _config

    try:
        logger.info("Initializing STEP1 runtime...")

        # Store configuration
        _config = {
            'checkpoint_path': checkpoint_path,
            'vector_db_path': vector_db_path,
            'metadata_path': metadata_path,
            'device': device
        }

        # Initialize database connector
        if db_config:
            _db_connector = DatabaseConnector(db_config)
        else:
            # Use default configuration from environment
            _db_connector = DatabaseConnector()

        logger.info("✓ Database connector initialized")

        # Initialize model loader
        _model_loader = ModelLoader(
            checkpoint_path=checkpoint_path,
            device=device
        )

        # Update metadata_path in config if provided
        if metadata_path:
            _model_loader.config['metadata_path'] = metadata_path

        # Load model
        _model_loader.load_model()
        notice_tower = _model_loader.model.notice_tower
        company_tower = _model_loader.model.company_tower
        schema = _model_loader.schema
        preprocessor = _model_loader.preprocessor

        logger.info("✓ Model loaded from checkpoint")

        # Initialize TorchRec adapter
        _adapter = TorchRecAdapter(
            schema=schema,
            metadata_path=metadata_path,
            device=device
        )

        logger.info("✓ TorchRec adapter initialized")

        # Initialize predictor
        _predictor = TwoTowerPredictor(
            notice_tower=notice_tower,
            company_tower=company_tower,
            db_engine=_db_connector.engine,
            schema=schema,
            preprocessor=preprocessor,
            use_vector_db=vector_db_path is not None,
            vector_db_path=vector_db_path,
            device=device
        )

        logger.info("✓ Predictor initialized")

        logger.info(f"STEP1 runtime initialization complete")
        logger.info(f"  - Checkpoint: {checkpoint_path}")
        logger.info(f"  - Device: {device}")
        logger.info(f"  - Vector DB: {'Enabled' if vector_db_path else 'Disabled'}")

        return True

    except Exception as e:
        logger.error(f"Failed to initialize STEP1 runtime: {e}")
        raise


def run_step1_prediction(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main entry point for STEP1 prediction

    Args:
        params: Dictionary containing:
            - bidntceno: 공고번호 (str)
            - bidntceord: 공고차수 (str)
            - top_k: Number of top companies to return (int, default=10)
            - min_similarity: Minimum similarity threshold (float, optional)
            - return_embeddings: Whether to return embeddings (bool, default=False)

    Returns:
        Dictionary containing:
            - notice: Notice information with optional embedding
            - top_k_companies: List of top companies with scores
            - metadata: Additional metadata (processing time, etc.)
    """
    global _predictor

    if _predictor is None:
        raise RuntimeError("Runtime not initialized. Call initialize() first.")

    # Extract parameters
    bidntceno = params['bidntceno']
    bidntceord = params.get('bidntceord', '00')
    top_k = params.get('top_k', 10)
    min_similarity = params.get('min_similarity', None)
    return_embeddings = params.get('return_embeddings', False)

    try:
        logger.info(f"Predicting for notice: {bidntceno}-{bidntceord}")

        # Run prediction
        result = _predictor.predict_for_notice(
            bidntceno=bidntceno,
            bidntceord=bidntceord,
            top_k=top_k,
            min_similarity=min_similarity,
            return_embeddings=return_embeddings
        )

        # Format output
        output = {
            'notice': {
                'bidntceno': bidntceno,
                'bidntceord': bidntceord,
                'title': result.get('notice_title', 'Unknown')
            },
            'top_k_companies': [],
            'metadata': {
                'model_version': '1.0.0',
                'device': _config.get('device', 'unknown')
            }
        }

        # Add notice embedding if requested
        if return_embeddings and 'notice_embedding' in result:
            output['notice']['embedding'] = result['notice_embedding'].tolist()

        # Process top-k companies
        for company_id, similarity in result.get('top_k_companies', []):
            company_info = {
                'bizno': company_id,
                'similarity': float(similarity),
                'name': result.get('company_names', {}).get(company_id, 'Unknown')
            }

            # Add company embedding if requested
            if return_embeddings and 'company_embeddings' in result:
                if company_id in result['company_embeddings']:
                    company_info['embedding'] = result['company_embeddings'][company_id].tolist()

            output['top_k_companies'].append(company_info)

        logger.info(f"✓ Prediction completed: {len(output['top_k_companies'])} companies returned")

        return output

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise


def predict_batch(
    notices: List[Dict[str, Any]],
    top_k: int = 10,
    return_embeddings: bool = False
) -> List[Dict[str, Any]]:
    """
    Batch prediction for multiple notices

    Args:
        notices: List of notice dictionaries with bidntceno and bidntceord
        top_k: Number of top companies per notice
        return_embeddings: Whether to return embeddings

    Returns:
        List of prediction results
    """
    results = []

    for notice in notices:
        params = {
            'bidntceno': notice['bidntceno'],
            'bidntceord': notice.get('bidntceord', '00'),
            'top_k': top_k,
            'return_embeddings': return_embeddings
        }
        result = run_step1_prediction(params)
        results.append(result)

    return results


def get_notice_embedding(bidntceno: str, bidntceord: str = '00') -> Optional[torch.Tensor]:
    """
    Get embedding vector for a specific notice

    Args:
        bidntceno: 공고번호
        bidntceord: 공고차수

    Returns:
        Notice embedding tensor or None if not found
    """
    global _predictor

    if _predictor is None:
        raise RuntimeError("Runtime not initialized. Call initialize() first.")

    return _predictor.get_notice_embedding(bidntceno, bidntceord)


def get_company_embedding(bizno: str) -> Optional[torch.Tensor]:
    """
    Get embedding vector for a specific company

    Args:
        bizno: 사업자번호

    Returns:
        Company embedding tensor or None if not found
    """
    global _predictor

    if _predictor is None:
        raise RuntimeError("Runtime not initialized. Call initialize() first.")

    return _predictor.get_company_embedding(bizno)


def shutdown():
    """
    Clean shutdown of the runtime
    """
    global _db_connector, _predictor

    if _db_connector:
        _db_connector.close()
        logger.info("Database connection closed")

    if _predictor:
        # Clean up any resources
        del _predictor
        _predictor = None

    logger.info("STEP1 runtime shutdown complete")


# Convenience function for testing
if __name__ == "__main__":
    # Test initialization
    import os
    from dotenv import load_dotenv

    load_dotenv()

    # Initialize with test parameters
    initialize(
        checkpoint_path="/data/dev/jodalroB-twoTower/output/models/20251118_071938/checkpoint_epoch_1.pt",
        vector_db_path=None,  # Will create from DB
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Test prediction
    test_params = {
        'bidntceno': '20210408676',
        'bidntceord': '00',
        'top_k': 5,
        'return_embeddings': False
    }

    result = run_step1_prediction(test_params)
    print(f"Notice: {result['notice']['title']}")
    print(f"Top companies: {len(result['top_k_companies'])}")
    for company in result['top_k_companies']:
        print(f"  - {company['bizno']}: {company['name']} (score: {company['similarity']:.4f})")

    # Shutdown
    shutdown()
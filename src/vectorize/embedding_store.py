"""
Faiss-based Vector DB for storing and retrieving notice/company embeddings.

Supports composite keys (tuples) for IDs:
- Notice IDs: (bidntceno, bidntceord)
- Company IDs: can be single value or tuple
"""
import os
import pickle
from typing import List, Dict, Optional, Tuple, Union
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "Faiss is not installed. Please install it with: "
        "pip install faiss-cpu (or faiss-gpu for GPU support)"
    )


class EmbeddingStore:
    """
    Vector database for storing notice and company embeddings.

    Uses Faiss for efficient storage and retrieval of embeddings.
    Supports saving/loading from disk for persistence.
    """

    def __init__(self, dimension: int = 128):
        """
        Initialize the embedding store.

        Args:
            dimension: Dimension of the embedding vectors (default: 128)
        """
        self.dimension = dimension

        # Initialize Faiss indexes (using L2 distance)
        self.notice_index = faiss.IndexFlatL2(dimension)
        self.company_index = faiss.IndexFlatL2(dimension)

        # Store IDs for mapping index positions to original IDs
        self.notice_ids = []
        self.company_ids = []

        # ID to index mapping for quick lookup
        self.notice_id_to_idx = {}
        self.company_id_to_idx = {}

    def add_notices(
        self,
        notice_ids: List[Union[Tuple, str]],
        embeddings: np.ndarray
    ):
        """
        Add notice embeddings to the store.

        Args:
            notice_ids: List of notice IDs (can be tuples like (bidntceno, bidntceord) or strings)
            embeddings: Numpy array of shape (N, dimension)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"expected dimension {self.dimension}"
            )

        start_idx = len(self.notice_ids)

        # Add to Faiss index
        self.notice_index.add(embeddings.astype('float32'))

        # Update ID mappings
        for i, notice_id in enumerate(notice_ids):
            self.notice_ids.append(notice_id)
            self.notice_id_to_idx[notice_id] = start_idx + i

        print(f"Added {len(notice_ids)} notice embeddings. "
              f"Total: {len(self.notice_ids)}")

    def add_companies(
        self,
        company_ids: List[Union[Tuple, str]],
        embeddings: np.ndarray
    ):
        """
        Add company embeddings to the store.

        Args:
            company_ids: List of company IDs (can be tuples or strings)
            embeddings: Numpy array of shape (N, dimension)
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(
                f"Embedding dimension {embeddings.shape[1]} does not match "
                f"expected dimension {self.dimension}"
            )

        start_idx = len(self.company_ids)

        # Add to Faiss index
        self.company_index.add(embeddings.astype('float32'))

        # Update ID mappings
        for i, company_id in enumerate(company_ids):
            self.company_ids.append(company_id)
            self.company_id_to_idx[company_id] = start_idx + i

        print(f"Added {len(company_ids)} company embeddings. "
              f"Total: {len(self.company_ids)}")

    def get_notice_embedding(
        self,
        notice_id: Union[Tuple, str]
    ) -> Optional[np.ndarray]:
        """
        Get notice embedding by ID.

        Args:
            notice_id: Notice ID (tuple like (bidntceno, bidntceord) or string)

        Returns:
            Embedding vector or None if not found
        """
        idx = self.notice_id_to_idx.get(notice_id)
        if idx is None:
            return None
        return self.notice_index.reconstruct(idx)

    def get_company_embedding(
        self,
        company_id: Union[Tuple, str]
    ) -> Optional[np.ndarray]:
        """
        Get company embedding by ID.

        Args:
            company_id: Company ID (tuple or string)

        Returns:
            Embedding vector or None if not found
        """
        idx = self.company_id_to_idx.get(company_id)
        if idx is None:
            return None
        return self.company_index.reconstruct(idx)

    def get_notice_embeddings_batch(
        self,
        notice_ids: List[Union[Tuple, str]]
    ) -> Tuple[List[Union[Tuple, str]], np.ndarray]:
        """
        Get multiple notice embeddings at once.

        Args:
            notice_ids: List of notice IDs (tuples or strings)

        Returns:
            Tuple of (valid_ids, embeddings array)
        """
        valid_ids = []
        embeddings = []

        for notice_id in notice_ids:
            emb = self.get_notice_embedding(notice_id)
            if emb is not None:
                valid_ids.append(notice_id)
                embeddings.append(emb)

        if embeddings:
            return valid_ids, np.array(embeddings)
        return [], np.array([])

    def get_company_embeddings_batch(
        self,
        company_ids: List[Union[Tuple, str]]
    ) -> Tuple[List[Union[Tuple, str]], np.ndarray]:
        """
        Get multiple company embeddings at once.

        Args:
            company_ids: List of company IDs (tuples or strings)

        Returns:
            Tuple of (valid_ids, embeddings array)
        """
        valid_ids = []
        embeddings = []

        for company_id in company_ids:
            emb = self.get_company_embedding(company_id)
            if emb is not None:
                valid_ids.append(company_id)
                embeddings.append(emb)

        if embeddings:
            return valid_ids, np.array(embeddings)
        return [], np.array([])

    def save(self, path_prefix: str):
        """
        Save the embedding store to disk.

        Args:
            path_prefix: Path prefix for saving files
                        Will create:
                        - {path_prefix}_notice.index
                        - {path_prefix}_company.index
                        - {path_prefix}_metadata.pkl
        """
        os.makedirs(os.path.dirname(path_prefix) or '.', exist_ok=True)

        # Save Faiss indexes
        faiss.write_index(self.notice_index, f"{path_prefix}_notice.index")
        faiss.write_index(self.company_index, f"{path_prefix}_company.index")

        # Save metadata (IDs and mappings)
        metadata = {
            'dimension': self.dimension,
            'notice_ids': self.notice_ids,
            'company_ids': self.company_ids,
            'notice_id_to_idx': self.notice_id_to_idx,
            'company_id_to_idx': self.company_id_to_idx,
        }

        with open(f"{path_prefix}_metadata.pkl", 'wb') as f:
            pickle.dump(metadata, f)

        print(f"Saved embedding store to {path_prefix}_*")
        print(f"  - Notices: {len(self.notice_ids)}")
        print(f"  - Companies: {len(self.company_ids)}")

    def load(self, path_prefix: str):
        """
        Load the embedding store from disk.

        Args:
            path_prefix: Path prefix used when saving
        """
        # Load Faiss indexes
        self.notice_index = faiss.read_index(f"{path_prefix}_notice.index")
        self.company_index = faiss.read_index(f"{path_prefix}_company.index")

        # Load metadata
        with open(f"{path_prefix}_metadata.pkl", 'rb') as f:
            metadata = pickle.load(f)

        self.dimension = metadata['dimension']
        self.notice_ids = metadata['notice_ids']
        self.company_ids = metadata['company_ids']
        self.notice_id_to_idx = metadata['notice_id_to_idx']
        self.company_id_to_idx = metadata['company_id_to_idx']

        print(f"Loaded embedding store from {path_prefix}_*")
        print(f"  - Notices: {len(self.notice_ids)}")
        print(f"  - Companies: {len(self.company_ids)}")

    def get_stats(self) -> Dict:
        """
        Get statistics about the embedding store.

        Returns:
            Dictionary with statistics
        """
        return {
            'dimension': self.dimension,
            'num_notices': len(self.notice_ids),
            'num_companies': len(self.company_ids),
            'notice_index_size': self.notice_index.ntotal,
            'company_index_size': self.company_index.ntotal,
        }

    def __repr__(self) -> str:
        return (
            f"EmbeddingStore(dimension={self.dimension}, "
            f"notices={len(self.notice_ids)}, "
            f"companies={len(self.company_ids)})"
        )

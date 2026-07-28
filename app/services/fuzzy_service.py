from typing import Protocol, List, Dict, Any, Optional
from thefuzz import fuzz
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

SIMILARITY_THRESHOLD = 90

class FuzzyMatchingService(Protocol):
    """Protocol for name matching services."""
    def match_names(
        self,
        extracted_names: List[str],
        query_names: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        ...

class FuzzyService:
    """
    Service for fuzzy string matching between extracted names and query names.
    Uses thefuzz library for similarity calculations.
    """
    
    def __init__(self, threshold: Optional[int] = None):
        """
        Initialize FuzzyService.

        Args:
            threshold: Minimum score (0-100) to consider a match successful.
        """
        if threshold is None:
            threshold = int(os.getenv("FUZZY_THRESHOLD", str(SIMILARITY_THRESHOLD)))
            
        if not (0 <= threshold <= 100):
            raise ValueError("Threshold must be between 0 and 100")
        self.threshold = threshold
        logger.debug(f"FuzzyService initialized with threshold={self.threshold}")

    def match_names(
        self,
        extracted_names: List[str],
        query_names: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Perform fuzzy matching between extracted and query names.

        Args:
            extracted_names: List of names extracted from documents.
            query_names: List of dictionaries containing 'first_name' and 'last_name'.

        Returns:
            List of dictionaries containing matched names and their scores.
        """
        if not extracted_names or not query_names:
            logger.debug("Empty input provided to FuzzyService.match_names")
            return []

        logger.info(
            f"Matching {len(extracted_names)} extracted names "
            f"against {len(query_names)} query names (threshold: {self.threshold})"
        )
        
        matches = []
        for query in query_names:
            try:
                first_name = query.get('first_name')
                last_name = query.get('last_name')
                
                if not first_name or not last_name:
                    logger.warning(f"Invalid query name entry: {query}. Skipping.")
                    continue

                query_full = f"{first_name} {last_name}".strip()
                best_match: Optional[str] = None
                best_score = 0

                for extracted in extracted_names:
                    if not isinstance(extracted, str):
                        continue
                    
                    # Use a combination of token_sort_ratio and partial_ratio
                    # partial_ratio is good for OCR errors where name might be part of longer string
                    # token_sort_ratio is better for matching names with different order or minor typos
                    score = fuzz.token_sort_ratio(query_full, extracted)
                    if score > best_score:
                        best_score = score
                        best_match = extracted

                if best_score >= self.threshold:
                    matches.append({
                        "extracted_name": best_match,
                        "matched_name": query_full,
                        "score": round(best_score / 100.0, 2),
                    })

            except Exception as e:
                logger.error(f"Unexpected error matching query {query}: {e}")
                continue

        logger.info(f"FuzzyService found {len(matches)} matches.")
        return matches

    def fuzzy_match_names(
        self,
        extracted_names: List[str],
        query_names: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Legacy alias for fuzzy matching (used in some tests)."""
        return self.match_names(extracted_names, query_names)


def match_names(
    extracted_names: List[str],
    query_names: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Convenience wrapper for FuzzyService.match_names."""
    return FuzzyService().match_names(extracted_names, query_names)

def fuzzy_match_names(
    extracted_names: List[str],
    query_names: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Legacy alias for fuzzy matching (used in some tests)."""
    return match_names(extracted_names, query_names)

import threading
import spacy
import os
from loguru import logger
from typing import Optional, List, Dict
from dotenv import load_dotenv

load_dotenv()


class SpacyModel:
    _instance: Optional[spacy.language.Language] = None
    _lock = threading.Lock()

    @classmethod
    def get_instance(cls) -> spacy.language.Language:
        """Get thread-safe singleton instance of spacy model."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    try:
                        model_name = os.getenv("SPACY_MODEL", "en_core_web_sm")
                        cls._instance = spacy.load(model_name)
                        logger.info(f"Successfully loaded spaCy model '{model_name}'")
                    except Exception as e:
                        logger.error(f"Failed to load spaCy model: {e}")
                        raise
        return cls._instance


class NerService:
    def __init__(self, text: str, nlp_model: Optional[spacy.language.Language] = None):
        """
        Initialize NerService.
        
        Args:
            text: The text to process.
            nlp_model: Optional spacy model instance (for dependency injection).
        """
        if not isinstance(text, str):
            raise ValueError(f"Input text must be a string, got {type(text)}")
            
        self.text = text
        self._nlp = nlp_model or SpacyModel.get_instance()

    def extract_names(self, include_orgs: bool = True) -> List[str]:
        """
        Extract person names (and optionally organizations) from text using spaCy NER.
        
        Args:
            include_orgs: Whether to include ORG entities. Defaults to True for backward compatibility.
            
        Returns:
            List of extracted names.
        """
        logger.info(f"Extracting names from text (length: {len(self.text)})")

        try:
            doc = self._nlp(self.text)
            
            allowed_labels = {"PERSON", "ORG"} if include_orgs else {"PERSON"}
            names = [ent.text for ent in doc.ents if ent.label_ in allowed_labels]
            
            logger.info(f"Extracted {len(names)} names")
            return names

        except Exception as e:
            logger.exception(f"Error in NerService.extract_names: {e}")
            return []

    def extract_names_with_positions(self, include_orgs: bool = True) -> List[Dict]:
        """
        Extract person names (and optionally organizations) with their character positions.
        
        Args:
            include_orgs: Whether to include ORG entities. Defaults to True for backward compatibility.
            
        Returns:
            List of dicts containing name, start_char, end_char, and label.
        """
        logger.info(f"Extracting names with positions from text (length: {len(self.text)})")

        try:
            doc = self._nlp(self.text)
            
            allowed_labels = {"PERSON", "ORG"} if include_orgs else {"PERSON"}
            results = [
                {
                    "name": ent.text,
                    "start_char": ent.start_char,
                    "end_char": ent.end_char,
                    "label": ent.label_,
                }
                for ent in doc.ents if ent.label_ in allowed_labels
            ]

            logger.info(f"Extracted {len(results)} names with positions")
            return results

        except Exception as e:
            logger.exception(f"Error in NerService.extract_names_with_positions: {e}")
            return []

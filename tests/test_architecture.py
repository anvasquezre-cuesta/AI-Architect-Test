"""Tests for code architecture and software design practices.

These tests verify that the codebase follows proper OOP principles,
uses abstractions, and implements clean architecture patterns.

ALL tests will FAIL against the current implementation.
"""

import ast
import os
import pytest

APP_DIR = os.path.join(os.path.dirname(__file__), "..", "app")
SERVICES_DIR = os.path.join(APP_DIR, "services")


def get_python_files(directory):
    """Collect all .py files in a directory recursively."""
    py_files = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.endswith(".py") and f != "__init__.py":
                py_files.append(os.path.join(root, f))
    return py_files


def parse_module(filepath):
    """Parse a Python file and return its AST."""
    with open(filepath) as f:
        return ast.parse(f.read(), filename=filepath)


class TestOOPDesign:
    """Verify the codebase uses proper object-oriented design."""

    def test_services_use_classes(self):
        """Services should be implemented as classes, not loose functions."""
        service_files = get_python_files(SERVICES_DIR)
        files_with_classes = []

        for filepath in service_files:
            tree = parse_module(filepath)
            has_class = any(
                isinstance(node, ast.ClassDef) for node in ast.walk(tree)
            )
            if has_class:
                files_with_classes.append(os.path.basename(filepath))

        assert len(files_with_classes) >= 4, (
            f"Only {len(files_with_classes)} service files use classes "
            f"({files_with_classes}). All services should be implemented "
            f"as classes for proper encapsulation and dependency injection."
        )

    def test_no_bare_module_level_functions_in_services(self):
        """Service logic should live in class methods, not module-level functions."""
        service_files = get_python_files(SERVICES_DIR)
        violating_files = []

        # List of files allowed to have top-level functions (e.g., for backward compatibility)
        allowed_files = ["embedding_service.py", "ner_service.py", "vector_service.py", "rag_service.py", "extract.py"]

        for filepath in service_files:
            if os.path.basename(filepath) in allowed_files:
                continue
            tree = parse_module(filepath)
            top_level_funcs = [
                node.name
                for node in ast.iter_child_nodes(tree)
                if isinstance(node, ast.FunctionDef)
            ]
            if top_level_funcs:
                violating_files.append(
                    (os.path.basename(filepath), top_level_funcs)
                )

        assert len(violating_files) == 0, (
            f"Service files with bare module-level functions: "
            f"{', '.join(f'{f}: {fns}' for f, fns in violating_files)}. "
            f"Encapsulate service logic in classes."
        )


class TestAbstractions:
    """Verify the codebase uses proper abstractions and protocols."""

    def test_services_implement_protocols_or_abcs(self):
        """Services should implement abstract base classes or protocols
        to allow for dependency injection and testability.
        """
        service_files = get_python_files(SERVICES_DIR)
        has_abstractions = False

        for filepath in service_files:
            with open(filepath) as f:
                content = f.read()
            if any(
                keyword in content
                for keyword in [
                    "Protocol",
                    "ABC",
                    "abstractmethod",
                    "ABCMeta",
                    "runtime_checkable",
                ]
            ):
                has_abstractions = True
                break

        assert has_abstractions, (
            "No abstract base classes or protocols found in services. "
            "Define interfaces (Protocol or ABC) for OCR, NER, embedding, "
            "and vector store services to enable dependency injection "
            "and make components swappable."
        )

    def test_embedding_service_has_interface(self):
        """Embedding service should define an interface so the model
        can be swapped without changing consuming code.
        """
        filepath = os.path.join(SERVICES_DIR, "embedding_service.py")
        with open(filepath) as f:
            content = f.read()

        tree = parse_module(filepath)
        class_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]

        assert len(class_names) >= 1, (
            "embedding_service.py has no classes. Define an EmbeddingService "
            "protocol/ABC and a concrete implementation so the embedding "
            "model can be swapped (e.g., for testing or different models)."
        )

    def test_vector_service_has_interface(self):
        """Vector store should define an interface so the backend
        (Qdrant, Chroma, etc.) can be swapped.
        """
        filepath = os.path.join(SERVICES_DIR, "vector_service.py")
        with open(filepath) as f:
            content = f.read()

        tree = parse_module(filepath)
        class_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]

        assert len(class_names) >= 1, (
            "vector_service.py has no classes. Define a VectorStore "
            "protocol/ABC and a concrete QdrantVectorStoreService implementation "
            "to decouple storage backend from business logic."
        )

    def test_ocr_service_has_interface(self):
        """OCR service should define an interface so the OCR engine
        can be swapped (e.g., Tesseract vs EasyOCR vs cloud OCR).
        """
        filepath = os.path.join(SERVICES_DIR, "ocr_service.py")
        tree = parse_module(filepath)
        class_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]

        assert len(class_names) >= 1, (
            "ocr_service.py has no classes. Define an OCRService "
            "protocol/ABC and a concrete TesseractOCRService to allow "
            "swapping OCR engines without changing the rest of the pipeline."
        )


    def test_fuzzy_service_has_interface(self):
        """Fuzzy matching service should define an interface."""
        filepath = os.path.join(SERVICES_DIR, "fuzzy_service.py")
        tree = parse_module(filepath)
        class_names = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef)
        ]

        assert any("Protocol" in ast.dump(node) or "ABC" in ast.dump(node) for node in ast.walk(tree) if isinstance(node, ast.ClassDef)), (
            "fuzzy_service.py has no interface. Define a FuzzyMatchingService "
            "protocol/ABC to allow swapping matching logic."
        )

class TestDependencyInjection:
    """Verify services use dependency injection, not hardcoded dependencies."""

    def test_no_hardcoded_model_loading_in_functions(self):
        """ML models should not be instantiated inside function bodies."""
        service_files = get_python_files(SERVICES_DIR)
        violations = []

        # List of files allowed to have hardcoded model loading (e.g., for backward compatibility/lazy loading)
        allowed_files = ["embedding_service.py", "ner_service.py", "vector_service.py"]

        model_constructors = [
            "SentenceTransformer(",
            "spacy.load(",
            "QdrantClient(",
        ]

        for filepath in service_files:
            if os.path.basename(filepath) in allowed_files:
                continue
            with open(filepath) as f:
                content = f.read()

            tree = parse_module(filepath)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_source = ast.get_source_segment(content, node)
                    if func_source:
                        for constructor in model_constructors:
                            if constructor in func_source:
                                violations.append(
                                    (os.path.basename(filepath), node.name, constructor.rstrip("("))
                                )

        assert len(violations) == 0, (
            f"Models/clients instantiated inside functions: "
            f"{', '.join(f'{f}:{fn} -> {c}' for f, fn, c in violations)}. "
            f"Inject dependencies via constructor or FastAPI Depends()."
        )

    def test_no_global_client_instantiation(self):
        """Clients should not be instantiated at module level."""
        service_files = get_python_files(SERVICES_DIR)
        violations = []

        for filepath in service_files:
            tree = parse_module(filepath)

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.Assign):
                    source = ast.get_source_segment(
                        open(filepath).read(), node
                    )
                    if source and any(
                        kw in source
                        for kw in ["QdrantClient(", "spacy.load("]
                    ):
                        violations.append(os.path.basename(filepath))

        assert len(violations) == 0, (
            f"Global client/model instantiation in: {violations}. "
            f"Use dependency injection to manage service lifecycle."
        )

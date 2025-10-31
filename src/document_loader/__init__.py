"""Document loader module."""

from src.document_loader.base import DocumentLoaderInterface
from src.document_loader.factory import create_document_loader

__all__ = ["DocumentLoaderInterface", "create_document_loader"]

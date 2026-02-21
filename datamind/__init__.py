"""
DataMind AI - Intelligent Database Documentation Agent

A production-grade AI system that automatically generates comprehensive,
human-readable user manuals for any relational database.
"""

__version__ = "1.0.0"
__author__ = "DataMind AI Team"

from datamind.config import DataMindConfig
from datamind.core.intelligence_store import IntelligenceStore

__all__ = ["DataMindConfig", "IntelligenceStore", "__version__"]

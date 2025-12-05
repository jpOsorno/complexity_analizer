"""
Módulo de análisis de complejidad computacional.
"""

from .complexity_analyzer import BasicComplexityAnalyzer, ComplexityResult
from .unified_analyzer import analyze_complexity_unified, UnifiedComplexityResult
from .recursion_analyzer import RecursionAnalyzerVisitor
from .recurrence_solver import solve_recurrence

# CORRECCIÓN: Importar analizador mejorado para aplicar parches automáticamente
try:
    from . import improved_analyzer
except ImportError:
    pass  # El analizador mejorado es opcional

__all__ = [
    'BasicComplexityAnalyzer',
    'ComplexityResult',
    'analyze_complexity_unified',
    'UnifiedComplexityResult',
    'RecursionAnalyzerVisitor',
    'solve_recurrence',
]

"""
Detector de Patrones Algorítmicos
==================================

Detecta patrones comunes en algoritmos y los compara con algoritmos conocidos.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class AlgorithmPattern:
    """Patrón algorítmico detectado"""
    name: str
    confidence: float  # 0.0 - 1.0
    description: str
    examples: List[str]
    characteristics: List[str]


class PatternDetector:
    """
    Detecta patrones algorítmicos comunes.
    
    Patrones soportados:
    - Divide and Conquer
    - Dynamic Programming
    - Greedy
    - Backtracking
    - Sorting algorithms
    - Searching algorithms
    """
    
    def __init__(self):
        self.known_patterns = self._load_known_patterns()
    
    def detect_patterns(self, code: str, complexity: str, is_recursive: bool) -> List[AlgorithmPattern]:
        """
        Detecta patrones en el código.
        
        Args:
            code: Código pseudocódigo
            complexity: Complejidad calculada (ej: "O(n²)")
            is_recursive: Si el algoritmo es recursivo
            
        Returns:
            Lista de patrones detectados
        """
        patterns = []
        
        # Detectar Divide and Conquer
        if self._is_divide_and_conquer(code, is_recursive):
            patterns.append(AlgorithmPattern(
                name="Divide and Conquer",
                confidence=0.9,
                description="Algoritmo que divide el problema en subproblemas más pequeños",
                examples=["MergeSort", "QuickSort", "Binary Search"],
                characteristics=[
                    "División del problema en partes",
                    "Resolución recursiva de subproblemas",
                    "Combinación de soluciones"
                ]
            ))
        
        # Detectar algoritmos de ordenamiento
        sort_pattern = self._detect_sorting_algorithm(code, complexity)
        if sort_pattern:
            patterns.append(sort_pattern)
        
        # Detectar algoritmos de búsqueda
        search_pattern = self._detect_search_algorithm(code, complexity)
        if search_pattern:
            patterns.append(search_pattern)
        
        # Detectar programación dinámica
        if self._is_dynamic_programming(code, is_recursive):
            patterns.append(AlgorithmPattern(
                name="Dynamic Programming",
                confidence=0.7,
                description="Algoritmo que resuelve subproblemas overlapping",
                examples=["Fibonacci con memoization", "Longest Common Subsequence"],
                characteristics=[
                    "Subproblemas overlapping",
                    "Memoization o tabulación",
                    "Optimal substructure"
                ]
            ))
        
        # Detectar greedy
        if self._is_greedy(code):
            patterns.append(AlgorithmPattern(
                name="Greedy Algorithm",
                confidence=0.6,
                description="Algoritmo que toma decisiones localmente óptimas",
                examples=["Dijkstra", "Huffman Coding", "Activity Selection"],
                characteristics=[
                    "Decisiones localmente óptimas",
                    "No backtracking",
                    "Solución construida paso a paso"
                ]
            ))
        
        return patterns
    
    def _is_divide_and_conquer(self, code: str, is_recursive: bool) -> bool:
        """Detecta patrón divide and conquer"""
        if not is_recursive:
            return False
        
        # Buscar división del problema
        has_division = any(keyword in code.lower() for keyword in [
            'floor', 'ceiling', '/ 2', 'mid', 'pivot'
        ])
        
        # Buscar múltiples llamadas recursivas
        recursive_calls = code.lower().count('call')
        
        return has_division and recursive_calls >= 2
    
    def _detect_sorting_algorithm(self, code: str, complexity: str) -> Optional[AlgorithmPattern]:
        """Detecta algoritmos de ordenamiento"""
        code_lower = code.lower()
        
        # BubbleSort
        if 'bubblesort' in code_lower or (
            'for' in code_lower and 
            code.count('for') >= 2 and
            ('swap' in code_lower or 'temp' in code_lower)
        ):
            return AlgorithmPattern(
                name="Bubble Sort",
                confidence=0.95,
                description="Algoritmo de ordenamiento por intercambio",
                examples=["BubbleSort clásico"],
                characteristics=[
                    "Dos loops anidados",
                    "Intercambio de elementos adyacentes",
                    "O(n²) en peor caso"
                ]
            )
        
        # InsertionSort
        if 'insertionsort' in code_lower or (
            'for' in code_lower and
            'while' in code_lower and
            'key' in code_lower
        ):
            return AlgorithmPattern(
                name="Insertion Sort",
                confidence=0.9,
                description="Algoritmo de ordenamiento por inserción",
                examples=["InsertionSort clásico"],
                characteristics=[
                    "Loop externo + while interno",
                    "Inserción en posición correcta",
                    "O(n²) peor caso, O(n) mejor caso"
                ]
            )
        
        # MergeSort
        if 'mergesort' in code_lower or 'merge' in code_lower:
            return AlgorithmPattern(
                name="Merge Sort",
                confidence=0.95,
                description="Algoritmo de ordenamiento por mezcla (divide and conquer)",
                examples=["MergeSort clásico"],
                characteristics=[
                    "Divide and conquer",
                    "Dos llamadas recursivas",
                    "O(n log n) en todos los casos"
                ]
            )
        
        # QuickSort
        if 'quicksort' in code_lower or 'partition' in code_lower:
            return AlgorithmPattern(
                name="Quick Sort",
                confidence=0.95,
                description="Algoritmo de ordenamiento por partición",
                examples=["QuickSort clásico"],
                characteristics=[
                    "Divide and conquer",
                    "Partición con pivot",
                    "O(n log n) promedio, O(n²) peor caso"
                ]
            )
        
        # SelectionSort
        if 'selectionsort' in code_lower or (
            'min_idx' in code_lower or 'min' in code_lower
        ):
            return AlgorithmPattern(
                name="Selection Sort",
                confidence=0.85,
                description="Algoritmo de ordenamiento por selección",
                examples=["SelectionSort clásico"],
                characteristics=[
                    "Dos loops anidados",
                    "Selección del mínimo",
                    "O(n²) en todos los casos"
                ]
            )
        
        return None
    
    def _detect_search_algorithm(self, code: str, complexity: str) -> Optional[AlgorithmPattern]:
        """Detecta algoritmos de búsqueda"""
        code_lower = code.lower()
        
        # Binary Search
        if 'binarysearch' in code_lower or (
            'mid' in code_lower and
            'left' in code_lower and
            'right' in code_lower
        ):
            return AlgorithmPattern(
                name="Binary Search",
                confidence=0.95,
                description="Búsqueda binaria en array ordenado",
                examples=["BinarySearch clásico"],
                characteristics=[
                    "Divide and conquer",
                    "Array ordenado requerido",
                    "O(log n) complejidad"
                ]
            )
        
        # Linear Search
        if 'linearsearch' in code_lower or (
            'for' in code_lower and
            code.count('for') == 1 and
            'return' in code_lower
        ):
            return AlgorithmPattern(
                name="Linear Search",
                confidence=0.8,
                description="Búsqueda lineal secuencial",
                examples=["LinearSearch clásico"],
                characteristics=[
                    "Un solo loop",
                    "Búsqueda secuencial",
                    "O(n) complejidad"
                ]
            )
        
        return None
    
    def _is_dynamic_programming(self, code: str, is_recursive: bool) -> bool:
        """Detecta patrón de programación dinámica"""
        code_lower = code.lower()
        
        # Buscar memoization
        has_memo = any(keyword in code_lower for keyword in [
            'memo', 'cache', 'dp', 'table'
        ])
        
        # Buscar overlapping subproblems (múltiples llamadas con mismos parámetros)
        has_overlap = is_recursive and code.lower().count('call') >= 2
        
        return has_memo or (has_overlap and 'fibonacci' in code_lower)
    
    def _is_greedy(self, code: str) -> bool:
        """Detecta patrón greedy"""
        code_lower = code.lower()
        
        # Buscar selección de máximo/mínimo
        has_selection = any(keyword in code_lower for keyword in [
            'max', 'min', 'best', 'optimal'
        ])
        
        # Buscar construcción iterativa
        has_iteration = 'for' in code_lower or 'while' in code_lower
        
        # No debe tener backtracking
        no_backtracking = 'backtrack' not in code_lower
        
        return has_selection and has_iteration and no_backtracking
    
    def _load_known_patterns(self) -> Dict[str, AlgorithmPattern]:
        """Carga patrones conocidos"""
        return {
            "divide_and_conquer": AlgorithmPattern(
                name="Divide and Conquer",
                confidence=1.0,
                description="Divide el problema, resuelve recursivamente, combina soluciones",
                examples=["MergeSort", "QuickSort", "Binary Search", "Karatsuba"],
                characteristics=["División", "Recursión", "Combinación"]
            ),
            "dynamic_programming": AlgorithmPattern(
                name="Dynamic Programming",
                confidence=1.0,
                description="Resuelve subproblemas overlapping con memoization",
                examples=["Fibonacci", "LCS", "Knapsack", "Edit Distance"],
                characteristics=["Overlapping subproblems", "Optimal substructure", "Memoization"]
            ),
            "greedy": AlgorithmPattern(
                name="Greedy",
                confidence=1.0,
                description="Toma decisiones localmente óptimas",
                examples=["Dijkstra", "Prim", "Kruskal", "Huffman"],
                characteristics=["Decisión local óptima", "No backtracking", "Construcción incremental"]
            ),
            "backtracking": AlgorithmPattern(
                name="Backtracking",
                confidence=1.0,
                description="Explora todas las soluciones posibles con poda",
                examples=["N-Queens", "Sudoku Solver", "Subset Sum"],
                characteristics=["Exploración exhaustiva", "Poda", "Recursión con retroceso"]
            )
        }


def demo():
    """Demo del detector de patrones"""
    
    test_cases = [
        {
            "name": "BubbleSort",
            "code": """BubbleSort(A[], n)
begin
    for i ← 1 to n-1 do
        for j ← 1 to n-i do
            if (A[j] > A[j+1]) then
                swap(A[j], A[j+1])
end""",
            "complexity": "O(n²)",
            "is_recursive": False
        },
        {
            "name": "MergeSort",
            "code": """MergeSort(A[], p, r)
begin
    if (p < r) then
        q ← floor((p + r) / 2)
        call MergeSort(A, p, q)
        call MergeSort(A, q+1, r)
        call Merge(A, p, q, r)
end""",
            "complexity": "O(n log n)",
            "is_recursive": True
        }
    ]
    
    detector = PatternDetector()
    
    print("="*70)
    print("DEMO: DETECTOR DE PATRONES")
    print("="*70)
    
    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"Algoritmo: {test['name']}")
        print(f"{'='*70}")
        
        patterns = detector.detect_patterns(
            test['code'],
            test['complexity'],
            test['is_recursive']
        )
        
        if patterns:
            for pattern in patterns:
                print(f"\n✓ Patrón detectado: {pattern.name}")
                print(f"  Confianza: {pattern.confidence*100:.0f}%")
                print(f"  Descripción: {pattern.description}")
                print(f"  Ejemplos: {', '.join(pattern.examples)}")
                print(f"  Características:")
                for char in pattern.characteristics:
                    print(f"    - {char}")
        else:
            print("\nNo se detectaron patrones conocidos")


if __name__ == "__main__":
    demo()

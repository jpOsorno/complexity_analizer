"""
Sugeridor de Optimizaciones
============================

Analiza algoritmos y sugiere optimizaciones posibles.
"""

from typing import List, Dict
from dataclasses import dataclass


@dataclass
class Optimization:
    """Sugerencia de optimización"""
    title: str
    description: str
    impact: str  # "High", "Medium", "Low"
    difficulty: str  # "Easy", "Medium", "Hard"
    example: str


class OptimizationSuggester:
    """
    Sugiere optimizaciones para algoritmos.
    
    Tipos de optimizaciones:
    - Early termination
    - Estructuras de datos más eficientes
    - Memoization/caching
    - Eliminación de operaciones redundantes
    - Paralelización
    """
    
    def suggest_optimizations(
        self,
        code: str,
        complexity: str,
        pattern_name: str = None
    ) -> List[Optimization]:
        """
        Sugiere optimizaciones para el código.
        
        Args:
            code: Código pseudocódigo
            complexity: Complejidad actual
            pattern_name: Patrón detectado (opcional)
            
        Returns:
            Lista de optimizaciones sugeridas
        """
        suggestions = []
        
        # Detectar oportunidades de early termination
        if self._can_add_early_termination(code):
            suggestions.append(Optimization(
                title="Early Termination",
                description="Agregar condición de salida temprana cuando se encuentra el resultado",
                impact="Medium",
                difficulty="Easy",
                example="""
// Antes:
for i ← 1 to n do
    if (A[i] = x) then
        found ← true

// Después:
for i ← 1 to n do
    if (A[i] = x) then
        return i  // Salida temprana
"""
            ))
        
        # Detectar oportunidades de memoization
        if self._needs_memoization(code, pattern_name):
            suggestions.append(Optimization(
                title="Memoization",
                description="Cachear resultados de subproblemas para evitar recálculos",
                impact="High",
                difficulty="Medium",
                example="""
// Agregar tabla de memoization:
memo[n]  // Array para cachear resultados

Fibonacci(n, memo)
begin
    if (memo[n] ≠ null) then
        return memo[n]  // Retornar valor cacheado
    
    if (n ≤ 1) then
        return n
    
    memo[n] ← call Fibonacci(n-1, memo) + call Fibonacci(n-2, memo)
    return memo[n]
end
"""
            ))
        
        # Detectar uso ineficiente de estructuras de datos
        if self._can_use_hash_table(code, complexity):
            suggestions.append(Optimization(
                title="Usar Hash Table",
                description="Reemplazar búsqueda lineal con hash table para O(1) lookup",
                impact="High",
                difficulty="Medium",
                example="""
// En lugar de buscar en array O(n):
for i ← 1 to n do
    if (A[i] = x) then
        return true

// Usar hash table O(1):
if (hashTable.contains(x)) then
    return true
"""
            ))
        
        # Detectar operaciones redundantes
        redundant_ops = self._detect_redundant_operations(code)
        if redundant_ops:
            suggestions.append(Optimization(
                title="Eliminar Operaciones Redundantes",
                description=f"Detectadas operaciones que se pueden pre-computar o eliminar",
                impact="Medium",
                difficulty="Easy",
                example=redundant_ops
            ))
        
        # Sugerir paralelización si es aplicable
        if self._can_parallelize(code):
            suggestions.append(Optimization(
                title="Paralelización",
                description="Operaciones independientes pueden ejecutarse en paralelo",
                impact="High",
                difficulty="Hard",
                example="""
// Loops independientes pueden paralelizarse:
parallel for i ← 1 to n do
    process(A[i])  // Cada iteración es independiente
"""
            ))
        
        # Optimizaciones específicas para BubbleSort
        if 'bubblesort' in code.lower():
            suggestions.append(Optimization(
                title="Optimizar BubbleSort",
                description="Agregar flag para detectar cuando el array ya está ordenado",
                impact="Medium",
                difficulty="Easy",
                example="""
BubbleSort(A[], n)
begin
    for i ← 1 to n-1 do
    begin
        swapped ← false
        for j ← 1 to n-i do
        begin
            if (A[j] > A[j+1]) then
            begin
                swap(A[j], A[j+1])
                swapped ← true
            end
        end
        if (not swapped) then
            break  // Array ya ordenado, salir
    end
end
"""
            ))
        
        # Optimizaciones específicas para InsertionSort
        if 'insertionsort' in code.lower():
            suggestions.append(Optimization(
                title="Binary Insertion Sort",
                description="Usar búsqueda binaria para encontrar posición de inserción",
                impact="Low",
                difficulty="Medium",
                example="""
// Usar binary search para encontrar posición:
// Reduce comparaciones de O(n) a O(log n)
// Pero movimientos siguen siendo O(n)
pos ← call BinarySearch(A, 1, i-1, key)
// Insertar en posición encontrada
"""
            ))
        
        return suggestions
    
    def _can_add_early_termination(self, code: str) -> bool:
        """Detecta si se puede agregar early termination"""
        code_lower = code.lower()
        
        # Buscar loops con búsqueda
        has_search_loop = (
            'for' in code_lower and
            ('if' in code_lower) and
            ('return' not in code_lower or code.lower().count('return') < 2)
        )
        
        return has_search_loop
    
    def _needs_memoization(self, code: str, pattern_name: str = None) -> bool:
        """Detecta si necesita memoization"""
        code_lower = code.lower()
        
        # Fibonacci sin memoization
        if 'fibonacci' in code_lower and 'memo' not in code_lower:
            return True
        
        # Recursión con múltiples llamadas y sin memoization
        if (
            'call' in code_lower and
            code.lower().count('call') >= 2 and
            'memo' not in code_lower and
            'cache' not in code_lower
        ):
            return True
        
        return False
    
    def _can_use_hash_table(self, code: str, complexity: str) -> bool:
        """Detecta si puede beneficiarse de hash table"""
        code_lower = code.lower()
        
        # Búsqueda lineal que puede ser hash table
        has_linear_search = (
            'for' in code_lower and
            ('if' in code_lower) and
            ('=' in code or '==' in code)
        )
        
        # Complejidad O(n) o peor que puede mejorarse
        needs_improvement = 'n' in complexity.lower()
        
        return has_linear_search and needs_improvement
    
    def _detect_redundant_operations(self, code: str) -> str:
        """Detecta operaciones redundantes"""
        code_lower = code.lower()
        
        # Cálculos repetidos dentro de loops
        if 'for' in code_lower:
            # Buscar expresiones que se repiten
            if '/' in code or '*' in code:
                return """
// Pre-computar valores constantes fuera del loop:
// Antes:
for i ← 1 to n do
    result ← A[i] * (n / 2)  // n/2 se calcula n veces

// Después:
half_n ← n / 2
for i ← 1 to n do
    result ← A[i] * half_n  // Calculado una vez
"""
        
        return ""
    
    def _can_parallelize(self, code: str) -> bool:
        """Detecta si puede paralelizarse"""
        code_lower = code.lower()
        
        # Loops simples sin dependencias
        has_simple_loop = 'for' in code_lower
        
        # No debe tener dependencias entre iteraciones
        no_dependencies = (
            'a[i-1]' not in code_lower and
            'a[j-1]' not in code_lower and
            'prev' not in code_lower
        )
        
        return has_simple_loop and no_dependencies


def demo():
    """Demo del sugeridor de optimizaciones"""
    
    test_cases = [
        {
            "name": "LinearSearch sin early termination",
            "code": """LinearSearch(A[], n, x)
begin
    found ← false
    for i ← 1 to n do
        if (A[i] = x) then
            found ← true
    return found
end""",
            "complexity": "O(n)"
        },
        {
            "name": "Fibonacci sin memoization",
            "code": """Fibonacci(n)
begin
    if (n ≤ 1) then
        return n
    return call Fibonacci(n-1) + call Fibonacci(n-2)
end""",
            "complexity": "O(2^n)"
        }
    ]
    
    suggester = OptimizationSuggester()
    
    print("="*70)
    print("DEMO: SUGERIDOR DE OPTIMIZACIONES")
    print("="*70)
    
    for test in test_cases:
        print(f"\n{'='*70}")
        print(f"Algoritmo: {test['name']}")
        print(f"Complejidad actual: {test['complexity']}")
        print(f"{'='*70}")
        
        optimizations = suggester.suggest_optimizations(
            test['code'],
            test['complexity']
        )
        
        if optimizations:
            for i, opt in enumerate(optimizations, 1):
                print(f"\n{i}. {opt.title}")
                print(f"   Impacto: {opt.impact} | Dificultad: {opt.difficulty}")
                print(f"   {opt.description}")
                if opt.example:
                    print(f"   Ejemplo:{opt.example}")
        else:
            print("\nNo se encontraron optimizaciones obvias")


if __name__ == "__main__":
    demo()

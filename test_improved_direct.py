"""
Test directo del analizador mejorado
"""
import sys
sys.path.insert(0, 'src')

from src.parser.parser import parse
from src.analyzer.improved_analyzer import ImprovedComplexityAnalyzer

# Test InsertionSort
code = """InsertionSort(A[], n)
begin
    for i ← 2 to n do
    begin
        key ← A[i]
        j ← i - 1
        
        while (j > 0 and A[j] > key) do
        begin
            A[j+1] ← A[j]
            j ← j - 1
        end
        
        A[j+1] ← key
    end
end"""

print("="*70)
print("TEST DIRECTO: InsertionSort")
print("="*70)

ast = parse(code)
procedure = ast.procedures[0]

analyzer = ImprovedComplexityAnalyzer()
result = analyzer.analyze_procedure(procedure)

print(f"\nResultado del analizador mejorado:")
print(f"  Worst:   {result['worst']}")
print(f"  Best:    {result['best']}")
print(f"  Average: {result['average']}")
print(f"  Max depth: {result['max_depth']}")

print(f"\nEsperado: O(n²) (loop for anidado con while)")
print(f"Obtenido: {result['worst']}")

if result['worst'] == 'O(n²)':
    print("\n✓ CORRECTO!")
else:
    print(f"\n✗ INCORRECTO - Esperaba O(n²), obtuvo {result['worst']}")

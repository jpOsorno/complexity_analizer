"""
Script de Diagnóstico para MergeSort
====================================

Ejecutar: python debug_merge_sort.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser.parser import parse
from analyzer.recursion_analyzer import RecursionAnalyzerVisitor


code = """
MergeSort(A[], p, r)
begin
    if (p < r) then
    begin
        q ← floor((p + r) / 2)
        call MergeSort(A, p, q)
        call MergeSort(A, q+1, r)
        call Merge(A, p, q, r)
    end
end
"""

print("="*70)
print("DIAGNÓSTICO MERGE SORT")
print("="*70)

try:
    # Parse
    ast = parse(code)
    proc = ast.procedures[0]
    
    print(f"\n✓ Procedimiento: {proc.name}")
    print(f"✓ Parámetros: {[p.name for p in proc.parameters]}")
    
    # Analizar con el visitor
    visitor = RecursionAnalyzerVisitor(proc.name)
    result = visitor.visit_procedure(proc)
    
    print(f"\n{'='*70}")
    print("LLAMADAS RECURSIVAS DETECTADAS")
    print(f"{'='*70}")
    print(f"Total: {len(visitor.recursive_calls)}")
    
    for i, call in enumerate(visitor.recursive_calls, 1):
        print(f"\nLlamada {i}:")
        print(f"  Función: {call.function_name}")
        print(f"  Reducción: {call.depth_reduction}")
        print(f"  Location: {call.location}")
        print(f"  in_return: {call.in_return}")
        print(f"  return_group: {call.return_group}")
        print(f"  path_condition: {call.path_condition}")
    
    print(f"\n{'='*70}")
    print("ECUACIÓN GENERADA")
    print(f"{'='*70}")
    
    if result.recurrence_equation:
        eq = result.recurrence_equation
        print(f"\nWorst case: {eq.worst_case_equation}")
        print(f"Explicación: {eq.worst_case_explanation}")
        print(f"\nBest case: {eq.best_case_equation}")
        print(f"Average case: {eq.average_case_equation}")
        
        print(f"\n{'='*70}")
        print("ANÁLISIS INTERNO")
        print(f"{'='*70}")
        
        # Recrear el análisis paso a paso
        from collections import defaultdict
        
        in_return_calls = defaultdict(list)
        statement_calls = []
        
        for call in visitor.recursive_calls:
            if call.in_return:
                in_return_calls[call.return_group].append(call)
            else:
                statement_calls.append(call)
        
        print(f"\nStatement calls: {len(statement_calls)}")
        for call in statement_calls:
            print(f"  - {call.function_name}({call.depth_reduction})")
        
        print(f"\nReturn calls: {len(in_return_calls)} grupos")
        for group_id, calls in in_return_calls.items():
            print(f"  Grupo {group_id}: {len(calls)} llamadas")
            for call in calls:
                print(f"    - {call.function_name}({call.depth_reduction})")
        
        # Verificar exclusividad
        are_mutually_exclusive = len(in_return_calls) > 1
        print(f"\n¿Mutuamente excluyentes?: {are_mutually_exclusive}")
        
        # Extraer términos
        print(f"\n{'='*70}")
        print("TÉRMINOS EXTRAÍDOS")
        print(f"{'='*70}")
        
        terms = eq._extract_terms(visitor.recursive_calls)
        print(f"\nTérminos: {terms}")
        
        # Compactar
        compacted = eq._sum_and_compact_terms(terms)
        print(f"Compactados: {compacted}")
        
        print(f"\n{'='*70}")
        print("VERIFICACIÓN")
        print(f"{'='*70}")
        
        if "2T(n/2)" in eq.worst_case_equation:
            print("\n✅ CORRECTO: Ecuación contiene 2T(n/2)")
        else:
            print("\n❌ ERROR: Ecuación NO contiene 2T(n/2)")
            print(f"   Obtenido: {eq.worst_case_equation}")
            print(f"   Esperado: T(n) = 2T(n/2) + n")
    
    else:
        print("\n❌ No se generó ecuación de recurrencia")

except Exception as e:
    print(f"\n❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
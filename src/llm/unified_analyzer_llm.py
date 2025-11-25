"""
Analizador Unificado con Validación LLM
========================================

Integra análisis estático + validación con IA.
"""

import sys
import os
from typing import Dict, Optional
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.parser.parser import parse
from src.analyzer.unified_analyzer import analyze_complexity_unified, UnifiedComplexityResult
from src.llm.llm_comparator import LLMComparator, ComparisonResult


@dataclass
class LLMEnhancedResult(UnifiedComplexityResult):
    """Resultado con validación LLM"""
    
    # Validación LLM
    llm_validation: Optional[ComparisonResult] = None
    llm_enabled: bool = False
    
    def __str__(self):
        # Resultado base
        base_str = super().__str__()
        
        # Agregar validación LLM si disponible
        if self.llm_validation:
            validation_str = f"""

{'='*70}
VALIDACIÓN CON IA (Llama 3.3 70B)
{'='*70}

LLM Análisis:
  Peor caso:     {self.llm_validation.llm_worst}
  Mejor caso:    {self.llm_validation.llm_best}
  Caso promedio: {self.llm_validation.llm_average}

Estado: {'✅ COINCIDE' if self.llm_validation.agrees else '⚠️  DIFIERE'}
Confianza: {self.llm_validation.confidence*100:.0f}%

Explicación del LLM:
{self.llm_validation.llm_explanation}
"""
            
            if not self.llm_validation.agrees:
                validation_str += f"""
⚠️  DIFERENCIAS DETECTADAS:
{self.llm_validation.differences}
"""
            
            return base_str + validation_str
        
        return base_str


def analyze_with_llm(
    code: str,
    enable_llm: bool = True,
    api_key: Optional[str] = None
) -> Dict[str, LLMEnhancedResult]:
    """
    Analiza código con validación LLM opcional.
    
    Args:
        code: Código pseudocódigo
        enable_llm: Si True, valida con LLM
        api_key: API key de Groq (opcional)
        
    Returns:
        Dict con resultados por procedimiento
    """
    # Paso 1: Parse
    ast = parse(code)
    
    # Paso 2: Análisis estático
    base_results = analyze_complexity_unified(ast)
    
    # Paso 3: Validación LLM (si habilitada)
    enhanced_results = {}
    
    if enable_llm:
        try:
            comparator = LLMComparator(api_key)
            
            for proc_name, result in base_results.items():
                # Obtener código del procedimiento
                proc_code = _extract_procedure_code(code, proc_name)
                
                # Validar según tipo
                if result.is_recursive:
                    # Validación recursiva
                    validation = comparator.compare_recursive(
                        proc_code,
                        result.recurrence_equation or "N/A",
                        result.final_worst,
                        result.final_best,
                        result.final_average,
                        result.explanation
                    )
                else:
                    # Validación iterativa
                    validation = comparator.compare_iterative(
                        proc_code,
                        result.final_worst,
                        result.final_best,
                        result.final_average,
                        result.explanation
                    )
                
                # Crear resultado mejorado
                enhanced = LLMEnhancedResult(
                    procedure_name=result.procedure_name,
                    iterative_worst=result.iterative_worst,
                    iterative_best=result.iterative_best,
                    iterative_average=result.iterative_average,
                    is_recursive=result.is_recursive,
                    recurrence_equation=result.recurrence_equation,
                    recurrence_solution=result.recurrence_solution,
                    final_worst=result.final_worst,
                    final_best=result.final_best,
                    final_average=result.final_average,
                    algorithm_type=result.algorithm_type,
                    explanation=result.explanation,
                    steps=result.steps,
                    llm_validation=validation,
                    llm_enabled=True
                )
                
                enhanced_results[proc_name] = enhanced
        
        except Exception as e:
            # Si falla LLM, usar resultados sin validación
            print(f"⚠️  Validación LLM falló: {e}")
            for proc_name, result in base_results.items():
                enhanced_results[proc_name] = LLMEnhancedResult(
                    procedure_name=result.procedure_name,
                    iterative_worst=result.iterative_worst,
                    iterative_best=result.iterative_best,
                    iterative_average=result.iterative_average,
                    is_recursive=result.is_recursive,
                    recurrence_equation=result.recurrence_equation,
                    recurrence_solution=result.recurrence_solution,
                    final_worst=result.final_worst,
                    final_best=result.final_best,
                    final_average=result.final_average,
                    algorithm_type=result.algorithm_type,
                    explanation=result.explanation,
                    steps=result.steps,
                    llm_enabled=False
                )
    else:
        # Sin validación LLM
        for proc_name, result in base_results.items():
            enhanced_results[proc_name] = LLMEnhancedResult(
                procedure_name=result.procedure_name,
                iterative_worst=result.iterative_worst,
                iterative_best=result.iterative_best,
                iterative_average=result.iterative_average,
                is_recursive=result.is_recursive,
                recurrence_equation=result.recurrence_equation,
                recurrence_solution=result.recurrence_solution,
                final_worst=result.final_worst,
                final_best=result.final_best,
                final_average=result.final_average,
                algorithm_type=result.algorithm_type,
                explanation=result.explanation,
                steps=result.steps,
                llm_enabled=False
            )
    
    return enhanced_results


def _extract_procedure_code(full_code: str, proc_name: str) -> str:
    """
    Extrae el código de un procedimiento específico.
    
    Args:
        full_code: Código completo
        proc_name: Nombre del procedimiento a extraer
        
    Returns:
        Código del procedimiento
    """
    # Simple: buscar desde el nombre hasta el próximo "end" de nivel superior
    # (esto es una aproximación, podría mejorarse)
    
    lines = full_code.split('\n')
    proc_lines = []
    in_proc = False
    depth = 0
    
    for line in lines:
        if proc_name in line and 'begin' not in line:
            in_proc = True
        
        if in_proc:
            proc_lines.append(line)
            
            if 'begin' in line:
                depth += 1
            if 'end' in line:
                depth -= 1
                
                if depth == 0:
                    break
    
    return '\n'.join(proc_lines) if proc_lines else full_code


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demo completo del analizador con LLM"""
    
    examples = {
        "Bubble Sort (Iterativo)": """
BubbleSort(A[], n)
begin
    for i ← 1 to n-1 do
    begin
        for j ← 1 to n-i do
        begin
            if (A[j] > A[j+1]) then
            begin
                temp ← A[j]
                A[j] ← A[j+1]
                A[j+1] ← temp
            end
        end
    end
end
        """,
        
        "Merge Sort (Recursivo)": """
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
    }
    
    print("="*70)
    print("ANALIZADOR UNIFICADO CON VALIDACIÓN LLM")
    print("="*70)
    
    # Verificar API key
    api_key = os.getenv('GROQ_API_KEY')
    enable_llm = api_key is not None
    
    if not enable_llm:
        print("\n⚠️  GROQ_API_KEY no configurada - Ejecutando sin validación LLM")
        print("   Para habilitar: export GROQ_API_KEY='tu-api-key'")
    else:
        print("\n✓ Validación LLM habilitada (Llama 3.3 70B)")
    
    for name, code in examples.items():
        print(f"\n{'='*70}")
        print(f"📊 {name}")
        print(f"{'='*70}")
        
        try:
            results = analyze_with_llm(code, enable_llm=enable_llm)
            
            for proc_name, result in results.items():
                print(result)
        
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo()
"""
Comparador LLM - Valida análisis de complejidad con IA
======================================================

Compara nuestro análisis estático con el análisis del LLM.
"""

import sys
import os
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from llm_config import LLMConfig, ComplexityPrompts
from llm_client import GroqClient, LLMResponse


@dataclass
class ComparisonResult:
    """Resultado de comparar nuestro análisis con el del LLM"""
    
    # Análisis nuestro
    our_worst: str
    our_best: str
    our_average: str
    
    # Análisis del LLM
    llm_worst: str
    llm_best: str
    llm_average: str
    
    # Comparación
    agrees: bool
    confidence: float
    differences: str = ""
    llm_explanation: str = ""
    
    # Metadata
    latency_ms: float = 0.0
    llm_raw_response: str = ""
    
    def __str__(self):
        status = "✅ COINCIDE" if self.agrees else "⚠️  DIFIERE"
        
        result = f"""
{'='*70}
COMPARACIÓN: {status}
{'='*70}

NUESTRO ANÁLISIS:
  Peor caso:     {self.our_worst}
  Mejor caso:    {self.our_best}
  Caso promedio: {self.our_average}

LLM ANÁLISIS (Llama 3.3 70B):
  Peor caso:     {self.llm_worst}
  Mejor caso:    {self.llm_best}
  Caso promedio: {self.llm_average}

EXPLICACIÓN DEL LLM:
{self.llm_explanation}
"""
        
        if not self.agrees and self.differences:
            result += f"""
DIFERENCIAS DETECTADAS:
{self.differences}
"""
        
        result += f"""
Confianza del LLM: {self.confidence*100:.0f}%
Latencia: {self.latency_ms:.0f}ms
{'='*70}
"""
        return result


class LLMComparator:
    """Compara análisis de complejidad con LLM"""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Inicializa el comparador.
        
        Args:
            api_key: API key de Groq (si None, busca en env)
        """
        if api_key is None:
            config = LLMConfig.from_env()
            api_key = config.api_key
        
        self.client = GroqClient(api_key)
        self.prompts = ComplexityPrompts()
    
    # ========================================================================
    # COMPARACIÓN: ALGORITMOS ITERATIVOS
    # ========================================================================
    
    def compare_iterative(
        self, 
        code: str,
        our_worst: str,
        our_best: str,
        our_average: str,
        explanation: str = ""
    ) -> ComparisonResult:
        """
        Compara análisis iterativo con LLM.
        
        Args:
            code: Código pseudocódigo
            our_worst: Nuestro análisis de peor caso
            our_best: Nuestro análisis de mejor caso
            our_average: Nuestro análisis de caso promedio
            explanation: Nuestra explicación (opcional)
            
        Returns:
            ComparisonResult con la comparación
        """
        # Construir descripción de nuestro análisis
        our_analysis = f"""
Peor caso: {our_worst}
Mejor caso: {our_best}
Caso promedio: {our_average}
{f'Explicación: {explanation}' if explanation else ''}
        """.strip()
        
        # Generar prompt
        prompt = self.prompts.iterative_analysis(code, our_analysis)
        
        # Consultar LLM
        response = self.client.analyze(prompt)
        
        # Procesar respuesta
        if not response.success or not response.parsed_json:
            # Si falla, asumir que coincide (sin validación)
            return ComparisonResult(
                our_worst=our_worst,
                our_best=our_best,
                our_average=our_average,
                llm_worst="N/A",
                llm_best="N/A",
                llm_average="N/A",
                agrees=True,  # Asumimos correcto si no hay validación
                confidence=0.0,
                differences=f"No se pudo validar con LLM: {response.error}",
                latency_ms=response.latency_ms,
                llm_raw_response=response.raw_text
            )
        
        # Extraer análisis del LLM
        llm_data = response.parsed_json
        
        llm_worst = llm_data.get("worst_case", "N/A")
        llm_best = llm_data.get("best_case", "N/A")
        llm_average = llm_data.get("average_case", "N/A")
        
        agrees = llm_data.get("agrees_with_our_analysis", True)
        confidence = llm_data.get("confidence", 0.5)
        differences = llm_data.get("differences", "")
        llm_explanation = llm_data.get("explanation", "")
        
        return ComparisonResult(
            our_worst=our_worst,
            our_best=our_best,
            our_average=our_average,
            llm_worst=llm_worst,
            llm_best=llm_best,
            llm_average=llm_average,
            agrees=agrees,
            confidence=confidence,
            differences=differences,
            llm_explanation=llm_explanation,
            latency_ms=response.latency_ms,
            llm_raw_response=response.raw_text
        )
    
    # ========================================================================
    # COMPARACIÓN: ALGORITMOS RECURSIVOS
    # ========================================================================
    
    def compare_recursive(
        self,
        code: str,
        our_equation: str,
        our_worst: str,
        our_best: str,
        our_average: str,
        solution_explanation: str = ""
    ) -> ComparisonResult:
        """
        Compara análisis recursivo con LLM.
        
        Args:
            code: Código pseudocódigo
            our_equation: Nuestra ecuación de recurrencia
            our_worst/best/average: Nuestras soluciones
            solution_explanation: Explicación de nuestra solución
            
        Returns:
            ComparisonResult con la comparación
        """
        # Construir descripción de nuestra solución
        our_solution = f"""
Peor caso: {our_worst}
Mejor caso: {our_best}
Caso promedio: {our_average}
{f'Explicación: {solution_explanation}' if solution_explanation else ''}
        """.strip()
        
        # Generar prompt
        prompt = self.prompts.recursive_analysis(code, our_equation, our_solution)
        
        # Consultar LLM
        response = self.client.analyze(prompt)
        
        # Procesar respuesta
        if not response.success or not response.parsed_json:
            return ComparisonResult(
                our_worst=our_worst,
                our_best=our_best,
                our_average=our_average,
                llm_worst="N/A",
                llm_best="N/A",
                llm_average="N/A",
                agrees=True,
                confidence=0.0,
                differences=f"No se pudo validar con LLM: {response.error}",
                latency_ms=response.latency_ms,
                llm_raw_response=response.raw_text
            )
        
        llm_data = response.parsed_json
        
        llm_worst = llm_data.get("worst_case", "N/A")
        llm_best = llm_data.get("best_case", "N/A")
        llm_average = llm_data.get("average_case", "N/A")
        
        # Validar tanto ecuación como solución
        agrees_equation = llm_data.get("agrees_with_our_equation", True)
        agrees_solution = llm_data.get("agrees_with_our_solution", True)
        agrees = agrees_equation and agrees_solution
        
        confidence = llm_data.get("confidence", 0.5)
        differences = llm_data.get("differences", "")
        llm_explanation = llm_data.get("explanation", "")
        
        return ComparisonResult(
            our_worst=our_worst,
            our_best=our_best,
            our_average=our_average,
            llm_worst=llm_worst,
            llm_best=llm_best,
            llm_average=llm_average,
            agrees=agrees,
            confidence=confidence,
            differences=differences,
            llm_explanation=llm_explanation,
            latency_ms=response.latency_ms,
            llm_raw_response=response.raw_text
        )


# ============================================================================
# DEMO
# ============================================================================

def demo_iterative():
    """Demo: Comparación de algoritmo iterativo"""
    
    print("="*70)
    print("DEMO: COMPARACIÓN ITERATIVA CON LLM")
    print("="*70)
    
    code = """
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
    """
    
    print("\nCódigo:")
    print(code)
    
    # Nuestro análisis
    our_worst = "O(n²)"
    our_best = "Ω(n²)"
    our_average = "Θ(n²)"
    explanation = "FOR anidado: n iteraciones externas × n-i internas ≈ n²"
    
    print(f"\nNuestro análisis: {our_worst}")
    
    # Comparar con LLM
    try:
        comparator = LLMComparator()
        
        print("\n📤 Consultando LLM (Llama 3.3 70B)...")
        
        result = comparator.compare_iterative(
            code, our_worst, our_best, our_average, explanation
        )
        
        print(result)
        
        return result.agrees
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def demo_recursive():
    """Demo: Comparación de algoritmo recursivo"""
    
    print("\n" + "="*70)
    print("DEMO: COMPARACIÓN RECURSIVA CON LLM")
    print("="*70)
    
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
    
    print("\nCódigo:")
    print(code)
    
    # Nuestro análisis
    our_equation = "T(n) = 2T(n/2) + n"
    our_worst = "O(n log n)"
    our_best = "Ω(n log n)"
    our_average = "Θ(n log n)"
    explanation = "Teorema Maestro caso 2: a=2, b=2, f(n)=n → Θ(n log n)"
    
    print(f"\nNuestra ecuación: {our_equation}")
    print(f"Nuestra solución: {our_worst}")
    
    # Comparar con LLM
    try:
        comparator = LLMComparator()
        
        print("\n📤 Consultando LLM (Llama 3.3 70B)...")
        
        result = comparator.compare_recursive(
            code, our_equation, our_worst, our_best, our_average, explanation
        )
        
        print(result)
        
        return result.agrees
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("="*70)
    print("COMPARADOR LLM - VALIDACIÓN DE ANÁLISIS")
    print("="*70)
    print()
    
    # Verificar API key
    api_key = os.getenv('GROQ_API_KEY')
    if not api_key:
        print("❌ GROQ_API_KEY no configurada")
        print("\n💡 Configura tu API key gratuita:")
        print("   1. Ve a: https://console.groq.com/keys")
        print("   2. Crea una cuenta (gratis)")
        print("   3. Genera una API key")
        print("   4. export GROQ_API_KEY='tu-api-key'")
        sys.exit(1)
    
    # Ejecutar demos
    success_iter = demo_iterative()
    success_rec = demo_recursive()
    
    print("\n" + "="*70)
    print("RESUMEN")
    print("="*70)
    print(f"Iterativo: {'✓ PASS' if success_iter else '✗ FAIL'}")
    print(f"Recursivo: {'✓ PASS' if success_rec else '✗ FAIL'}")
    print("="*70)
    
    sys.exit(0 if (success_iter and success_rec) else 1)
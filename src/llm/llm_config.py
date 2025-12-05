"""
Configuración para integración con LLM (Groq API)
=================================================

Modelo: Llama 3.3 70B (gratuito, rápido, excelente en análisis técnico)
API: Groq (https://console.groq.com)

Características:
- 30,000 tokens/minuto gratuitos
- Latencia ultra-baja (~500ms)
- Excelente para análisis de algoritmos
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Intentar cargar variables de entorno desde un archivo .env si está disponible
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # Si python-dotenv no está instalado, continuar sin fallar
    pass


@dataclass
class LLMConfig:
    """Configuración del LLM"""
    
    # API Configuration
    api_key: str
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.3-70b-versatile"
    
    # Generation Parameters
    temperature: float = 0.1  # Baja para análisis técnico preciso
    max_tokens: int = 2000
    top_p: float = 0.9
    
    # Retry Configuration
    max_retries: int = 3
    timeout: int = 30
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """Carga configuración desde variables de entorno"""
        api_key = os.getenv('GROQ_API_KEY')
        
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY no encontrada. "
                "Obtén tu API key gratuita en: https://console.groq.com/keys"
            )
        
        return cls(api_key=api_key)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a diccionario para logging"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url
        }


# ============================================================================
# PROMPTS PARA ANÁLISIS DE COMPLEJIDAD
# ============================================================================

class ComplexityPrompts:
    """Plantillas de prompts para análisis de complejidad"""
    
    @staticmethod
    def iterative_analysis(code: str, our_analysis: str) -> str:
        """Prompt para análisis de algoritmos iterativos"""
        return f"""Eres un experto en análisis de complejidad computacional. Analiza el siguiente algoritmo ITERATIVO en pseudocódigo y determina su complejidad temporal.

ALGORITMO:
```
{code}
```

NUESTRO ANÁLISIS:
{our_analysis}

INSTRUCCIONES:
1. Analiza la complejidad temporal del algoritmo
2. Identifica:
   - Ciclos (FOR, WHILE, REPEAT)
   - Profundidad de anidamiento
   - Condiciones de salida temprana (early exit)
   - Operaciones dominantes

3. Determina:
   - Peor caso: O(?)
   - Mejor caso: Ω(?)
   - Caso promedio: Θ(?)

4. Compara con nuestro análisis y valida si es correcto

FORMATO DE RESPUESTA (JSON):
{{
  "worst_case": "O(...)",
  "best_case": "Ω(...)",
  "average_case": "Θ(...)",
  "explanation": "Explicación detallada del análisis",
  "agrees_with_our_analysis": true/false,
  "differences": "Explicación de diferencias si las hay",
  "confidence": 0.0-1.0
}}

Responde SOLO con el JSON, sin texto adicional."""

    @staticmethod
    def recursive_analysis(code: str, our_equation: str, our_solution: str) -> str:
        """Prompt para análisis de algoritmos recursivos"""
        return f"""Eres un experto en análisis de algoritmos recursivos y resolución de ecuaciones de recurrencia.

ALGORITMO RECURSIVO:
```
{code}
```

NUESTRA ECUACIÓN DE RECURRENCIA:
{our_equation}

NUESTRA SOLUCIÓN:
{our_solution}

INSTRUCCIONES:
1. Analiza el algoritmo recursivo
2. Deriva la ecuación de recurrencia correcta
3. Identifica:
   - Tipo de recursión (lineal, binaria, divide-y-vencerás)
   - Número de llamadas recursivas
   - Reducción de tamaño del problema
   - Costo no recursivo

4. Resuelve la ecuación usando:
   - Teorema Maestro (si aplica)
   - Método de Iteración
   - Árbol de Recursión
   - Ecuación Característica (si aplica)

5. Valida nuestra ecuación y solución

FORMATO DE RESPUESTA (JSON):
{{
  "recurrence_equation": "T(n) = ...",
  "recursion_type": "lineal/binaria/divide-conquer/...",
  "solution_method": "master_theorem/iteration/tree/characteristic",
  "worst_case": "O(...)",
  "best_case": "Ω(...)",
  "average_case": "Θ(...)",
  "explanation": "Explicación paso a paso",
  "agrees_with_our_equation": true/false,
  "agrees_with_our_solution": true/false,
  "differences": "Explicación de diferencias",
  "confidence": 0.0-1.0
}}

Responde SOLO con el JSON, sin texto adicional."""

    @staticmethod
    def hybrid_analysis(code: str, iterative_part: str, recursive_part: str) -> str:
        """Prompt para algoritmos híbridos (ej: QuickSort)"""
        return f"""Eres un experto en análisis de algoritmos híbridos (que combinan iteración y recursión).

ALGORITMO HÍBRIDO:
```
{code}
```

ANÁLISIS ITERATIVO:
{iterative_part}

ANÁLISIS RECURSIVO:
{recursive_part}

INSTRUCCIONES:
1. Identifica componentes iterativos y recursivos
2. Analiza cómo interactúan (multiplicación, suma, dominancia)
3. Determina complejidad combinada
4. Considera casos especiales (ej: partición desbalanceada en QuickSort)

FORMATO DE RESPUESTA (JSON):
{{
  "algorithm_type": "hybrid",
  "iterative_component": "O(...)",
  "recursive_component": "O(...)",
  "combined_complexity": "O(...)",
  "worst_case": "O(...)",
  "best_case": "Ω(...)",
  "average_case": "Θ(...)",
  "explanation": "Cómo se combinan los componentes",
  "confidence": 0.0-1.0
}}

Responde SOLO con el JSON, sin texto adicional."""


# ============================================================================
# CONFIGURACIÓN DE EJEMPLO
# ============================================================================

if __name__ == "__main__":
    # Ejemplo de uso
    print("="*70)
    print("CONFIGURACIÓN LLM - GROQ")
    print("="*70)
    
    # Cargar configuración
    try:
        config = LLMConfig.from_env()
        print("\n✓ Configuración cargada exitosamente")
        print(f"\nDetalles:")
        for key, value in config.to_dict().items():
            print(f"  {key}: {value}")
        
        print("\n💡 Para obtener tu API key gratuita:")
        print("   1. Ve a: https://console.groq.com/keys")
        print("   2. Crea una cuenta (gratis)")
        print("   3. Genera una API key")
        print("   4. Configura: export GROQ_API_KEY='tu-api-key'")
        
    except ValueError as e:
        print(f"\n❌ {e}")
        print("\n📝 Instrucciones:")
        print("   1. Obtén tu API key en: https://console.groq.com/keys")
        print("   2. En tu terminal:")
        print("      export GROQ_API_KEY='tu-api-key'")
        print("   3. O crea un archivo .env:")
        print("      GROQ_API_KEY=tu-api-key")
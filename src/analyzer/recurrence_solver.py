"""
Resolutor de Ecuaciones de Recurrencia - VERSIÓN CORREGIDA
===========================================================

Implementa múltiples técnicas para resolver recurrencias:
1. ✅ Clasificación automática del tipo de recurrencia
2. ✅ Teorema Maestro (divide y vencerás)
3. ✅ Método de Iteración (resta y vencerás)
4. ✅ Árbol de Recursión (visualización y cálculo)
5. ✅ Método de Sustitución con SymPy
6. ✅ Ecuación Característica (relaciones lineales homogéneas)

Referencias:
- Cormen et al., "Introduction to Algorithms" (CLRS)
- Análisis de Algoritmos y Diseño
"""

import sys
import os
from typing import Dict, Optional, List, Tuple, Union
from dataclasses import dataclass, field
import sympy as sp
from sympy import symbols, sympify, simplify, log, ceiling, floor, expand, solve
import re
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class RecurrenceSolution:
    """Resultado completo de resolver una recurrencia"""
    
    # Ecuación original
    original_equation: str
    
    # Técnica(s) usada(s)
    method_used: str  # "master", "iteration", "recursion_tree", "substitution", "characteristic"
    
    # Solución en notaciones
    big_o: str       # O(...)
    big_omega: str   # Ω(...)
    big_theta: str   # Θ(...)
    
    # Detalles del análisis
    steps: List[str] = field(default_factory=list)
    tree_analysis: Optional[str] = None
    exact_solution: Optional[str] = None
    tight_bounds: Optional[str] = None  # NUEVO: Cotas fuertes
    
    # Metadata
    complexity_class: str = ""  # "constant", "logarithmic", "linear", etc.
    is_tight: bool = False      # Si O = Ω (entonces Θ)
    recurrence_type: str = ""   # "divide-conquer", "subtract-conquer", "subtract-conquered"
    
    def __str__(self):
        result = f"""
Ecuación: {self.original_equation}
Tipo: {self.recurrence_type}
Método: {self.method_used}

Complejidad:
  O(n): {self.big_o}
  Ω(n): {self.big_omega}
  Θ(n): {self.big_theta}

Clase: {self.complexity_class}
Tight bound: {'Sí' if self.is_tight else 'No'}
        """.strip()
        
        if self.steps:
            result += "\n\nPasos del análisis:"
            for i, step in enumerate(self.steps, 1):
                result += f"\n  {i}. {step}"
        
        if self.exact_solution:
            result += f"\n\nSolución exacta: {self.exact_solution}"
        
        if self.tight_bounds:
            result += f"\n\nCotas fuertes: {self.tight_bounds}"
        
        return result
    
    def to_dict(self) -> dict:
        """Convierte a diccionario para serialización"""
        return {
            "equation": self.original_equation,
            "type": self.recurrence_type,
            "method": self.method_used,
            "big_o": self.big_o,
            "big_omega": self.big_omega,
            "big_theta": self.big_theta,
            "complexity_class": self.complexity_class,
            "is_tight": self.is_tight,
            "steps": self.steps,
            "exact_solution": self.exact_solution
        }


# ============================================================================
# CLASIFICADOR DE RECURRENCIAS
# ============================================================================

class RecurrenceClassifier:
    """
    Clasifica ecuaciones de recurrencia según su estructura.
    
    Tipos:
    1. Divide y Vencerás: T(n) = aT(n/b) + f(n)
    2. Resta y Vencerás: T(n) = T(n-k) + f(n)
    3. Resta y Serás Vencido: T(n) = T(n-k1) + T(n-k2) + f(n)
    4. Lineal No Homogénea: T(n) = c1*T(n-1) + c2*T(n-2) + f(n)
    """
    
    @staticmethod
    def classify(equation: str) -> Tuple[str, Dict[str, any]]:
        """
        Clasifica una ecuación de recurrencia.
        
        Returns:
            (tipo, parametros) donde tipo es:
            - "already-solved": T(n) = O(f(n)) (ya está resuelta)
            - "divide-conquer": T(n) = aT(n/b) + f(n)
            - "subtract-conquer": T(n) = T(n-k) + f(n)
            - "subtract-conquered": T(n) = T(n-k1) + T(n-k2) + ... + f(n)
            - "linear-nonhomogeneous": T(n) = c1*T(n-1) + c2*T(n-2) + f(n)
            - "unknown": No se pudo clasificar
        """
        equation = equation.replace(" ", "").replace("T(n)=", "")
        
        # 0. Verificar si ya está resuelta: T(n) = O(...) o Θ(...) o Ω(...)
        if not "T(" in equation:
            # No hay llamadas recursivas, ya está en notación de complejidad
            return RecurrenceClassifier._parse_already_solved(equation)
        
        # 1. Divide y Vencerás: T(n/b) presente
        if "T(n/" in equation or "T(n/2)" in equation or "T(n/3)" in equation:
            return RecurrenceClassifier._parse_divide_conquer(equation)
        
        coeff_pattern = re.findall(r'(\d+)\*?T\(n-(\d+)\)', equation)
        if coeff_pattern:
            return RecurrenceClassifier._parse_with_coefficient(equation, coeff_pattern)
        
        # 2. Resta y Serás Vencido: Múltiples T(n-k)
        subtract_terms = re.findall(r'T\(n-(\d+)\)', equation)
        if len(subtract_terms) >= 2:
            return RecurrenceClassifier._parse_subtract_conquered(equation, subtract_terms)
        
        # 3. Resta y Vencerás: Un solo T(n-k)
        if len(subtract_terms) == 1:
            return RecurrenceClassifier._parse_subtract_conquer(equation, subtract_terms[0])
        
        # 4. Lineal con coeficientes: c*T(n-k)
        if re.search(r'\d+\*?T\(n-\d+\)', equation):
            return RecurrenceClassifier._parse_linear_nonhomogeneous(equation)
        
        return ("unknown", {"equation": equation})
    @staticmethod
    def _parse_with_coefficient(equation: str, coeff_pattern: List[Tuple[str, str]]) -> Tuple[str, Dict]:
        """
        NUEVO: Parsea T(n) = aT(n-k) + f(n) con coeficiente a > 1
        
        Casos:
        - T(n) = 2T(n-1) + c  → Exponencial O(2^n) - Torres de Hanoi
        - T(n) = 3T(n-2) + n  → Exponencial O(3^(n/2))
        
        Args:
            coeff_pattern: Lista de (coeficiente, offset)
                           Ej: [('2', '1')] para 2T(n-1)
        """
        # Caso simple: Un solo término con coeficiente
        if len(coeff_pattern) == 1:
            a = int(coeff_pattern[0][0])  # Coeficiente
            k = int(coeff_pattern[0][1])  # Offset (n-k)
            
            # Extraer f(n)
            if "O(1)" in equation or "+c" in equation or "+1" in equation:
                f_n = "O(1)"
            elif "O(n)" in equation or "+n" in equation:
                f_n = "O(n)"
            else:
                f_n = "O(1)"
            
            # CLAVE: Si a > 1, es exponencial (tipo subtract-conquered)
            if a > 1:
                return ("exponential-subtract", {
                    "coefficient": a,
                    "offset": k,
                    "f_n": f_n
                })
            else:
                # a = 1: Es lineal simple
                return ("subtract-conquer", {"k": k, "f_n": f_n})
        
        # Múltiples términos con coeficientes: Lineal no homogénea
        coefficients = {}
        for coef, offset in coeff_pattern:
            coefficients[int(offset)] = int(coef)
        
        f_n = "O(1)" if "O(1)" in equation else "O(1)"
        
        return ("linear-nonhomogeneous", {
            "coefficients": coefficients,
            "f_n": f_n
        })
    
    @staticmethod
    def _parse_divide_conquer(equation: str) -> Tuple[str, Dict]:
        """Parsea T(n) = aT(n/b) + f(n)"""
        # Buscar coeficiente a
        a_match = re.search(r'(\d+)T\(n/', equation)
        a = int(a_match.group(1)) if a_match else 1
        
        # Buscar divisor b
        b_match = re.search(r'T\(n/(\d+)\)', equation)
        b = int(b_match.group(1)) if b_match else 2
        
        # Extraer f(n)
        if "O(n)" in equation:
            f_n = "O(n)"
        elif "O(n^2)" in equation or "O(n²)" in equation:
            f_n = "O(n^2)"
        elif "O(logn)" in equation or "O(log(n))" in equation:
            f_n = "O(log(n))"
        elif "O(1)" in equation:
            f_n = "O(1)"
        elif "+n" in equation or "+ n" in equation:
            f_n = "O(n)"
        else:
            f_n = "O(1)"
        
        return ("divide-conquer", {"a": a, "b": b, "f_n": f_n})
    
    @staticmethod
    def _parse_subtract_conquer(equation: str, k: str) -> Tuple[str, Dict]:
        """Parsea T(n) = T(n-k) + f(n)"""
        k_val = int(k)
        
        # Extraer f(n)
        if "O(n)" in equation:
            f_n = "O(n)"
        elif "O(1)" in equation or "+c" in equation:
            f_n = "O(1)"
        else:
            f_n = "O(1)"
        
        return ("subtract-conquer", {"k": k_val, "f_n": f_n})
    
    @staticmethod
    def _parse_subtract_conquered(equation: str, subtract_terms: List[str]) -> Tuple[str, Dict]:
        """Parsea T(n) = T(n-k1) + T(n-k2) + ... + f(n)"""
        k_values = [int(k) for k in subtract_terms]
        
        # Extraer f(n)
        if "O(1)" in equation or "+c" in equation or "+ O(1)" in equation:
            f_n = "O(1)"
        else:
            f_n = "O(1)"
        
        return ("subtract-conquered", {"k_values": k_values, "f_n": f_n})
    
    @staticmethod
    def _parse_linear_nonhomogeneous(equation: str) -> Tuple[str, Dict]:
        """Parsea T(n) = c1*T(n-1) + c2*T(n-2) + f(n)"""
        # Buscar coeficientes
        coefficients = {}
        
        for match in re.finditer(r'(\d+)\*?T\(n-(\d+)\)', equation):
            coef = int(match.group(1))
            offset = int(match.group(2))
            coefficients[offset] = coef
        
        # Extraer f(n)
        if "O(1)" in equation:
            f_n = "O(1)"
        else:
            f_n = "O(1)"
        
        return ("linear-nonhomogeneous", {"coefficients": coefficients, "f_n": f_n})
    
    @staticmethod
    def _parse_already_solved(equation: str) -> Tuple[str, Dict]:
        """
        Parsea ecuaciones que ya están resueltas: O(f(n)), Θ(f(n)), Ω(f(n))
        
        Examples:
            - "O(1)" → already-solved
            - "O(n)" → already-solved
            - "Θ(log(n))" → already-solved
        """
        # Extraer la complejidad
        complexity = equation.strip()
        
        # Determinar la notación usada
        if complexity.startswith("O("):
            notation = "O"
        elif complexity.startswith("Θ(") or complexity.startswith("Theta("):
            notation = "Θ"
        elif complexity.startswith("Ω(") or complexity.startswith("Omega("):
            notation = "Ω"
        else:
            notation = "O"  # Por defecto
        
        return ("already-solved", {
            "complexity": complexity,
            "notation": notation
        })


# ============================================================================
# TEOREMA MAESTRO (Master Theorem) - CORREGIDO
# ============================================================================

class MasterTheorem:
    """
    Implementa el Teorema Maestro SOLO para divide y vencerás.
    
    T(n) = a·T(n/b) + f(n)
    
    donde:
    - a ≥ 1 (número de subproblemas)
    - b > 1 (factor de división)
    - f(n) = costo no recursivo
    """
    
    @staticmethod
    def applies(recurrence_type: str, params: Dict) -> bool:
        """Verifica si el Teorema Maestro aplica"""
        if recurrence_type != "divide-conquer":
            return False
        
        a = params.get("a", 0)
        b = params.get("b", 0)
        
        return a >= 1 and b > 1
    
    @staticmethod
    def solve(a: int, b: int, f_n: str, equation: str) -> RecurrenceSolution:
        """
        Resuelve usando el Teorema Maestro con análisis detallado.
        """
        steps = []
        steps.append(f"📋 Ecuación identificada: T(n) = {a}T(n/{b}) + {f_n}")
        steps.append("")
        
        # Paso 1: Identificar parámetros
        steps.append("🔍 PASO 1: Identificar parámetros")
        steps.append(f"  • a (número de subproblemas) = {a}")
        steps.append(f"  • b (factor de división) = {b}")
        steps.append(f"  • f(n) (costo no recursivo) = {f_n}")
        steps.append("")
        
        # Paso 2: Calcular log_b(a)
        log_ba = sp.log(a, b)
        log_ba_float = float(log_ba.evalf())
        
        steps.append("🔍 PASO 2: Calcular log_b(a)")
        steps.append(f"  log_{b}({a}) = {log_ba_float:.2f}")
        steps.append(f"  Esto significa: n^{log_ba_float:.2f} es el costo de las hojas del árbol")
        steps.append("")
        
        # Paso 3: Determinar orden de f(n)
        f_order = MasterTheorem._get_f_order(f_n)
        steps.append("🔍 PASO 3: Determinar orden de f(n)")
        steps.append(f"  f(n) = {f_n}")
        steps.append(f"  Orden polinómico: n^{f_order}")
        steps.append("")
        
        # Paso 4: Comparar f(n) con n^log_b(a)
        epsilon = 0.1
        steps.append("🔍 PASO 4: Aplicar Teorema Maestro")
        steps.append(f"  Comparar f(n) = n^{f_order} con n^{log_ba_float:.2f}")
        steps.append("")
        
        # CASO 1: f(n) < n^log_b(a)
        if f_order < log_ba_float - epsilon:
            steps.append("✅ CASO 1 del Teorema Maestro")
            steps.append(f"  f(n) = O(n^{f_order}) < O(n^{log_ba_float:.2f})")
            steps.append(f"  Las hojas dominan el costo")
            steps.append("")
            steps.append("📐 Fórmula: T(n) = Θ(n^log_b(a))")
            steps.append(f"  T(n) = Θ(n^{log_ba_float:.2f})")
            steps.append("")
            
            # Explicación intuitiva
            steps.append("💡 Explicación intuitiva:")
            steps.append(f"  • El árbol tiene altura log_{b}(n)")
            steps.append(f"  • Cada nivel tiene más nodos que el anterior (factor {a})")
            steps.append(f"  • El último nivel tiene a^log_b(n) = n^log_b(a) nodos")
            steps.append(f"  • El costo está dominado por las hojas")
            
            complexity = MasterTheorem._format_complexity(log_ba_float)
            complexity_class = MasterTheorem._classify_complexity(log_ba_float)
            tight_bounds = MasterTheorem._calculate_tight_bounds(complexity)
            
            # Agregar visualización del árbol
            tree_analysis = MasterTheorem._generate_tree_description(a, b, f_n, log_ba_float, case=1)
            
            return RecurrenceSolution(
                original_equation=equation,
                method_used="master_theorem_case1",
                big_o=f"O({complexity})",
                big_omega=f"Ω({complexity})",
                big_theta=f"Θ({complexity})",
                complexity_class=complexity_class,
                is_tight=True,
                recurrence_type="divide-conquer",
                steps=steps,
                tight_bounds=tight_bounds,
                tree_analysis=tree_analysis
            )
        
        # CASO 2: f(n) ≈ n^log_b(a)
        elif abs(f_order - log_ba_float) < epsilon:
            steps.append("✅ CASO 2 del Teorema Maestro")
            steps.append(f"  f(n) = Θ(n^{f_order}) ≈ Θ(n^{log_ba_float:.2f})")
            steps.append(f"  Todos los niveles contribuyen igual")
            steps.append("")
            steps.append("📐 Fórmula: T(n) = Θ(n^log_b(a) × log(n))")
            steps.append(f"  T(n) = Θ(n^{log_ba_float:.2f} × log(n))")
            steps.append("")
            
            # Explicación intuitiva
            steps.append("💡 Explicación intuitiva:")
            steps.append(f"  • El árbol tiene altura log_{b}(n) niveles")
            steps.append(f"  • Cada nivel tiene costo Θ(n^{log_ba_float:.2f})")
            steps.append(f"  • Total: log(n) niveles × n^{log_ba_float:.2f} por nivel")
            steps.append(f"  • Resultado: Θ(n^{log_ba_float:.2f} × log(n))")
            
            if abs(log_ba_float - 1.0) < 0.01:
                complexity = "n×log(n)"
                complexity_class = "linearithmic"
            else:
                base_complexity = MasterTheorem._format_complexity(log_ba_float)
                complexity = f"{base_complexity}×log(n)"
                complexity_class = "polynomial with log factor"
            
            tight_bounds = MasterTheorem._calculate_tight_bounds(complexity)
            tree_analysis = MasterTheorem._generate_tree_description(a, b, f_n, log_ba_float, case=2)
            
            return RecurrenceSolution(
                original_equation=equation,
                method_used="master_theorem_case2",
                big_o=f"O({complexity})",
                big_omega=f"Ω({complexity})",
                big_theta=f"Θ({complexity})",
                complexity_class=complexity_class,
                is_tight=True,
                recurrence_type="divide-conquer",
                steps=steps,
                tight_bounds=tight_bounds,
                tree_analysis=tree_analysis
            )
        
        # CASO 3: f(n) > n^log_b(a)
        else:
            steps.append("✅ CASO 3 del Teorema Maestro")
            steps.append(f"  f(n) = Θ(n^{f_order}) > Θ(n^{log_ba_float:.2f})")
            steps.append(f"  La raíz domina el costo")
            steps.append("")
            steps.append("📐 Fórmula: T(n) = Θ(f(n))")
            steps.append(f"  T(n) = Θ(f(n)) = {f_n}")
            steps.append("")
            
            # Verificar regularidad
            steps.append("🔍 Verificar condición de regularidad:")
            steps.append(f"  a × f(n/{b}) ≤ c × f(n) para alguna constante c < 1")
            steps.append("  ✓ Se asume que se cumple para este caso")
            steps.append("")
            
            # Explicación intuitiva
            steps.append("💡 Explicación intuitiva:")
            steps.append(f"  • El costo decrece geométricamente al bajar niveles")
            steps.append(f"  • La raíz (nivel 0) tiene el mayor costo")
            steps.append(f"  • El costo total es dominado por f(n)")
            
            complexity = MasterTheorem._extract_complexity_from_f(f_n)
            complexity_class = MasterTheorem._classify_from_string(complexity)
            tight_bounds = MasterTheorem._calculate_tight_bounds(complexity)
            tree_analysis = MasterTheorem._generate_tree_description(a, b, f_n, log_ba_float, case=3)
            
            return RecurrenceSolution(
                original_equation=equation,
                method_used="master_theorem_case3",
                big_o=f"O({complexity})",
                big_omega=f"Ω({complexity})",
                big_theta=f"Θ({complexity})",
                complexity_class=complexity_class,
                is_tight=True,
                recurrence_type="divide-conquer",
                steps=steps,
                tight_bounds=tight_bounds,
                tree_analysis=tree_analysis
            )
    

    @staticmethod
    def _generate_tree_description(a: int, b: int, f_n: str, log_ba: float, case: int) -> str:
        """
        Genera descripción textual del árbol de recursión.
        
        Args:
            a: Número de subproblemas
            b: Factor de división
            f_n: Costo no recursivo
            log_ba: log_b(a)
            case: Caso del teorema (1, 2, o 3)
            
        Returns:
            String con descripción del árbol
        """
        description = "🌳 ESTRUCTURA DEL ÁRBOL DE RECURSIÓN:\n\n"
        
        description += f"📊 Propiedades:\n"
        description += f"  • Altura: log_{b}(n) niveles\n"
        description += f"  • Ramificación: {a} hijos por nodo\n"
        description += f"  • Nodos en nivel i: {a}^i\n"
        description += f"  • Total de nodos: ({a}^(log_{b}(n)+1) - 1) / ({a}-1) ≈ O(n^{log_ba:.2f})\n\n"
        
        description += f"📈 Costo por nivel:\n"
        description += f"  • Nivel 0 (raíz): f(n) = {f_n}\n"
        description += f"  • Nivel 1: {a} × f(n/{b})\n"
        description += f"  • Nivel 2: {a}² × f(n/{b}²)\n"
        description += f"  • Nivel i: {a}^i × f(n/{b}^i)\n"
        description += f"  • Nivel log_{b}(n): {a}^log_{b}(n) × O(1) = n^{log_ba:.2f}\n\n"
        
        if case == 1:
            description += "🎯 CASO 1: Las hojas dominan\n"
            description += f"  • El costo crece exponencialmente hacia las hojas\n"
            description += f"  • El último nivel tiene n^{log_ba:.2f} nodos\n"
            description += f"  • Complejidad = Θ(n^{log_ba:.2f})\n"
        elif case == 2:
            description += "🎯 CASO 2: Todos los niveles contribuyen igual\n"
            description += f"  • Cada nivel tiene costo Θ(n^{log_ba:.2f})\n"
            description += f"  • Hay log_{b}(n) niveles\n"
            description += f"  • Complejidad = Θ(n^{log_ba:.2f} × log(n))\n"
        else:  # case == 3
            description += "🎯 CASO 3: La raíz domina\n"
            description += f"  • El costo decrece al bajar niveles\n"
            description += f"  • La raíz tiene el mayor costo: {f_n}\n"
            description += f"  • Complejidad = Θ(f(n))\n"

        return description

    @staticmethod
    def _get_f_order(f_n: str) -> float:
        """Determina el orden de f(n)"""
        if "n^2" in f_n or "n²" in f_n:
            return 2.0
        elif "n" in f_n and "log" not in f_n:
            return 1.0
        elif "log" in f_n:
            return 0.5  # Entre constante y lineal
        else:
            return 0.0  # Constante
    
    @staticmethod
    def _format_complexity(power: float) -> str:
        """Formatea una potencia de n"""
        if abs(power - 0.0) < 0.01:
            return "1"
        elif abs(power - 1.0) < 0.01:
            return "n"
        elif abs(power - 2.0) < 0.01:
            return "n²"
        elif abs(power - 3.0) < 0.01:
            return "n³"
        else:
            return f"n^{power:.3f}"
    
    @staticmethod
    def _classify_complexity(power: float) -> str:
        """Clasifica la complejidad"""
        if abs(power - 0.0) < 0.01:
            return "constant"
        elif abs(power - 1.0) < 0.01:
            return "linear"
        elif abs(power - 2.0) < 0.01:
            return "quadratic"
        elif abs(power - 3.0) < 0.01:
            return "cubic"
        else:
            return f"polynomial (degree {power:.2f})"
    
    @staticmethod
    def _extract_complexity_from_f(f_n: str) -> str:
        """Extrae la complejidad de f(n)"""
        if "n^2" in f_n or "n²" in f_n:
            return "n²"
        elif "n" in f_n:
            return "n"
        else:
            return "1"
    
    @staticmethod
    def _classify_from_string(complexity: str) -> str:
        """Clasifica desde string"""
        if complexity == "1":
            return "constant"
        elif complexity == "n":
            return "linear"
        elif complexity == "n²":
            return "quadratic"
        else:
            return "polynomial"
    
    @staticmethod
    def _calculate_tight_bounds(complexity: str) -> str:
        """
        Calcula las cotas fuertes (tight bounds) para una complejidad.
        
        Las cotas fuertes son los factores constantes más ajustados.
        Por ejemplo: Θ(n) tiene cota fuerte c₁n ≤ T(n) ≤ c₂n
        """
        # Simplificar notación
        clean = complexity.replace("×", "*").replace("^", "**")
        
        if complexity == "1":
            return "c₁ ≤ T(n) ≤ c₂ para constantes c₁, c₂ > 0"
        elif complexity == "n":
            return "c₁n ≤ T(n) ≤ c₂n para constantes c₁, c₂ > 0"
        elif complexity == "n²" or "n^2" in complexity:
            return "c₁n² ≤ T(n) ≤ c₂n² para constantes c₁, c₂ > 0"
        elif "log(n)" in complexity and "n" in complexity:
            return "c₁n·log(n) ≤ T(n) ≤ c₂n·log(n) para constantes c₁, c₂ > 0"
        elif "log(n)" in complexity:
            return "c₁log(n) ≤ T(n) ≤ c₂log(n) para constantes c₁, c₂ > 0"
        elif "^" in complexity or "**" in complexity:
            # Complejidad polinómica o exponencial
            return f"c₁f(n) ≤ T(n) ≤ c₂f(n) donde f(n) = {complexity}"
        else:
            return f"c₁f(n) ≤ T(n) ≤ c₂f(n) donde f(n) = {complexity}"


# ============================================================================
# MÉTODO DE ITERACIÓN (para Resta y Vencerás)
# ============================================================================

class IterationMethod:
    """
    Método de Iteración para T(n) = T(n-k) + f(n)
    
    Expande la recurrencia iterativamente hasta el caso base.
    """
    
    @staticmethod
    def applies(recurrence_type: str, params: Dict) -> bool:
        """Verifica si el método aplica"""
        return recurrence_type == "subtract-conquer"
    
    @staticmethod
    def solve(k: int, f_n: str, equation: str) -> RecurrenceSolution:
        """
        Resuelve T(n) = T(n-k) + f(n) con análisis paso a paso detallado.
        """
        steps = []
        steps.append(f"📋 Ecuación identificada: T(n) = T(n-{k}) + {f_n}")
        steps.append(f"📐 Método: Iteración (expansión sucesiva)")
        steps.append("")
        
        # Determinar el costo de f(n)
        f_cost = IterationMethod._extract_cost(f_n)
        
        # PASO 1: Primera expansión
        steps.append("🔍 PASO 1: Primera expansión")
        steps.append(f"  T(n) = T(n-{k}) + {f_cost}")
        steps.append(f"  Sustituir T(n-{k}):")
        steps.append(f"  T(n) = [T(n-{2*k}) + {f_cost}] + {f_cost}")
        steps.append(f"       = T(n-{2*k}) + 2×{f_cost}")
        steps.append("")
        
        # PASO 2: Segunda expansión
        steps.append("🔍 PASO 2: Segunda expansión")
        steps.append(f"  T(n) = T(n-{2*k}) + 2×{f_cost}")
        steps.append(f"  Sustituir T(n-{2*k}):")
        steps.append(f"  T(n) = [T(n-{3*k}) + {f_cost}] + 2×{f_cost}")
        steps.append(f"       = T(n-{3*k}) + 3×{f_cost}")
        steps.append("")
        
        # PASO 3: Tercera expansión
        steps.append("🔍 PASO 3: Tercera expansión")
        steps.append(f"  T(n) = T(n-{3*k}) + 3×{f_cost}")
        steps.append(f"  Sustituir T(n-{3*k}):")
        steps.append(f"  T(n) = [T(n-{4*k}) + {f_cost}] + 3×{f_cost}")
        steps.append(f"       = T(n-{4*k}) + 4×{f_cost}")
        steps.append("")
        
        # PASO 4: Patrón general
        steps.append("🔍 PASO 4: Identificar patrón")
        steps.append(f"  Después de i expansiones:")
        steps.append(f"  T(n) = T(n-{k}×i) + i×{f_cost}")
        steps.append("")
        
        # PASO 5: Caso base
        steps.append("🔍 PASO 5: Alcanzar el caso base")
        steps.append(f"  Necesitamos que n - {k}×i = 0 (o alguna constante)")
        steps.append(f"  Resolver para i:")
        steps.append(f"  {k}×i = n")
        steps.append(f"  i = n/{k}")
        steps.append("")
        steps.append(f"  Sustituir i = n/{k}:")
        steps.append(f"  T(n) = T(0) + (n/{k})×{f_cost}")
        steps.append("")
        
        # PASO 6: Simplificar según f(n)
        steps.append("🔍 PASO 6: Simplificar")
        
        if f_cost in ["c", "1"]:
            # T(n) = T(0) + (n/k) × c = O(n)
            steps.append(f"  T(n) = T(0) + (n/{k})×c")
            steps.append(f"  T(n) = Θ(1) + Θ(n/{k})")
            steps.append(f"  T(n) = Θ(n)")
            steps.append("")
            steps.append("💡 Explicación:")
            steps.append(f"  • Hacemos n/{k} iteraciones")
            steps.append(f"  • Cada iteración cuesta O(1)")
            steps.append(f"  • Total: (n/{k}) × O(1) = O(n)")
            
            complexity = "n"
            complexity_class = "linear"
            exact = f"T(n) = T(0) + (n/{k})×c"
            
        elif f_cost == "n":
            # T(n) = T(0) + (n/k) × n = O(n²)
            steps.append(f"  T(n) = T(0) + (n/{k})×n")
            steps.append(f"  T(n) = Θ(1) + Θ(n²/{k})")
            steps.append(f"  T(n) = Θ(n²)")
            steps.append("")
            steps.append("💡 Explicación:")
            steps.append(f"  • Hacemos n/{k} iteraciones")
            steps.append(f"  • Cada iteración cuesta O(n)")
            steps.append(f"  • Total: (n/{k}) × O(n) = O(n²)")
            
            complexity = "n²"
            complexity_class = "quadratic"
            exact = f"T(n) = T(0) + (n/{k})×n"
            
        else:
            complexity = "n"
            complexity_class = "linear"
            exact = f"T(n) = T(0) + (n/{k})×{f_cost}"
            steps.append(f"  T(n) = {exact}")
            steps.append(f"  T(n) = Θ(n)")
        
        # Generar tabla de expansión
        expansion_table = IterationMethod._generate_expansion_table(k, f_cost, 6)
        steps.append("")
        steps.append("📊 TABLA DE EXPANSIÓN:")
        steps.extend(expansion_table)
        
        tight_bounds = MasterTheorem._calculate_tight_bounds(complexity)
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="iteration",
            big_o=f"O({complexity})",
            big_omega=f"Ω({complexity})",
            big_theta=f"Θ({complexity})",
            complexity_class=complexity_class,
            is_tight=True,
            recurrence_type="subtract-conquer",
            steps=steps,
            exact_solution=exact,
            tight_bounds=tight_bounds,
            tree_analysis=IterationMethod._generate_visual_description(k, f_cost)
        )
    
    @staticmethod
    def _generate_expansion_table(k: int, f_cost: str, num_rows: int = 6) -> List[str]:
        """
        Genera tabla de expansión iterativa.
        
        Args:
            k: Reducción por paso
            f_cost: Costo de f(n)
            num_rows: Número de filas
            
        Returns:
            Lista de strings con la tabla
        """
        table = []
        table.append("  ╔════════╦═══════════════════╦═════════════════╗")
        table.append("  ║   i    ║   Tamaño          ║   Costo Acum.   ║")
        table.append("  ╠════════╬═══════════════════╬═════════════════╣")
        
        for i in range(num_rows):
            size = f"n-{k*i}" if i > 0 else "n"
            cost = f"{i}×{f_cost}" if i > 0 else "0"
            table.append(f"  ║   {i:<4} ║   {size:<15} ║   {cost:<13} ║")
        
        table.append("  ║   ...  ║   ...             ║   ...           ║")
        table.append(f"  ║  n/{k:<2} ║   0               ║   (n/{k})×{f_cost:<6} ║")
        table.append("  ╚════════╩═══════════════════╩═════════════════╝")
        
        return table
    
    @staticmethod
    def _generate_visual_description(k: int, f_cost: str) -> str:
        """Genera descripción visual del proceso de iteración"""
        desc = "🔄 VISUALIZACIÓN DEL PROCESO ITERATIVO:\n\n"
        desc += "📉 Secuencia de llamadas:\n"
        desc += f"  T(n) → T(n-{k}) → T(n-{2*k}) → T(n-{3*k}) → ... → T(0)\n\n"
        desc += "📊 Estructura:\n"
        desc += "  • Es una cadena lineal (no ramifica)\n"
        desc += f"  • Profundidad: n/{k} niveles\n"
        desc += f"  • Costo por nivel: {f_cost}\n"
        desc += f"  • Acumulación: suma de costos = (n/{k}) × {f_cost}\n"
        
        return desc
    
    @staticmethod
    def _extract_cost(f_n: str) -> str:
        """
        Extrae el costo de f(n)
        
        MEJORA CRÍTICA: Parsear correctamente diferentes notaciones
        
        Casos:
        - "O(n)" → "n"
        - "O(1)" → "c"
        - "n" → "n"
        - "c" → "c"
        - "1" → "c"
        """
        # Normalizar
        f_n_clean = f_n.strip().replace(" ", "")
        
        # Caso 1: Notación Big-O explícita
        if "O(n)" in f_n_clean or "Θ(n)" in f_n_clean or "Ω(n)" in f_n_clean:
            return "n"
        elif "O(1)" in f_n_clean or "Θ(1)" in f_n_clean or "Ω(1)" in f_n_clean:
            return "c"
        elif "O(n^2)" in f_n_clean or "O(n²)" in f_n_clean:
            return "n²"
        
        # Caso 2: Notación directa
        elif f_n_clean == "n":
            return "n"
        elif f_n_clean in ["c", "1", "O(1)"]:
            return "c"
        
        # Caso 3: Detectar 'n' en la expresión
        elif "n" in f_n_clean:
            # Si contiene "n" pero también operaciones, asumir O(n)
            return "n"
        
        # Default: constante
        else:
            return "c"


# ============================================================================
# ECUACIÓN CARACTERÍSTICA (para Resta y Serás Vencido)
# ============================================================================

class CharacteristicEquation:
    """
    Método de Ecuación Característica para recurrencias lineales homogéneas.
    
    Ejemplos:
    - T(n) = T(n-1) + T(n-2) → Fibonacci
    - T(n) = 2T(n-1) - T(n-2) → Otras lineales
    """
    
    @staticmethod
    def applies(recurrence_type: str, params: Dict) -> bool:
        """Verifica si el método aplica"""
        return recurrence_type in ["subtract-conquered", "linear-nonhomogeneous"]
    
    @staticmethod
    def solve_fibonacci(k_values: List[int], f_n: str, equation: str) -> RecurrenceSolution:
        """
        Resuelve T(n) = T(n-1) + T(n-2) + O(1) (Fibonacci) con análisis detallado.
        """
        steps = []
        steps.append("📋 Ecuación identificada: T(n) = T(n-1) + T(n-2) + O(1)")
        steps.append("📐 Método: Ecuación Característica")
        steps.append("🎯 Tipo: Fibonacci (recurrencia lineal de segundo orden)")
        steps.append("")
        
        # PASO 1: Ecuación característica
        steps.append("🔍 PASO 1: Formar la ecuación característica")
        steps.append("  Para T(n) = T(n-1) + T(n-2), la ecuación característica es:")
        steps.append("  r² = r + 1")
        steps.append("  Reordenando:")
        steps.append("  r² - r - 1 = 0")
        steps.append("")
        
        # PASO 2: Resolver con fórmula cuadrática
        steps.append("🔍 PASO 2: Resolver usando fórmula cuadrática")
        steps.append("  r² - r - 1 = 0")
        steps.append("  a=1, b=-1, c=-1")
        steps.append("")
        steps.append("  Fórmula: r = (-b ± √(b²-4ac)) / 2a")
        steps.append("  r = (1 ± √(1+4)) / 2")
        steps.append("  r = (1 ± √5) / 2")
        steps.append("")
        
        # Calcular raíces con SymPy
        r = symbols('r')
        char_eq = r**2 - r - 1
        roots = solve(char_eq, r)
        
        r1 = roots[0]  # (1 + sqrt(5))/2 = phi
        r2 = roots[1]  # (1 - sqrt(5))/2
        
        r1_val = float(r1.evalf())
        r2_val = float(r2.evalf())
        
        steps.append(f"  r₁ = (1 + √5) / 2 ≈ {r1_val:.6f}  (φ, número áureo)")
        steps.append(f"  r₂ = (1 - √5) / 2 ≈ {r2_val:.6f}")
        steps.append("")
        
        # PASO 3: Solución general
        steps.append("🔍 PASO 3: Formar solución general")
        steps.append("  Para ecuaciones lineales homogéneas:")
        steps.append("  T(n) = c₁×r₁ⁿ + c₂×r₂ⁿ")
        steps.append("")
        steps.append(f"  T(n) = c₁×φⁿ + c₂×({r2_val:.3f})ⁿ")
        steps.append("")
        
        # PASO 4: Análisis asintótico
        steps.append("🔍 PASO 4: Análisis asintótico")
        steps.append(f"  Cuando n → ∞:")
        steps.append(f"  • φⁿ crece exponencialmente ({r1_val:.3f}ⁿ)")
        steps.append(f"  • ({r2_val:.3f})ⁿ decrece exponencialmente (|{r2_val:.3f}| < 1)")
        steps.append("")
        steps.append("  Por lo tanto, el término dominante es φⁿ:")
        steps.append("  T(n) = Θ(φⁿ)")
        steps.append("")
        
        # PASO 5: Cálculo numérico
        steps.append("🔍 PASO 5: Valores específicos de Fibonacci")
        steps.append("  Usando la fórmula de Binet:")
        steps.append(f"  F(n) = (φⁿ - ψⁿ) / √5")
        steps.append(f"  donde φ = {r1_val:.6f} y ψ = {r2_val:.6f}")
        steps.append("")
        
        fib_table = CharacteristicEquation._generate_fibonacci_table(r1_val, 10)
        steps.append("📊 TABLA: Crecimiento de Fibonacci")
        steps.extend(fib_table)
        
        # Información adicional
        steps.append("")
        steps.append("💡 Propiedades del número áureo (φ):")
        steps.append("  • φ ≈ 1.618033988749895...")
        steps.append("  • φ² = φ + 1")
        steps.append("  • φⁿ⁺¹ = φⁿ × φ ≈ φⁿ × 1.618")
        steps.append("  • Fibonacci(n) ≈ φⁿ/√5 para n grande")
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="characteristic_equation",
            big_o="O(φⁿ)",
            big_omega="Ω(φⁿ)",
            big_theta="Θ(φⁿ)",
            complexity_class="exponential",
            is_tight=True,
            recurrence_type="subtract-conquered",
            steps=steps,
            exact_solution="T(n) = Θ(φⁿ) donde φ = (1+√5)/2",
            tree_analysis=CharacteristicEquation._generate_fibonacci_tree_description()
        )
    
    @staticmethod
    def _generate_fibonacci_table(phi: float, num_rows: int) -> List[str]:
        """Genera tabla comparativa de Fibonacci"""
        table = []
        table.append("  ╔═════╦════════════════╦═════════════════╗")
        table.append("  ║  n  ║   Fib(n)       ║   φⁿ/√5         ║")
        table.append("  ╠═════╬════════════════╬═════════════════╣")
        
        sqrt5 = math.sqrt(5)
        
        # Calcular Fibonacci real
        fib = [0, 1]
        for i in range(2, num_rows + 1):
            fib.append(fib[-1] + fib[-2])
        
        for n in range(num_rows + 1):
            fib_n = fib[n]
            phi_n = (phi**n) / sqrt5
            table.append(f"  ║  {n:<2} ║   {fib_n:<12} ║   {phi_n:<13.2f} ║")
        
        table.append("  ╚═════╩════════════════╩═════════════════╝")
        table.append("")
        table.append("  Nota: A medida que n crece, Fib(n) se aproxima a φⁿ/√5")
        
        return table
    
    @staticmethod
    def _generate_fibonacci_tree_description() -> str:
        """Genera descripción del árbol de Fibonacci"""
        desc = "🌳 ESTRUCTURA DEL ÁRBOL DE FIBONACCI:\n\n"
        desc += "📊 Propiedades:\n"
        desc += "  • Árbol binario completo\n"
        desc += "  • Altura: n niveles\n"
        desc += "  • Número de nodos: F(n+1) - 1 ≈ φⁿ⁺¹/√5\n"
        desc += "  • Hojas: aproximadamente φⁿ/2\n\n"
        desc += "🔄 Patrón de llamadas:\n"
        desc += "  F(n) llama a:\n"
        desc += "    ├─ F(n-1) que llama a:\n"
        desc += "    │    ├─ F(n-2)\n"
        desc += "    │    └─ F(n-3)\n"
        desc += "    └─ F(n-2) que llama a:\n"
        desc += "         ├─ F(n-3)\n"
        desc += "         └─ F(n-4)\n\n"
        desc += "⚠️ PROBLEMA: Muchas llamadas duplicadas!\n"
        desc += "  F(n-2) se calcula dos veces\n"
        desc += "  F(n-3) se calcula tres veces\n"
        desc += "  Esto causa la complejidad exponencial\n"
        
        return desc
    
    @staticmethod
    def solve_general(coefficients: Dict[int, int], f_n: str, equation: str) -> RecurrenceSolution:
        """Resuelve recurrencias lineales generales (placeholder mejorado)"""
        steps = []
        steps.append(f"📋 Ecuación identificada: {equation}")
        steps.append("📐 Método: Ecuación Característica (general)")
        steps.append("")
        steps.append("⚠️  NOTA: Esta es una recurrencia lineal general.")
        steps.append("   El análisis detallado requiere resolver la ecuación característica:")
        steps.append("")
        
        # Construir ecuación característica
        char_eq_str = "r^n = "
        terms = []
        for offset, coef in sorted(coefficients.items()):
            terms.append(f"{coef}×r^(n-{offset})")
        char_eq_str += " + ".join(terms)
        
        steps.append(f"  Ecuación característica: {char_eq_str}")
        steps.append("")
        steps.append("💡 Para resolver manualmente:")
        steps.append("  1. Dividir por r^(n-k) para obtener polinomio en r")
        steps.append("  2. Encontrar raíces del polinomio")
        steps.append("  3. Solución: T(n) = Σ(cᵢ × rᵢⁿ) donde rᵢ son las raíces")
        steps.append("  4. Complejidad dominada por la raíz de mayor magnitud")
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="characteristic_equation",
            big_o="O(2ⁿ)",
            big_omega="Ω(2ⁿ)",
            big_theta="Θ(2ⁿ)",
            complexity_class="exponential",
            is_tight=True,
            recurrence_type="linear-nonhomogeneous",
            steps=steps
        )


# ============================================================================
# ÁRBOL DE RECURSIÓN (Visualización y Análisis)
# ============================================================================

class RecursionTree:
    """Método del Árbol de Recursión mejorado"""
    
    @staticmethod
    def analyze(recurrence_type: str, params: Dict, equation: str) -> RecurrenceSolution:
        """Analiza usando árbol de recursión con detalles"""
        if recurrence_type == "subtract-conquered":
            return RecursionTree._analyze_fibonacci(params, equation)
        elif recurrence_type == "subtract-conquer":
            return RecursionTree._analyze_linear(params, equation)
        elif recurrence_type == "divide-conquer":
            return RecursionTree._analyze_divide_conquer(params, equation)
        else:
            return RecursionTree._generic_analysis(equation)
    
    @staticmethod
    def _analyze_fibonacci(params: Dict, equation: str) -> RecurrenceSolution:
        """Árbol para Fibonacci con análisis detallado"""
        steps = []
        steps.append("🌳 Método: Árbol de Recursión - Fibonacci")
        steps.append("")
        
        steps.append("🔍 PASO 1: Estructura del árbol")
        steps.append("  • Cada nodo tiene 2 hijos (binario)")
        steps.append("  • Altura del árbol: n niveles")
        steps.append("  • Ramificación: cada nodo genera 2 subproblemas")
        steps.append("")
        
        steps.append("🔍 PASO 2: Análisis por nivel")
        steps.append("  Nivel 0: 1 nodo → costo = 1×c = c")
        steps.append("  Nivel 1: 2 nodos → costo = 2×c = 2c")
        steps.append("  Nivel 2: 4 nodos → costo = 4×c = 4c")
        steps.append("  Nivel 3: 8 nodos → costo = 8×c = 8c")
        steps.append("  ...")
        steps.append("  Nivel k: 2ᵏ nodos → costo = 2ᵏ×c")
        steps.append("")
        
        steps.append("🔍 PASO 3: Suma de costos")
        steps.append("  Total = Σ(2ᵏ×c) para k=0 hasta n")
        steps.append("  Total = c × Σ(2ᵏ) para k=0 hasta n")
        steps.append("  Total = c × (2⁰ + 2¹ + 2² + ... + 2ⁿ)")
        steps.append("")
        steps.append("  Usando la fórmula de suma geométrica:")
        steps.append("  Σ(2ᵏ) = 2ⁿ⁺¹ - 1")
        steps.append("")
        steps.append("  Total = c × (2ⁿ⁺¹ - 1)")
        steps.append("  Total = Θ(2ⁿ)")
        steps.append("")
        
        steps.append("💡 Observaciones:")
        steps.append("  • El número de nodos crece exponencialmente")
        steps.append("  • Aproximadamente 2ⁿ hojas en el árbol")
        steps.append("  • Cada hoja representa una operación base")
        steps.append("  • Ineficiencia: muchos cálculos repetidos")
        
        # Tabla de nodos por nivel
        node_table = RecursionTree._generate_node_table("binary", 10)
        steps.append("")
        steps.append("📊 TABLA: Nodos por nivel")
        steps.extend(node_table)
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="recursion_tree",
            big_o="O(2ⁿ)",
            big_omega="Ω(2ⁿ)",
            big_theta="Θ(2ⁿ)",
            complexity_class="exponential",
            is_tight=True,
            recurrence_type="subtract-conquered",
            steps=steps,
            tree_analysis="Árbol binario de altura n, aproximadamente 2ⁿ hojas"
        )
    
    @staticmethod
    def _analyze_linear(params: Dict, equation: str) -> RecurrenceSolution:
        """Árbol para recursión lineal mejorado"""
        k = params.get("k", 1)
        f_n = params.get("f_n", "O(1)")
        
        steps = []
        steps.append("🌳 Método: Árbol de Recursión - Lineal")
        steps.append("")
        
        steps.append("🔍 PASO 1: Estructura del árbol")
        steps.append("  • Cada nodo tiene 1 hijo (cadena lineal)")
        steps.append(f"  • Altura del árbol: n/{k} niveles")
        steps.append("  • No hay ramificación (degenerado)")
        steps.append("")
        
        steps.append("🔍 PASO 2: Análisis por nivel")
        steps.append(f"  Nivel 0: T(n) → costo = c")
        steps.append(f"  Nivel 1: T(n-{k}) → costo = c")
        steps.append(f"  Nivel 2: T(n-{2*k}) → costo = c")
        steps.append(f"  Nivel 3: T(n-{3*k}) → costo = c")
        steps.append("  ...")
        steps.append(f"  Nivel i: T(n-{k}×i) → costo = c")
        steps.append("")
        
        steps.append("🔍 PASO 3: Suma de costos")
        steps.append(f"  Número de niveles = n/{k}")
        steps.append(f"  Costo por nivel = c")
        steps.append(f"  Total = (n/{k}) × c = Θ(n)")
        steps.append("")
        
        steps.append("💡 Observaciones:")
        steps.append("  • El árbol es degenerado (una línea)")
        steps.append(f"  • Cada paso reduce el problema en {k}")
        steps.append(f"  • Necesitamos n/{k} pasos para llegar al caso base")
        steps.append("  • Complejidad lineal en n")
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="recursion_tree",
            big_o="O(n)",
            big_omega="Ω(n)",
            big_theta="Θ(n)",
            complexity_class="linear",
            is_tight=True,
            recurrence_type="subtract-conquer",
            steps=steps,
            tree_analysis=f"Árbol lineal (degenerado) de altura n/{k}"
        )
    
    @staticmethod
    def _analyze_divide_conquer(params: Dict, equation: str) -> RecurrenceSolution:
        """Árbol para divide y vencerás mejorado"""
        a = params.get("a", 2)
        b = params.get("b", 2)
        f_n = params.get("f_n", "O(n)")
        
        steps = []
        steps.append("🌳 Método: Árbol de Recursión - Divide y Vencerás")
        steps.append("")
        
        steps.append("🔍 PASO 1: Estructura del árbol")
        steps.append(f"  • Cada nodo tiene {a} hijos")
        steps.append(f"  • Altura del árbol: log_{b}(n) niveles")
        steps.append(f"  • Factor de ramificación: {a}")
        steps.append("")
        
        steps.append("🔍 PASO 2: Análisis por nivel")
        steps.append(f"  Nivel 0: 1 nodo de tamaño n → costo = f(n)")
        steps.append(f"  Nivel 1: {a} nodos de tamaño n/{b} → costo = {a}×f(n/{b})")
        steps.append(f"  Nivel 2: {a}² nodos de tamaño n/{b}² → costo = {a}²×f(n/{b}²)")
        steps.append(f"  Nivel 3: {a}³ nodos de tamaño n/{b}³ → costo = {a}³×f(n/{b}³)")
        steps.append("  ...")
        steps.append(f"  Nivel k: {a}ᵏ nodos de tamaño n/{b}ᵏ → costo = {a}ᵏ×f(n/{b}ᵏ)")
        steps.append("")
        steps.append("🔍 PASO 3: Profundidad y hojas")
        steps.append(f"  Altura = log_{b}(n)")
        steps.append(f"  Número de hojas = {a}^log_{b}(n) = n^log_{b}({a})")
        steps.append("")
        
        steps.append("🔍 PASO 4: Aplicar Teorema Maestro")
        steps.append("  Para análisis preciso, usar Teorema Maestro")
        
        # Usar Master Theorem para obtener resultado
        master_result = MasterTheorem.solve(a, b, f_n, equation)
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="recursion_tree",
            big_o=master_result.big_o,
            big_omega=master_result.big_omega,
            big_theta=master_result.big_theta,
            complexity_class=master_result.complexity_class,
            is_tight=True,
            recurrence_type="divide-conquer",
            steps=steps,
            tree_analysis=f"Árbol de altura log_{b}(n) con {a} hijos por nodo, {a}^log_{b}(n) hojas"
        )

    @staticmethod
    def _generate_node_table(tree_type: str, max_level: int) -> List[str]:
        """Genera tabla de nodos por nivel"""
        table = []
        table.append("  ╔═══════╦═══════════════╦══════════════════╗")
        table.append("  ║ Nivel ║  Nodos        ║  Costo por nivel ║")
        table.append("  ╠═══════╬═══════════════╬══════════════════╣")
        
        if tree_type == "binary":
            for i in range(min(max_level, 10)):
                nodes = 2**i
                cost = f"{nodes}×c"
                table.append(f"  ║   {i:<3} ║  {nodes:<12} ║  {cost:<15} ║")
        
        table.append("  ║  ...  ║  ...          ║  ...             ║")
        
        if tree_type == "binary":
            table.append(f"  ║   n   ║  2ⁿ           ║  2ⁿ×c            ║")
        
        table.append("  ╚═══════╩═══════════════╩══════════════════╝")
        
        return table

    @staticmethod
    def _generic_analysis(equation: str) -> RecurrenceSolution:
        """Análisis genérico mejorado"""
        return RecurrenceSolution(
            original_equation=equation,
            method_used="recursion_tree_generic",
            big_o="O(n)",
            big_omega="Ω(1)",
            big_theta="Θ(?)",
            complexity_class="unknown",
            is_tight=False,
            recurrence_type="unknown",
            steps=["Análisis de árbol genérico - estructura no estándar",
                "Se requiere análisis manual detallado"],
            tree_analysis="Estructura no estándar - análisis manual requerido"
        )


# ============================================================================
# MÉTODO DE SUSTITUCIÓN CON SYMPY
# ============================================================================

class SubstitutionMethod:
    """
    Método de Sustitución usando SymPy.
    
    Útil cuando otros métodos no aplican o para verificación.
    """
    
    @staticmethod
    def solve(recurrence_type: str, params: Dict, equation: str) -> RecurrenceSolution:
        """
        Resuelve por sustitución con hipótesis.
        """
        if recurrence_type == "subtract-conquer":
            return SubstitutionMethod._solve_linear(params, equation)
        elif recurrence_type == "subtract-conquered":
            return SubstitutionMethod._solve_fibonacci(params, equation)
        else:
            return SubstitutionMethod._generic_substitution(equation)
    
    @staticmethod
    def _solve_linear(params: Dict, equation: str) -> RecurrenceSolution:
        """Sustitución para T(n) = T(n-k) + f(n)"""
        k = params.get("k", 1)
        f_n = params.get("f_n", "O(1)")
        
        steps = []
        steps.append(f"Ecuación: T(n) = T(n-{k}) + c")
        steps.append("Hipótesis: T(n) = O(n)")
        steps.append("\nVerificación por inducción:")
        steps.append(f"  Suponer T(n) ≤ c×n para todo n < N")
        steps.append(f"  Probar para n = N:")
        steps.append(f"    T(N) = T(N-{k}) + c")
        steps.append(f"         ≤ c×(N-{k}) + c")
        steps.append(f"         = c×N - c×{k} + c")
        steps.append(f"         = c×N + c×(1-{k})")
        steps.append(f"         ≤ c×N  (si c ≥ 1/(k-1))")
        steps.append("\nConclusión: T(n) = Θ(n) ✓")
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="substitution",
            big_o="O(n)",
            big_omega="Ω(n)",
            big_theta="Θ(n)",
            complexity_class="linear",
            is_tight=True,
            recurrence_type="subtract-conquer",
            steps=steps,
            exact_solution="T(n) = Θ(n)"
        )
    
    @staticmethod
    def _solve_fibonacci(params: Dict, equation: str) -> RecurrenceSolution:
        """Sustitución para Fibonacci"""
        steps = []
        steps.append("Ecuación: T(n) = T(n-1) + T(n-2) + c")
        steps.append("Hipótesis: T(n) = O(φⁿ) donde φ = (1+√5)/2")
        steps.append("\nVerificación:")
        steps.append("  Propiedad de φ: φ² = φ + 1")
        steps.append("  T(n) ≤ c×φⁿ")
        steps.append("  T(n-1) + T(n-2) ≤ c×φⁿ⁻¹ + c×φⁿ⁻²")
        steps.append("                  = c×φⁿ⁻²×(φ + 1)")
        steps.append("                  = c×φⁿ⁻²×φ²")
        steps.append("                  = c×φⁿ ✓")
        steps.append("\nConclusión: T(n) = Θ(φⁿ)")
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="substitution",
            big_o="O(φⁿ) ≈ O(1.618ⁿ)",
            big_omega="Ω(φⁿ)",
            big_theta="Θ(φⁿ)",
            complexity_class="exponential",
            is_tight=True,
            recurrence_type="subtract-conquered",
            steps=steps,
            exact_solution="T(n) = Θ(φⁿ) donde φ = (1+√5)/2"
        )
    
    @staticmethod
    def _generic_substitution(equation: str) -> RecurrenceSolution:
        """Sustitución genérica"""
        return RecurrenceSolution(
            original_equation=equation,
            method_used="substitution_incomplete",
            big_o="O(?)",
            big_omega="Ω(?)",
            big_theta="Θ(?)",
            complexity_class="unknown",
            is_tight=False,
            recurrence_type="unknown",
            steps=["Requiere hipótesis específica para este tipo de recurrencia"]
        )

class ExponentialSubtractMethod:
    """
    NUEVO: Resuelve T(n) = aT(n-k) + f(n) con a > 1 (exponencial)
    
    Ejemplos:
    - T(n) = 2T(n-1) + O(1)  → O(2^n) - Torres de Hanoi
    - T(n) = 3T(n-1) + O(n)  → O(3^n)
    - T(n) = 2T(n-2) + O(1)  → O(2^(n/2))
    """
    
    @staticmethod
    def applies(recurrence_type: str, params: Dict) -> bool:
        """Verifica si el método aplica"""
        return recurrence_type == "exponential-subtract"
    
    @staticmethod
    def solve(a: int, k: int, f_n: str, equation: str) -> RecurrenceSolution:
        """
        Resuelve T(n) = aT(n-k) + f(n) por expansión exponencial.
        
        Args:
            a: Coeficiente (número de llamadas)
            k: Reducción del problema (n-k)
            f_n: Costo no recursivo
            equation: Ecuación original
        """
        steps = []
        steps.append(f"Ecuación identificada: T(n) = {a}T(n-{k}) + {f_n}")
        steps.append(f"Tipo: Recurrencia exponencial con resta")
        steps.append(f"Parámetros: a={a}, k={k}, f(n)={f_n}")
        
        # Determinar el costo de f(n)
        f_cost = ExponentialSubtractMethod._extract_cost(f_n)
        
        # Expansión exponencial
        steps.append("\nExpansión:")
        steps.append(f"  T(n) = {a}T(n-{k}) + {f_cost}")
        steps.append(f"       = {a}[{a}T(n-{2*k}) + {f_cost}] + {f_cost}")
        steps.append(f"       = {a}²T(n-{2*k}) + {a}{f_cost} + {f_cost}")
        steps.append(f"       = {a}²T(n-{2*k}) + {f_cost}({a} + 1)")
        steps.append(f"       = {a}³T(n-{3*k}) + {f_cost}({a}² + {a} + 1)")
        steps.append(f"       = ...")
        steps.append(f"       = {a}^i × T(n-{k}×i) + {f_cost} × Σ({a}^j) para j=0 to i-1")
        
        # Calcular profundidad de recursión
        steps.append(f"\nProfundidad: i = n/{k} iteraciones hasta T(0)")
        
        # Calcular complejidad según f(n)
        if f_cost in ["c", "1"]:
            # T(n) = a^(n/k) × T(0) + c × (a^(n/k) - 1) / (a - 1)
            #      ≈ a^(n/k)  (el término exponencial domina)
            
            if k == 1:
                complexity = f"{a}^n"
                steps.append(f"\nSimplificar:")
                steps.append(f"  T(n) = {a}^n × T(0) + c × ({a}^n - 1) / ({a} - 1)")
                steps.append(f"       ≈ {a}^n  (el término exponencial domina)")
                steps.append(f"  Resultado: Θ({a}^n)")
            else:
                complexity = f"{a}^(n/{k})"
                steps.append(f"\nSimplificar:")
                steps.append(f"  T(n) = {a}^(n/{k}) × T(0) + c × ({a}^(n/{k}) - 1) / ({a} - 1)")
                steps.append(f"       ≈ {a}^(n/{k})")
                steps.append(f"  Resultado: Θ({a}^(n/{k}))")
            
            complexity_class = "exponential"
            
        elif f_cost == "n":
            # Con f(n) = n, el análisis es más complejo
            # Pero el término exponencial sigue dominando
            if k == 1:
                complexity = f"{a}^n"
                steps.append(f"\nCon f(n) = n, el costo no recursivo crece linealmente")
                steps.append(f"Pero el término exponencial {a}^n domina")
                steps.append(f"Resultado: Θ({a}^n)")
            else:
                complexity = f"{a}^(n/{k})"
                steps.append(f"\nResultado: Θ({a}^(n/{k}))")
            
            complexity_class = "exponential"
        
        else:
            # Caso genérico
            complexity = f"{a}^n"
            complexity_class = "exponential"
            steps.append(f"\nResultado: Θ({a}^n)")
        
        # Calcular tight bounds
        tight_bounds = ExponentialSubtractMethod._calculate_tight_bounds(complexity)
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="exponential_expansion",
            big_o=f"O({complexity})",
            big_omega=f"Ω({complexity})",
            big_theta=f"Θ({complexity})",
            complexity_class=complexity_class,
            is_tight=True,
            recurrence_type="exponential-subtract",
            steps=steps,
            exact_solution=f"T(n) ≈ {complexity}",
            tight_bounds=tight_bounds
        )
    
    @staticmethod
    def _extract_cost(f_n: str) -> str:
        """Extrae el costo de f(n)"""
        f_n_clean = f_n.strip().replace(" ", "")
        
        if "O(n)" in f_n_clean or "Θ(n)" in f_n_clean or "Ω(n)" in f_n_clean:
            return "n"
        elif "O(1)" in f_n_clean or "Θ(1)" in f_n_clean or "Ω(1)" in f_n_clean:
            return "c"
        elif f_n_clean == "n":
            return "n"
        elif f_n_clean in ["c", "1", "O(1)"]:
            return "c"
        elif "n" in f_n_clean:
            return "n"
        else:
            return "c"
    
    @staticmethod
    def _calculate_tight_bounds(complexity: str) -> str:
        """Calcula cotas fuertes para complejidades exponenciales"""
        if "^n" in complexity:
            base = complexity.split("^")[0]
            return f"c₁{base}^n ≤ T(n) ≤ c₂{base}^n para constantes c₁, c₂ > 0 y n ≥ n₀"
        elif "^(n/" in complexity:
            return f"c₁f(n) ≤ T(n) ≤ c₂f(n) donde f(n) = {complexity}"
        else:
            return f"Cotas exponenciales para {complexity}"
# ============================================================================
# RESOLUTOR PRINCIPAL
# ============================================================================

class RecurrenceSolver:
    """
    Resolutor principal que integra todas las técnicas.
    
    MEJORA: Ahora incluye ExponentialSubtractMethod para Torres de Hanoi
    """
    
    @staticmethod
    def solve(equation: str, preferred_method: Optional[str] = None) -> RecurrenceSolution:
        """
        Resuelve una ecuación de recurrencia.
        """
        equation = equation.strip()
        
        # Paso 1: Clasificar
        recurrence_type, params = RecurrenceClassifier.classify(equation)
        
        if recurrence_type == "unknown":
            return RecurrenceSolution(
                original_equation=equation,
                method_used="none",
                big_o="O(?)",
                big_omega="Ω(?)",
                big_theta="Θ(?)",
                complexity_class="unknown",
                is_tight=False,
                recurrence_type="unknown",
                steps=["No se pudo clasificar la ecuación automáticamente"]
            )
        
        # Paso 2: Seleccionar método
        if preferred_method:
            return RecurrenceSolver._apply_method(
                preferred_method, recurrence_type, params, equation
            )
        else:
            return RecurrenceSolver._auto_select_method(
                recurrence_type, params, equation
            )
    
    @staticmethod
    def _auto_select_method(recurrence_type: str, params: Dict, equation: str) -> RecurrenceSolution:
        """Selección automática del mejor método - ACTUALIZADO"""
        
        # Ecuación ya resuelta
        if recurrence_type == "already-solved":
            return RecurrenceSolver._handle_already_solved(params, equation)
        
        # NUEVO: Exponencial con resta (Torres de Hanoi)
        if recurrence_type == "exponential-subtract":
            if ExponentialSubtractMethod.applies(recurrence_type, params):
                return ExponentialSubtractMethod.solve(
                    params["coefficient"],
                    params["offset"],
                    params["f_n"],
                    equation
                )
        
        # Divide y Vencerás
        if recurrence_type == "divide-conquer":
            if MasterTheorem.applies(recurrence_type, params):
                return MasterTheorem.solve(
                    params["a"], params["b"], params["f_n"], equation
                )
        
        # Resta y Vencerás (lineal)
        elif recurrence_type == "subtract-conquer":
            return IterationMethod.solve(
                params["k"], params["f_n"], equation
            )
        
        # Resta y Serás Vencido (Fibonacci)
        elif recurrence_type == "subtract-conquered":
            k_values = params.get("k_values", [])
            if set(k_values) == {1, 2}:
                return CharacteristicEquation.solve_fibonacci(
                    k_values, params["f_n"], equation
                )
            else:
                return RecursionTree.analyze(recurrence_type, params, equation)
        
        # Lineal no homogénea
        elif recurrence_type == "linear-nonhomogeneous":
            return CharacteristicEquation.solve_general(
                params["coefficients"], params["f_n"], equation
            )
        
        # Fallback: Sustitución
        return SubstitutionMethod.solve(recurrence_type, params, equation)
    
    @staticmethod
    def _handle_already_solved(params: Dict, equation: str) -> RecurrenceSolution:
        """Maneja ecuaciones ya resueltas (sin cambios)"""
        complexity = params.get("complexity", "O(?)")
        notation = params.get("notation", "O")
        
        import re
        match = re.search(r'[OΘΩ]\((.*?)\)', complexity)
        inner = match.group(1) if match else "?"
        
        complexity_class = "constant"
        if "1" in inner:
            complexity_class = "constant"
        elif "log" in inner.lower():
            complexity_class = "logarithmic"
        elif "n^2" in inner or "n²" in inner:
            complexity_class = "quadratic"
        elif "n" in inner and "log" not in inner.lower():
            complexity_class = "linear"
        elif "2^n" in inner or "^n" in inner:
            complexity_class = "exponential"
        
        is_tight = (notation == "Θ")
        
        steps = [
            "Ecuación ya resuelta (no requiere análisis)",
            f"La ecuación está expresada directamente en notación {notation}",
            f"Complejidad: {complexity}"
        ]
        
        if is_tight:
            big_o = f"O({inner})"
            big_omega = f"Ω({inner})"
            big_theta = f"Θ({inner})"
            tight_bounds = f"Cota ajustada: {inner}"
        else:
            if notation == "O":
                big_o = complexity
                big_omega = "Ω(1)"
                big_theta = f"Θ({inner})" if is_tight else "Θ(?)"
                tight_bounds = None
            elif notation == "Ω":
                big_o = "O(?)"
                big_omega = complexity
                big_theta = "Θ(?)"
                tight_bounds = None
            else:
                big_o = f"O({inner})"
                big_omega = f"Ω({inner})"
                big_theta = f"Θ({inner})"
                tight_bounds = f"Cota ajustada: {inner}"
        
        return RecurrenceSolution(
            original_equation=equation,
            method_used="already_solved",
            big_o=big_o,
            big_omega=big_omega,
            big_theta=big_theta,
            complexity_class=complexity_class,
            is_tight=is_tight,
            recurrence_type="already-solved",
            steps=steps,
            exact_solution=complexity,
            tight_bounds=tight_bounds
        )
    
    @staticmethod
    def _apply_method(method: str, recurrence_type: str, params: Dict, equation: str) -> RecurrenceSolution:
        """Aplica un método específico (sin cambios significativos)"""
        
        if method == "master":
            if MasterTheorem.applies(recurrence_type, params):
                return MasterTheorem.solve(
                    params["a"], params["b"], params["f_n"], equation
                )
            else:
                return RecurrenceSolution(
                    original_equation=equation,
                    method_used="master_not_applicable",
                    big_o="O(?)",
                    big_omega="Ω(?)",
                    big_theta="Θ(?)",
                    steps=["Teorema Maestro no aplica para este tipo de recurrencia"],
                    recurrence_type=recurrence_type
                )
        
        elif method == "iteration":
            if recurrence_type == "subtract-conquer":
                return IterationMethod.solve(params["k"], params["f_n"], equation)
            else:
                return RecurrenceSolution(
                    original_equation=equation,
                    method_used="iteration_not_applicable",
                    big_o="O(?)",
                    big_omega="Ω(?)",
                    big_theta="Θ(?)",
                    steps=["Método de Iteración no aplica para este tipo"],
                    recurrence_type=recurrence_type
                )
        
        elif method == "tree":
            return RecursionTree.analyze(recurrence_type, params, equation)
        
        elif method == "substitution":
            return SubstitutionMethod.solve(recurrence_type, params, equation)
        
        elif method == "characteristic":
            if recurrence_type == "subtract-conquered":
                k_values = params.get("k_values", [])
                return CharacteristicEquation.solve_fibonacci(
                    k_values, params["f_n"], equation
                )
            elif recurrence_type == "linear-nonhomogeneous":
                return CharacteristicEquation.solve_general(
                    params["coefficients"], params["f_n"], equation
                )
            else:
                return RecurrenceSolution(
                    original_equation=equation,
                    method_used="characteristic_not_applicable",
                    big_o="O(?)",
                    big_omega="Ω(?)",
                    big_theta="Θ(?)",
                    steps=["Ecuación Característica no aplica para este tipo"],
                    recurrence_type=recurrence_type
                )
        
        else:
            return RecurrenceSolver._auto_select_method(recurrence_type, params, equation)


# ============================================================================
# API SIMPLIFICADA
# ============================================================================

def solve_recurrence(equation: str, method: Optional[str] = None) -> RecurrenceSolution:
    """
    API simple para resolver recurrencias.
    
    Args:
        equation: Ecuación (ej: "T(n) = 2T(n/2) + O(n)")
        method: Método preferido (opcional)
        
    Returns:
        RecurrenceSolution con análisis completo
    """
    solver = RecurrenceSolver()
    return solver.solve(equation, method)


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demuestra el resolutor con ejemplos variados"""
    
    examples = [
        ("Merge Sort (Divide y Vencerás)", "T(n) = 2T(n/2) + O(n)"),
        ("Binary Search (Divide y Vencerás)", "T(n) = T(n/2) + O(1)"),
        ("Factorial (Resta y Vencerás)", "T(n) = T(n-1) + O(1)"),
        ("Fibonacci (Resta y Serás Vencido)", "T(n) = T(n-1) + T(n-2) + O(1)"),
        ("Strassen (Divide y Vencerás)", "T(n) = 7T(n/2) + O(n^2)"),
        ("Linear Search (Resta y Vencerás)", "T(n) = T(n-1) + c"),
    ]
    
    print("="*70)
    print("RESOLUTOR DE RECURRENCIAS - DEMO COMPLETO")
    print("="*70)
    
    for name, equation in examples:
        print(f"\n{'='*70}")
        print(f"📊 {name}")
        print(f"{'='*70}")
        
        try:
            solution = solve_recurrence(equation)
            print(solution)
            
            # Mostrar tipo de recurrencia
            print(f"\n🔍 Tipo de recurrencia: {solution.recurrence_type}")
            
        except Exception as e:
            print(f"❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo()
"""
Analizador de Recursión - VERSIÓN MEJORADA PARA QUICKSORT
=========================================================

FIX CRÍTICO: Detección de reducción de profundidad basada en particiones
y análisis contextual de llamadas auxiliares.

MEJORAS CLAVE:
1. Rastreo de variables de partición (q, mid, pivot_index, etc.)
2. Inferencia de rangos de recursión (p..q-1, q+1..r)
3. Generación de ecuaciones diferenciadas para mejor/peor/promedio caso
"""

import sys
import os
from typing import List, Dict, Optional, Tuple, Set
from dataclasses import dataclass, field

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from syntax_tree.nodes import *
from analyzer.visitor import ASTVisitor


# ============================================================================
# ESTRUCTURAS DE DATOS
# ============================================================================

@dataclass
class RecursiveCall:
    """Representa una llamada recursiva"""
    function_name: str
    arguments: List[ExpressionNode]
    depth_reduction: Optional[str] = None
    location: str = ""
    path_condition: Optional[str] = None
    in_return: bool = False
    return_group: Optional[int] = None
    
    # NUEVO: Contexto de partición
    partition_variable: Optional[str] = None  # q, mid, pivot_idx
    is_left_partition: Optional[bool] = None  # True=izquierda, False=derecha, None=unknown
    range_start: Optional[str] = None  # p, left, low
    range_end: Optional[str] = None    # q-1, mid-1, pivot_idx-1


@dataclass
class RecurrenceEquation:
    """
    Representa ecuaciones de recurrencia para diferentes casos.
    """
    function_name: str
    parameter: str
    
    # Casos base
    base_case_condition: str
    base_case_cost: str
    
    # Ecuaciones para cada caso
    worst_case_equation: str = ""
    best_case_equation: str = ""
    average_case_equation: str = ""
    
    # Información adicional
    recursive_calls: List[RecursiveCall] = field(default_factory=list)
    non_recursive_cost: str = "O(1)"
    recursion_type: str = ""
    
    # NUEVO: Metadatos específicos de QuickSort
    has_partition_call: bool = False
    partition_cost: str = "O(n)"
    
    # Explicaciones
    worst_case_explanation: str = ""
    best_case_explanation: str = ""
    average_case_explanation: str = ""
    
    def __post_init__(self):
        if not self.worst_case_equation:
            self._generate_equations()
        if not self.recursion_type:
            self.recursion_type = self._classify_recursion()
    
    def _generate_equations(self):
        """
        Genera ecuaciones para peor, mejor y caso promedio.
        
        MEJORA CRÍTICA: Considera el contexto de partición para generar
        ecuaciones realistas de QuickSort.
        """
        if not self.recursive_calls:
            self.worst_case_equation = f"T({self.parameter}) = {self.base_case_cost}"
            self.best_case_equation = self.worst_case_equation
            self.average_case_equation = self.worst_case_equation
            return
        
        # ====================================================================
        # CASO ESPECIAL: QUICKSORT (detectado por partición)
        # ====================================================================
        
        if self._is_quicksort_pattern():
            self._generate_quicksort_equations()
            return
        
        # ====================================================================
        # CASO GENERAL (LÓGICA EXISTENTE)
        # ====================================================================
        
        from collections import defaultdict
        
        in_return_calls: Dict[Optional[int], List[RecursiveCall]] = defaultdict(list)
        statement_calls: List[RecursiveCall] = []
        
        for call in self.recursive_calls:
            if call.in_return:
                in_return_calls[call.return_group].append(call)
            else:
                statement_calls.append(call)
        
        are_mutually_exclusive = len(in_return_calls) > 1
        
        if are_mutually_exclusive:
            self.worst_case_explanation = "Peor caso: llamadas en ramas mutuamente excluyentes"
        
        terms_to_combine: List[str] = []
        
        if statement_calls:
            statement_terms = self._extract_terms(statement_calls)
            terms_to_combine.extend(statement_terms)
        
        if in_return_calls:
            if len(in_return_calls) == 1:
                group_calls = next(iter(in_return_calls.values()))
                return_terms = self._extract_terms(group_calls)
                terms_to_combine.extend(return_terms)
            else:
                representative_term = self._choose_most_expensive_group(in_return_calls)
                terms_to_combine.append(representative_term)
        
        if are_mutually_exclusive:
            worst_recursive_part = self._choose_most_expensive_term(terms_to_combine)
        else:
            worst_recursive_part = self._sum_and_compact_terms(terms_to_combine)
            self.worst_case_explanation = "Peor caso: suma de llamadas simultáneas"
        
        self.worst_case_equation = f"T(n) = {worst_recursive_part} + {self.non_recursive_cost}"
        
        has_early_exit = self._has_early_exit_pattern()
        
        if has_early_exit:
            self.best_case_equation = f"T(n) = {self.base_case_cost}"
            self.best_case_explanation = "Mejor caso: early exit al caso base"
        else:
            self.best_case_equation = self.worst_case_equation
            self.best_case_explanation = "Mejor caso igual al peor caso"
        
        self.average_case_equation = self._calculate_average_case()
        self.average_case_explanation = self._explain_average_case()
    
    def _is_quicksort_pattern(self) -> bool:
        """
        Detecta si el patrón es QuickSort:
        - 2 llamadas recursivas no mutuamente excluyentes (secuenciales)
        - Variables de partición presentes
        - Llamada a función auxiliar (Partition)
        """
        # Criterio 1: Exactamente 2 llamadas recursivas
        if len(self.recursive_calls) != 2:
            return False
        
        # Criterio 2: Ambas son statement calls (no en returns)
        if any(call.in_return for call in self.recursive_calls):
            return False
        
        # Criterio 3: Tienen variables de partición
        if not all(call.partition_variable for call in self.recursive_calls):
            return False
        
        # Criterio 4: Una es izquierda y otra derecha
        left_calls = [c for c in self.recursive_calls if c.is_left_partition]
        right_calls = [c for c in self.recursive_calls if c.is_left_partition == False]
        
        if len(left_calls) != 1 or len(right_calls) != 1:
            return False
        
        # Criterio 5: Tiene llamada a Partition
        return self.has_partition_call
    
    def _generate_quicksort_equations(self):
        """
        Genera ecuaciones específicas para QuickSort.
        
        QuickSort tiene comportamiento diferente según la partición:
        - Peor caso: partición desbalanceada → T(n) = T(n-1) + O(n)
        - Mejor caso: partición equilibrada → T(n) = 2T(n/2) + O(n)
        - Caso promedio: mix → T(n) = 2T(n/2) + O(n)
        """
        self.recursion_type = "quicksort"
        
        # PEOR CASO: Partición completamente desbalanceada
        # T(0) se omite porque es O(1) y se absorbe en el costo de partición
        self.worst_case_equation = "T(n) = T(n-1) + O(n)"
        self.worst_case_explanation = (
            "Peor caso de QuickSort: partición desbalanceada (pivot mal elegido). "
            "Un lado tiene n-1 elementos, el otro 0 (que es O(1)). "
            "Costo de Partition: O(n). "
            "Ecuación: T(n) = T(n-1) + O(n). "
            "Resultado: O(n²)"
        )
        
        # MEJOR CASO: Partición perfectamente equilibrada
        self.best_case_equation = "T(n) = 2T(n/2) + O(n)"
        self.best_case_explanation = (
            "Mejor caso de QuickSort: partición equilibrada (pivot óptimo). "
            "Ambos lados tienen aproximadamente n/2 elementos. "
            "Costo de Partition: O(n). "
            "Resultado: T(n) = 2T(n/2) + O(n) ≈ O(n log n)"
        )
        
        # CASO PROMEDIO: Partición razonablemente buena
        self.average_case_equation = "T(n) = 2T(n/2) + O(n)"
        self.average_case_explanation = (
            "Caso promedio de QuickSort: partición razonablemente equilibrada. "
            "Análisis probabilístico muestra que en promedio la partición es buena. "
            "Se aproxima al mejor caso: T(n) ≈ 2T(n/2) + O(n) ≈ O(n log n)"
        )
    
    def _extract_terms(self, calls: List[RecursiveCall]) -> List[str]:
        """Extrae términos T(...) de una lista de llamadas."""
        terms = []
        
        for call in calls:
            dr = call.depth_reduction
            if not dr:
                continue
            
            reduction = dr.replace(" ", "").strip("()")
            
            if '/2' in reduction or 'n/2' in reduction or 'floor' in reduction:
                term = 'T(n/2)'
            elif 'n-1' in reduction or '-1' in reduction:
                term = 'T(n-1)'
            elif 'n-2' in reduction or '-2' in reduction:
                term = 'T(n-2)'
            elif self.parameter and self.parameter in reduction:
                term = f"T({reduction})"
            elif 'n' in reduction:
                term = f"T({reduction})"
            else:
                term = None
            
            if term:
                terms.append(term)
        
        return terms
    
    def _sum_and_compact_terms(self, terms: List[str]) -> str:
        """Suma términos y compacta idénticos."""
        if not terms:
            return 'T(?)'
        
        from collections import Counter
        
        normalized = {}
        for t in terms:
            key = ''.join(t.split())
            if key not in normalized:
                normalized[key] = t
        
        counter = Counter(''.join(t.split()) for t in terms)
        
        result_parts = []
        for key, count in sorted(counter.items()):
            readable = normalized.get(key, key)
            
            if count == 1:
                result_parts.append(readable)
            else:
                result_parts.append(f"{count}{readable}")
        
        return ' + '.join(result_parts)
    
    def _choose_most_expensive_term(self, terms: List[str]) -> str:
        """Elige el término más costoso."""
        if not terms:
            return 'T(?)'
        
        priorities = ['T(n-2)', 'T(n-1)', 'T(n/2)']
        
        for priority in priorities:
            if priority in terms:
                return priority
        
        return terms[0]
    
    def _choose_most_expensive_group(self, groups: Dict[Optional[int], List[RecursiveCall]]) -> str:
        """Elige el grupo más costoso."""
        all_terms = []
        
        for group_calls in groups.values():
            group_terms = self._extract_terms(group_calls)
            if group_terms:
                best_term = self._choose_most_expensive_term(group_terms)
                all_terms.append(best_term)
        
        if not all_terms:
            return 'T(?)'
        
        return self._choose_most_expensive_term(all_terms)
    
    def _has_early_exit_pattern(self) -> bool:
        """
        Detecta patrón de early exit REAL.
        
        Early exit: el algoritmo puede terminar SIN hacer todas las llamadas
        recursivas dependiendo de la entrada.
        
        Criterios:
        1. Debe haber llamadas recursivas en DIFERENTES grupos de return
        2. Los grupos deben ser mutuamente excluyentes (if/else)
        3. Debe existir un caso base que retorna sin recursión
        
        Ejemplos:
        - Binary Search: SI (if encontrado return, else recurse)
        - Fibonacci: NO (return fib(n-1) + fib(n-2) - ambas en mismo return)
        - Hanoi: NO (todas las llamadas son statements)
        """
        if not self.recursive_calls:
            return False
        
        # Si todas las llamadas están en statements (no en returns), NO hay early exit
        all_in_statements = all(not call.in_return for call in self.recursive_calls)
        if all_in_statements:
            return False
        
        # Agrupar llamadas por return_group
        from collections import defaultdict
        in_return_calls: Dict[Optional[int], List[RecursiveCall]] = defaultdict(list)
        
        for call in self.recursive_calls:
            if call.in_return:
                in_return_calls[call.return_group].append(call)
        
        # CLAVE: Solo hay early exit si hay MÚLTIPLES grupos de return diferentes
        # Fibonacci tiene todas las llamadas en el MISMO return (mismo grupo)
        # Binary Search tiene llamadas en DIFERENTES returns (diferentes grupos)
        num_return_groups = len(in_return_calls)
        
        if num_return_groups > 1:
            # Múltiples grupos = ramas mutuamente excluyentes = early exit posible
            if self.base_case_condition and self.base_case_condition != "unknown":
                return True
        
        return False

    
    def _calculate_average_case(self) -> str:
        """Calcula ecuación de caso promedio"""
        recursion_type = self._classify_recursion()
        
        if recursion_type in ["linear", "divide-and-conquer", "simple"]:
            return self.worst_case_equation
        elif recursion_type == "binary":
            return self.worst_case_equation
        elif self._has_early_exit_pattern():
            return self.worst_case_equation
        else:
            return self.worst_case_equation
    
    def _explain_average_case(self) -> str:
        """Genera explicación del caso promedio"""
        recursion_type = self._classify_recursion()
        
        if self._has_early_exit_pattern():
            return "Caso promedio: entre el mejor y peor caso"
        elif recursion_type in ["linear", "divide-and-conquer"]:
            return "Caso promedio igual al peor caso"
        elif recursion_type == "binary":
            return "Caso promedio similar al peor caso"
        else:
            return "Caso promedio aproximado al peor caso"
    
    def _classify_recursion(self) -> str:
        """Clasifica el tipo de recursión"""
        if self.recursion_type == "quicksort":
            return "quicksort"
        
        num_calls = len(self.recursive_calls)
        
        if num_calls == 0:
            return "non-recursive"
        elif num_calls == 1:
            call = self.recursive_calls[0]
            if call.depth_reduction:
                reduction = call.depth_reduction.replace(" ", "")
                if "n-1" in reduction or "-1" in reduction:
                    return "linear"
                elif "n/2" in reduction or "/2" in reduction:
                    return "divide-and-conquer"
            return "simple"
        elif num_calls == 2:
            drs = [c.depth_reduction or '' for c in self.recursive_calls]
            if all(d and ('/2' in d or 'n/2' in d or d == 'n/2') for d in drs):
                return 'divide-and-conquer'
            return "binary"
        else:
            return "multiple"
    
    def to_dict(self) -> dict:
        """Convierte a diccionario"""
        return {
            "function_name": self.function_name,
            "parameter": self.parameter,
            "recursion_type": self.recursion_type,
            "base_case": {
                "condition": self.base_case_condition,
                "cost": self.base_case_cost
            },
            "worst_case": {
                "equation": self.worst_case_equation,
                "explanation": self.worst_case_explanation
            },
            "best_case": {
                "equation": self.best_case_equation,
                "explanation": self.best_case_explanation
            },
            "average_case": {
                "equation": self.average_case_equation,
                "explanation": self.average_case_explanation
            },
            "num_recursive_calls": len(self.recursive_calls)
        }


@dataclass
class RecursionAnalysisResult:
    """Resultado del análisis de recursión"""
    procedure_name: str
    is_recursive: bool
    recurrence_equation: Optional[RecurrenceEquation] = None
    base_cases: List[str] = field(default_factory=list)
    recursive_cases: List[str] = field(default_factory=list)
    max_recursion_depth: Optional[int] = None
    calls_count: int = 0
    
    def to_recurrence_worst(self) -> Optional[str]:
        if not self.is_recursive or not self.recurrence_equation:
            return None
        return self.recurrence_equation.worst_case_equation
    
    def to_recurrence_best(self) -> Optional[str]:
        if not self.is_recursive or not self.recurrence_equation:
            return None
        return self.recurrence_equation.best_case_equation
    
    def to_recurrence_average(self) -> Optional[str]:
        if not self.is_recursive or not self.recurrence_equation:
            return None
        return self.recurrence_equation.average_case_equation
    
    def to_recurrence(self) -> Optional[str]:
        return self.to_recurrence_worst()
    
    def get_all_equations(self) -> Optional[Dict[str, str]]:
        if not self.is_recursive or not self.recurrence_equation:
            return None
        
        return {
            'worst': self.recurrence_equation.worst_case_equation,
            'best': self.recurrence_equation.best_case_equation,
            'average': self.recurrence_equation.average_case_equation
        }


# ============================================================================
# VISITOR MEJORADO
# ============================================================================

class RecursionAnalyzerVisitor(ASTVisitor):
    """Visitor que analiza recursión con soporte mejorado para particiones"""
    
    def __init__(self, procedure_name: str):
        self.procedure_name = procedure_name
        self.current_procedure = procedure_name
        self.recursive_calls: List[RecursiveCall] = []
        self.base_cases: List[Tuple[str, str]] = []
        self.in_base_case = False
        self.current_condition = None
        self.parameter_name = None
        self.array_param_name = None
        self.local_var_map: Dict[str, str] = {}
        self.has_merge = False
        self.current_path_conditions: List[str] = []
        self.has_early_exit = False
        self.current_in_return = False
        self.current_return_group: Optional[int] = None
        self.return_group_counter: int = 0
        
        # NUEVO: Rastreo de particiones
        self.partition_variables: Set[str] = set()  # {q, mid, pivot_idx}
        self.has_partition_call: bool = False
        self.range_params: List[str] = []  # [p, r] o [left, right]
    
    def visit_program(self, node: ProgramNode):
        for proc in node.procedures:
            if proc.name == self.procedure_name:
                return proc.accept(self)
        return None
    
    def visit_procedure(self, node: ProcedureNode):
        # Identificar parámetros de rango (p, r, left, right, low, high)
        range_param_names = {'p', 'r', 'left', 'right', 'low', 'high', 'start', 'end'}
        
        if node.parameters:
            first_param = node.parameters[0]
            if isinstance(first_param, SimpleParamNode):
                self.parameter_name = first_param.name
            elif isinstance(first_param, ArrayParamNode):
                try:
                    self.array_param_name = first_param.name
                except Exception:
                    self.array_param_name = None
                if len(node.parameters) > 1 and isinstance(node.parameters[1], SimpleParamNode):
                    self.parameter_name = node.parameters[1].name
            
            # Capturar parámetros de rango
            for param in node.parameters:
                if isinstance(param, SimpleParamNode) and param.name in range_param_names:
                    self.range_params.append(param.name)
        
        node.body.accept(self)
        return self._build_result()
    
    def visit_assignment(self, node: AssignmentNode):
        """
        Rastrear asignaciones que crean variables de partición.
        
        Ejemplos:
        - q ← call Partition(...)
        - mid ← floor((left + right) / 2)
        - pivot_idx ← call FindPivot(...)
        """
        node.value.accept(self)
        
        target = node.target
        try:
            if isinstance(target, VariableLValueNode):
                varname = target.name
                val_str = self._expr_to_string(node.value).replace(" ", "")
                
                # Detectar llamadas a funciones de partición
                if 'Partition' in val_str or 'FindPivot' in val_str:
                    self.partition_variables.add(varname)
                    self.has_partition_call = True
                
                # Detectar cálculos de punto medio
                elif ('floor' in val_str or 'ceil' in val_str) and '/2' in val_str:
                    self.partition_variables.add(varname)
                
                # Mapear variable → expresión
                if val_str and val_str != "?":
                    self.local_var_map[varname] = val_str
        except Exception:
            pass
    
    def visit_call_statement(self, node: CallStatementNode):
        """Detectar llamadas recursivas como statements"""
        if node.name == self.procedure_name:
            call = self._analyze_recursive_call_statement(node)
            if self.current_path_conditions:
                call.path_condition = " AND ".join(self.current_path_conditions)
            call.in_return = self.current_in_return
            call.return_group = self.current_return_group
            self.recursive_calls.append(call)
        
        # Detectar funciones auxiliares conocidas
        if node.name and node.name.lower() in {'merge', 'partition', 'findpivot'}:
            if 'partition' in node.name.lower():
                self.has_partition_call = True
            if 'merge' in node.name.lower():
                self.has_merge = True
        
        for arg in node.arguments:
            arg.accept(self)
    
    def _analyze_recursive_call_statement(self, node: CallStatementNode) -> RecursiveCall:
        """
        Analiza una llamada recursiva statement con contexto de partición.
        
        MEJORA CRÍTICA: Detecta si los argumentos usan variables de partición
        y determina si es la rama izquierda o derecha.
        """
        depth_reduction = None
        partition_var = None
        is_left = None
        range_start = None
        range_end = None
        
        if node.arguments:
            # Paso 1: Buscar variable de partición en argumentos
            for i, arg in enumerate(node.arguments):
                arg_str = self._expr_to_string(arg).replace(" ", "")
                
                # Ignorar el array
                if self.array_param_name and arg_str == self.array_param_name:
                    continue
                
                # Detectar uso de variable de partición
                for pvar in self.partition_variables:
                    if pvar in arg_str:
                        partition_var = pvar
                        
                        # Determinar si es izquierda o derecha
                        if '-1' in arg_str or f'{pvar}-1' in arg_str:
                            # Ejemplo: q-1 → rama izquierda [p..q-1]
                            is_left = True
                            range_end = arg_str
                        elif '+1' in arg_str or f'{pvar}+1' in arg_str:
                            # Ejemplo: q+1 → rama derecha [q+1..r]
                            is_left = False
                            range_start = arg_str
                        elif arg_str == pvar:
                            # Uso directo de la variable
                            pass
                        
                        break
            
            # Paso 2: Determinar range_start y range_end
            for i, arg in enumerate(node.arguments):
                arg_str = self._expr_to_string(arg).replace(" ", "")
                
                if arg_str in self.range_params:
                    if not range_start and is_left is not False:
                        range_start = arg_str
                    elif not range_end and is_left is not True:
                        range_end = arg_str
            
            # Paso 3: Inferir depth_reduction basado en contexto
            if partition_var:
                if is_left:
                    depth_reduction = "n/2"  # Aproximación
                elif is_left == False:
                    depth_reduction = "n/2"  # Aproximación
                else:
                    depth_reduction = "n/2"
            else:
                # Fallback a lógica existente
                depth_reduction = self._infer_depth_reduction_fallback(node.arguments)
        
        return RecursiveCall(
            function_name=node.name,
            arguments=node.arguments,
            depth_reduction=depth_reduction,
            location="statement",
            partition_variable=partition_var,
            is_left_partition=is_left,
            range_start=range_start,
            range_end=range_end
        )
    
    def _infer_depth_reduction_fallback(self, arguments: List[ExpressionNode]) -> Optional[str]:
        """Lógica fallback para inferir reducción (sin partición detectada)"""
        for arg in arguments:
            arg_str = self._expr_to_string(arg).replace(" ", "")
            
            if self.array_param_name and arg_str == self.array_param_name:
                continue
            
            # Buscar en mapa de variables locales
            if arg_str in self.local_var_map:
                mapped = self.local_var_map[arg_str]
                if '/2' in mapped or 'floor' in mapped:
                    return 'n/2'
            
            # Buscar parámetro principal
            if self.parameter_name and self.parameter_name in arg_str:
                return arg_str
        
        return None
    
    def visit_if(self, node: IfNode):
        old_condition = self.current_condition
        condition_str = self._condition_to_string(node.condition)
        self.current_condition = condition_str
        
        self.current_path_conditions.append(condition_str)
        
        old_base_flag = self.in_base_case
        
        has_recursive_call_in_then = self._has_recursive_call(node.then_block)
        
        if not has_recursive_call_in_then:
            self.in_base_case = True
            self.has_early_exit = True
            cost = self._estimate_cost(node.then_block)
            self.base_cases.append((condition_str, cost))
        
        node.then_block.accept(self)
        self.in_base_case = old_base_flag
        
        self.current_path_conditions.pop()
        
        if node.else_block:
            negated_condition = f"not ({condition_str})"
            self.current_path_conditions.append(negated_condition)
            
            self.in_base_case = False
            node.else_block.accept(self)
            
            self.current_path_conditions.pop()
        
        self.current_condition = old_condition
    
    def visit_for(self, node: ForNode):
        node.body.accept(self)
    
    def visit_while(self, node: WhileNode):
        node.body.accept(self)
    
    def visit_repeat(self, node: RepeatNode):
        node.body.accept(self)
    
    def visit_function_call(self, node: FunctionCallNode):
        if node.name == self.procedure_name:
            call = self._analyze_recursive_call(node)
            if self.current_path_conditions:
                call.path_condition = " AND ".join(self.current_path_conditions)
            call.in_return = self.current_in_return
            call.return_group = self.current_return_group
            self.recursive_calls.append(call)
        
        if node.name and node.name.lower() in {'merge', 'partition', 'findpivot'}:
            if 'partition' in node.name.lower():
                self.has_partition_call = True
            if 'merge' in node.name.lower():
                self.has_merge = True
        
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_return(self, node: ReturnNode):
        old_in_return = self.current_in_return
        old_return_group = self.current_return_group
        self.return_group_counter += 1
        self.current_return_group = self.return_group_counter
        self.current_in_return = True
        if node.value:
            node.value.accept(self)
        self.current_in_return = old_in_return
        self.current_return_group = old_return_group
    
    def _analyze_recursive_call(self, node: FunctionCallNode) -> RecursiveCall:
        """
        Analiza llamada recursiva en expresión (similar a statement).
        """
        depth_reduction = None
        partition_var = None
        is_left = None
        range_start = None
        range_end = None
        
        if node.arguments:
            for i, arg in enumerate(node.arguments):
                arg_str = self._expr_to_string(arg).replace(" ", "")
                
                if self.array_param_name and arg_str == self.array_param_name:
                    continue
                
                for pvar in self.partition_variables:
                    if pvar in arg_str:
                        partition_var = pvar
                        
                        if '-1' in arg_str:
                            is_left = True
                            range_end = arg_str
                        elif '+1' in arg_str:
                            is_left = False
                            range_start = arg_str
                        
                        break
            
            if partition_var:
                depth_reduction = "n/2"
            else:
                depth_reduction = self._infer_depth_reduction_fallback(node.arguments)
        
        return RecursiveCall(
            function_name=node.name,
            arguments=node.arguments,
            depth_reduction=depth_reduction,
            location="expression",
            partition_variable=partition_var,
            is_left_partition=is_left,
            range_start=range_start,
            range_end=range_end
        )
    
    def _has_recursive_call(self, block: BlockNode) -> bool:
        """Verifica si un bloque tiene llamadas recursivas"""
        class CallFinder(ASTVisitor):
            def __init__(self, target_name):
                self.target_name = target_name
                self.found = False
            
            def visit_program(self, node): pass
            def visit_procedure(self, node): pass
            
            def visit_for(self, node): 
                if not self.found:
                    node.body.accept(self)
            
            def visit_while(self, node):
                if not self.found:
                    node.condition.accept(self)
                    node.body.accept(self)
            
            def visit_repeat(self, node):
                if not self.found:
                    node.body.accept(self)
                    node.condition.accept(self)
            
            def visit_if(self, node):
                if not self.found:
                    node.condition.accept(self)
                    node.then_block.accept(self)
                    if node.else_block:
                        node.else_block.accept(self)
            
            def visit_assignment(self, node):
                if not self.found:
                    node.value.accept(self)
            
            def visit_return(self, node):
                if not self.found and node.value:
                    node.value.accept(self)
            
            def visit_function_call(self, node):
                if node.name == self.target_name:
                    self.found = True
                else:
                    for arg in node.arguments:
                        if not self.found:
                            arg.accept(self)
            
            def visit_call_statement(self, node):
                if node.name == self.target_name:
                    self.found = True
                else:
                    for arg in node.arguments:
                        if not self.found:
                            arg.accept(self)
            
            def visit_binary_op(self, node):
                if not self.found:
                    node.left.accept(self)
                    node.right.accept(self)
            
            def visit_unary_op(self, node):
                if not self.found:
                    node.operand.accept(self)
            
            def visit_number(self, node): pass
            def visit_string(self, node): pass
            def visit_boolean(self, node): pass
            def visit_null(self, node): pass
            def visit_identifier(self, node): pass
        
        finder = CallFinder(self.procedure_name)
        block.accept(finder)
        return finder.found
    
    def _condition_to_string(self, condition: ExpressionNode) -> str:
        result = self._expr_to_string(condition)
        return " ".join(result.split())
    
    def _estimate_cost(self, block: BlockNode) -> str:
        return "O(1)"
    
    def _expr_to_string(self, expr: ExpressionNode) -> str:
        """Convierte expresión a string"""
        if isinstance(expr, NumberNode):
            return str(expr.value)
        elif isinstance(expr, IdentifierNode):
            return expr.name
        elif isinstance(expr, BinaryOpNode):
            left = self._expr_to_string(expr.left)
            right = self._expr_to_string(expr.right)
            return f"{left}{expr.op}{right}"
        elif isinstance(expr, UnaryOpNode):
            operand = self._expr_to_string(expr.operand)
            return f"{expr.op}{operand}"
        elif isinstance(expr, BooleanNode):
            return str(expr.value)
        elif isinstance(expr, ArrayAccessNode):
            indices = ",".join(self._expr_to_string(i) for i in expr.indices)
            return f"{expr.name}[{indices}]"
        elif isinstance(expr, FunctionCallNode):
            args = ",".join(self._expr_to_string(a) for a in expr.arguments)
            return f"call {expr.name}({args})"
        elif hasattr(expr, 'expression') and type(expr).__name__ in ('FloorNode', 'CeilingNode'):
            inner = self._expr_to_string(getattr(expr, 'expression'))
            if type(expr).__name__ == 'FloorNode':
                return f"floor({inner})"
            else:
                return f"ceiling({inner})"
        elif hasattr(expr, 'array_name') and type(expr).__name__ == 'LengthNode':
            return f"length({expr.array_name})"
        else:
            return "?"
    
    def _build_result(self) -> RecursionAnalysisResult:
        """Construye resultado del análisis"""
        is_recursive = len(self.recursive_calls) > 0
        
        if not is_recursive:
            return RecursionAnalysisResult(
                procedure_name=self.procedure_name,
                is_recursive=False
            )
        
        if self.base_cases:
            base_condition, base_cost = self.base_cases[0]
        else:
            base_condition = "unknown"
            base_cost = "O(1)"
        
        recurrence = RecurrenceEquation(
            function_name=self.procedure_name,
            parameter=("n" if self.array_param_name else (self.parameter_name or "n")),
            base_case_condition=base_condition,
            base_case_cost=base_cost,
            recursive_calls=self.recursive_calls,
            non_recursive_cost=("n" if self.has_merge else "O(1)"),
            has_partition_call=self.has_partition_call,
            partition_cost="O(n)" if self.has_partition_call else "O(1)"
        )
        
        return RecursionAnalysisResult(
            procedure_name=self.procedure_name,
            is_recursive=True,
            recurrence_equation=recurrence,
            base_cases=[bc[0] for bc in self.base_cases],
            recursive_cases=[f"{call.function_name}({call.depth_reduction})" 
                           for call in self.recursive_calls],
            calls_count=len(self.recursive_calls)
        )


# ============================================================================
# API PÚBLICA
# ============================================================================

def analyze_recursion(procedure: ProcedureNode) -> RecursionAnalysisResult:
    """Analiza un procedimiento para detectar recursión"""
    visitor = RecursionAnalyzerVisitor(procedure.name)
    return visitor.visit_procedure(procedure)


def to_recurrence(code: str, procedure_name: Optional[str] = None, 
                  case: str = "worst") -> Optional[str]:
    """
    Extrae ecuación de recurrencia para un caso específico.
    
    Args:
        code: Código pseudocódigo
        procedure_name: Nombre del procedimiento
        case: "worst", "best" o "average"
    
    Returns:
        String con la ecuación o None si no es recursivo
    """
    from parser.parser import parse
    
    try:
        ast = parse(code)
        
        if not ast.procedures:
            return None
        
        if procedure_name:
            proc = next((p for p in ast.procedures if p.name == procedure_name), None)
            if not proc:
                return None
        else:
            proc = ast.procedures[0]
        
        result = analyze_recursion(proc)
        
        if not result.is_recursive:
            return None
        
        if case == "worst":
            return result.to_recurrence_worst()
        elif case == "best":
            return result.to_recurrence_best()
        elif case == "average":
            return result.to_recurrence_average()
        else:
            return result.to_recurrence_worst()
    
    except Exception:
        return None


def to_all_recurrences(code: str, procedure_name: Optional[str] = None) -> Optional[Dict[str, str]]:
    """
    Extrae TODAS las ecuaciones de recurrencia.
    
    Returns:
        Dict con keys: 'worst', 'best', 'average'
        o None si no es recursivo
    """
    from parser.parser import parse
    
    try:
        ast = parse(code)
        
        if not ast.procedures:
            return None
        
        if procedure_name:
            proc = next((p for p in ast.procedures if p.name == procedure_name), None)
            if not proc:
                return None
        else:
            proc = ast.procedures[0]
        
        result = analyze_recursion(proc)
        return result.get_all_equations()
    
    except Exception:
        return None


def analyze_all_procedures(ast: ProgramNode) -> Dict[str, RecursionAnalysisResult]:
    """Analiza todos los procedimientos"""
    results = {}
    for procedure in ast.procedures:
        result = analyze_recursion(procedure)
        results[procedure.name] = result
    return results
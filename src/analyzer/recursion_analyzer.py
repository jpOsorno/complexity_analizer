"""
Analizador de Recursión - VERSIÓN CORREGIDA PARA MERGE SORT
===========================================================

FIX CRÍTICO: Manejo correcto de múltiples llamadas recursivas
simultáneas (no mutuamente excluyentes)
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
        
        FIX CRÍTICO: Detecta correctamente si las llamadas son:
        1. Mutuamente excluyentes (en diferentes returns/if-else) → tomar MAX
        2. Simultáneas (secuenciales en el mismo bloque) → SUMAR
        """
        if not self.recursive_calls:
            self.worst_case_equation = f"T({self.parameter}) = {self.base_case_cost}"
            self.best_case_equation = self.worst_case_equation
            self.average_case_equation = self.worst_case_equation
            return
        
        # ====================================================================
        # PASO 1: Clasificar las llamadas recursivas
        # ====================================================================
        
        from collections import defaultdict
        
        # Llamadas dentro de returns
        in_return_calls: Dict[Optional[int], List[RecursiveCall]] = defaultdict(list)
        # Llamadas fuera de returns (statements secuenciales)
        statement_calls: List[RecursiveCall] = []
        
        for call in self.recursive_calls:
            if call.in_return:
                in_return_calls[call.return_group].append(call)
            else:
                statement_calls.append(call)
        
        # ====================================================================
        # PASO 2: Determinar si las llamadas son mutuamente excluyentes
        # ====================================================================
        
        are_mutually_exclusive = False
        
        # Si hay múltiples grupos de return, son mutuamente excluyentes
        if len(in_return_calls) > 1:
            are_mutually_exclusive = True
            self.worst_case_explanation = "Peor caso: llamadas en ramas mutuamente excluyentes"
        
        # Si hay mezcla de return y statements, analizar path_conditions
        elif len(in_return_calls) == 1 and statement_calls:
            # Verificar si las path_conditions son diferentes
            path_conditions = set()
            for call in self.recursive_calls:
                if call.path_condition:
                    path_conditions.add(call.path_condition)
            
            if len(path_conditions) > 1:
                are_mutually_exclusive = True
                self.worst_case_explanation = "Peor caso: llamadas en diferentes condiciones"
        
        # ====================================================================
        # PASO 3: Construir términos recursivos
        # ====================================================================
        
        terms_to_combine: List[str] = []
        
        # Procesar llamadas statement (SIEMPRE SUMABLES)
        if statement_calls:
            statement_terms = self._extract_terms(statement_calls)
            terms_to_combine.extend(statement_terms)
        
        # Procesar llamadas en returns
        if in_return_calls:
            if len(in_return_calls) == 1:
                # Un solo grupo → SUMABLES
                group_calls = next(iter(in_return_calls.values()))
                return_terms = self._extract_terms(group_calls)
                terms_to_combine.extend(return_terms)
            else:
                # Múltiples grupos → MUTUAMENTE EXCLUYENTES
                # Elegir la más costosa
                representative_term = self._choose_most_expensive_group(in_return_calls)
                terms_to_combine.append(representative_term)
        
        # ====================================================================
        # PASO 4: Combinar términos
        # ====================================================================
        
        if are_mutually_exclusive:
            # Tomar el más costoso
            worst_recursive_part = self._choose_most_expensive_term(terms_to_combine)
        else:
            # Sumar y compactar términos idénticos
            worst_recursive_part = self._sum_and_compact_terms(terms_to_combine)
            self.worst_case_explanation = "Peor caso: suma de llamadas simultáneas"
        
        # ====================================================================
        # PASO 5: Construir ecuación final
        # ====================================================================
        
        self.worst_case_equation = f"T(n) = {worst_recursive_part} + {self.non_recursive_cost}"
        
        # ====================================================================
        # PASO 6: Mejor caso y promedio
        # ====================================================================
        
        has_early_exit = self._has_early_exit_pattern()
        
        if has_early_exit:
            self.best_case_equation = f"T(n) = {self.base_case_cost}"
            self.best_case_explanation = "Mejor caso: early exit al caso base"
        else:
            self.best_case_equation = self.worst_case_equation
            self.best_case_explanation = "Mejor caso igual al peor caso"
        
        self.average_case_equation = self._calculate_average_case()
        self.average_case_explanation = self._explain_average_case()
    
    def _extract_terms(self, calls: List[RecursiveCall]) -> List[str]:
        """
        Extrae términos T(...) de una lista de llamadas.
        
        Returns:
            Lista de términos como strings: ['T(n/2)', 'T(n-1)', ...]
        """
        terms = []
        
        for call in calls:
            dr = call.depth_reduction
            if not dr:
                continue
            
            reduction = dr.replace(" ", "").strip("()")
            
            # Normalizar patrones conocidos
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
        """
        Suma términos y compacta idénticos.
        
        Ejemplos:
        - ['T(n/2)', 'T(n/2)'] → '2T(n/2)'
        - ['T(n-1)', 'T(n-2)'] → 'T(n-1) + T(n-2)'
        - ['T(n/2)', 'T(n/2)', 'T(n-1)'] → '2T(n/2) + T(n-1)'
        """
        if not terms:
            return 'T(?)'
        
        from collections import Counter
        
        # Normalizar términos (sin espacios) para contar
        normalized = {}
        for t in terms:
            key = ''.join(t.split())
            if key not in normalized:
                normalized[key] = t
        
        # Contar ocurrencias
        counter = Counter(''.join(t.split()) for t in terms)
        
        # Construir resultado
        result_parts = []
        for key, count in sorted(counter.items()):
            readable = normalized.get(key, key)
            
            if count == 1:
                result_parts.append(readable)
            else:
                result_parts.append(f"{count}{readable}")
        
        return ' + '.join(result_parts)
    
    def _choose_most_expensive_term(self, terms: List[str]) -> str:
        """
        Elige el término más costoso de una lista.
        
        Orden de costo: T(n/2) < T(n-1) < T(n-2) < otros
        """
        if not terms:
            return 'T(?)'
        
        # Prioridad de búsqueda (del más costoso al menos costoso)
        priorities = ['T(n-2)', 'T(n-1)', 'T(n/2)']
        
        for priority in priorities:
            if priority in terms:
                return priority
        
        # Si no hay ninguno conocido, retornar el primero
        return terms[0]
    
    def _choose_most_expensive_group(self, groups: Dict[Optional[int], List[RecursiveCall]]) -> str:
        """
        Elige el grupo más costoso cuando hay múltiples grupos de return.
        """
        all_terms = []
        
        for group_calls in groups.values():
            group_terms = self._extract_terms(group_calls)
            if group_terms:
                # Tomar el más costoso del grupo
                best_term = self._choose_most_expensive_term(group_terms)
                all_terms.append(best_term)
        
        if not all_terms:
            return 'T(?)'
        
        # Elegir el más costoso entre los representantes
        return self._choose_most_expensive_term(all_terms)
    
    def _has_early_exit_pattern(self) -> bool:
        """Detecta patrón de early exit"""
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
            return self.worst_case_equation.replace("T(", "T(")
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
# VISITOR (sin cambios en la lógica de detección)
# ============================================================================

class RecursionAnalyzerVisitor(ASTVisitor):
    """Visitor que analiza recursión"""
    
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
    
    def visit_program(self, node: ProgramNode):
        for proc in node.procedures:
            if proc.name == self.procedure_name:
                return proc.accept(self)
        return None
    
    def visit_procedure(self, node: ProcedureNode):
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
        
        node.body.accept(self)
        return self._build_result()
    
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
        
        if node.name and node.name.lower() == 'merge':
            self.has_merge = True
        
        for arg in node.arguments:
            arg.accept(self)
    
    def visit_call_statement(self, node: CallStatementNode):
        if node.name == self.procedure_name:
            call = self._analyze_recursive_call_statement(node)
            if self.current_path_conditions:
                call.path_condition = " AND ".join(self.current_path_conditions)
            call.in_return = self.current_in_return
            call.return_group = self.current_return_group
            self.recursive_calls.append(call)
        
        if node.name and node.name.lower() == 'merge':
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
    
    def visit_assignment(self, node: AssignmentNode):
        node.value.accept(self)
        
        target = node.target
        try:
            if isinstance(target, VariableLValueNode):
                varname = target.name
                val_str = self._expr_to_string(node.value).replace(" ", "")
                if val_str and val_str != "?":
                    self.local_var_map[varname] = val_str
        except Exception:
            pass
    
    def _has_recursive_call(self, block: BlockNode) -> bool:
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
    
    def _expr_to_string(self, expr: ExpressionNode) -> str:
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
    
    def _analyze_recursive_call(self, node: FunctionCallNode) -> RecursiveCall:
        depth_reduction = None
        
        if node.arguments:
            # ESTRATEGIA MEJORADA (igual que para statements):
            # 1. Buscar argumentos que referencien variables mapeadas (ej. q -> n/2)
            # 2. Si el argumento es una variable mapeada a floor(...)/2, es n/2
            # 3. Caso contrario, usar heurística por parámetros
            
            # Paso 1: Intentar por variables locales mapeadas PRIMERO
            if self.local_var_map:
                for arg in node.arguments:
                    arg_str = self._expr_to_string(arg).replace(" ", "")
                    
                    # Ignorar el array (primer parámetro)
                    if self.array_param_name and arg_str == self.array_param_name:
                        continue
                    
                    # Verificar si el argumento ES una variable mapeada
                    if arg_str in self.local_var_map:
                        mapped = self.local_var_map[arg_str]
                        
                        # Si está mapeada a algo con /2 o floor, es n/2
                        if ('/2' in mapped) or ('floor' in mapped) or \
                           ('(p+r)/2' in mapped) or ('(left+right)/2' in mapped):
                            depth_reduction = 'n/2'
                            break
                    
                    # Verificar si el argumento CONTIENE una variable mapeada
                    for varname, mapped in self.local_var_map.items():
                        if varname in arg_str:
                            if ('/2' in mapped) or ('floor' in mapped) or \
                               ('(p+r)/2' in mapped) or ('(left+right)/2' in mapped):
                                depth_reduction = 'n/2'
                                break
                    
                    if depth_reduction:
                        break
            
            # Paso 2: Buscar por nombre del parámetro principal
            if depth_reduction is None:
                for arg in node.arguments:
                    arg_str = self._expr_to_string(arg).replace(" ", "")
                    
                    if self.array_param_name and arg_str == self.array_param_name:
                        continue
                    
                    if self.parameter_name and self.parameter_name in arg_str:
                        depth_reduction = arg_str
                        break
            
            # Paso 3: Fallback - tomar primer argumento no-array
            if depth_reduction is None:
                for arg in node.arguments:
                    arg_str = self._expr_to_string(arg).replace(" ", "")
                    if self.array_param_name and arg_str == self.array_param_name:
                        continue
                    depth_reduction = arg_str
                    break
        
        return RecursiveCall(
            function_name=node.name,
            arguments=node.arguments,
            depth_reduction=depth_reduction,
            location="expression"
        )
    
    def _analyze_recursive_call_statement(self, node: CallStatementNode) -> RecursiveCall:
        depth_reduction = None
        
        if node.arguments:
            # ESTRATEGIA MEJORADA:
            # 1. Buscar argumentos que referencien variables mapeadas (ej. q -> n/2)
            # 2. Si el argumento es una variable mapeada a floor(...)/2, es n/2
            # 3. Caso contrario, usar heurística por parámetros
            
            # Paso 1: Intentar por variables locales mapeadas PRIMERO
            if self.local_var_map:
                for arg in node.arguments:
                    arg_str = self._expr_to_string(arg).replace(" ", "")
                    
                    # Ignorar el array (primer parámetro)
                    if self.array_param_name and arg_str == self.array_param_name:
                        continue
                    
                    # Verificar si el argumento ES una variable mapeada
                    if arg_str in self.local_var_map:
                        mapped = self.local_var_map[arg_str]
                        
                        # Si está mapeada a algo con /2 o floor, es n/2
                        if ('/2' in mapped) or ('floor' in mapped) or \
                           ('(p+r)/2' in mapped) or ('(left+right)/2' in mapped):
                            depth_reduction = 'n/2'
                            break
                    
                    # Verificar si el argumento CONTIENE una variable mapeada
                    for varname, mapped in self.local_var_map.items():
                        if varname in arg_str:
                            if ('/2' in mapped) or ('floor' in mapped) or \
                               ('(p+r)/2' in mapped) or ('(left+right)/2' in mapped):
                                depth_reduction = 'n/2'
                                break
                    
                    if depth_reduction:
                        break
            
            # Paso 2: Buscar por nombre del parámetro principal
            if depth_reduction is None:
                for arg in node.arguments:
                    arg_str = self._expr_to_string(arg).replace(" ", "")
                    
                    if self.array_param_name and arg_str == self.array_param_name:
                        continue
                    
                    if self.parameter_name and self.parameter_name in arg_str:
                        depth_reduction = arg_str
                        break
            
            # Paso 3: Fallback - tomar primer argumento no-array
            if depth_reduction is None:
                for arg in node.arguments:
                    arg_str = self._expr_to_string(arg).replace(" ", "")
                    if self.array_param_name and arg_str == self.array_param_name:
                        continue
                    depth_reduction = arg_str
                    break
        
        return RecursiveCall(
            function_name=node.name,
            arguments=node.arguments,
            depth_reduction=depth_reduction,
            location="statement"
        )
    
    def _estimate_cost(self, block: BlockNode) -> str:
        return "O(1)"
    
    def _build_result(self) -> RecursionAnalysisResult:
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
            non_recursive_cost=("n" if self.has_merge else "O(1)")
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


# ============================================================================
# DEMO
# ============================================================================

def demo():
    """Demuestra el análisis corregido"""
    from parser.parser import parse
    
    examples = {
        "Merge Sort (FIX CRÍTICO)": """
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
        """,
        
        "Binary Search (mutuamente excluyente)": """
BinarySearch(A[], left, right, x)
begin
    if (left > right) then
    begin
        return -1
    end
    
    mid ← floor((left + right) / 2)
    
    if (A[mid] = x) then
    begin
        return mid
    end
    
    if (A[mid] < x) then
    begin
        return call BinarySearch(A, mid+1, right, x)
    end
    
    return call BinarySearch(A, left, mid-1, x)
end
        """,
        
        "Fibonacci (sumables en mismo return)": """
Fibonacci(n)
begin
    if (n ≤ 1) then
    begin
        return n
    end
    else
    begin
        return call Fibonacci(n-1) + call Fibonacci(n-2)
    end
end
        """
    }
    
    print("="*70)
    print("ANÁLISIS DE RECURSIÓN - FIX MERGE SORT")
    print("="*70)
    
    for name, code in examples.items():
        print(f"\n{'='*70}")
        print(f"📊 {name}")
        print(f"{'='*70}")
        
        try:
            equations = to_all_recurrences(code)
            
            if equations:
                print(f"\n🔴 Peor Caso:")
                print(f"   {equations['worst']}")
                
                print(f"\n🟢 Mejor Caso:")
                print(f"   {equations['best']}")
                
                print(f"\n🟡 Caso Promedio:")
                print(f"   {equations['average']}")
                
                # Verificar corrección para Merge Sort
                if "MergeSort" in name:
                    if "2T(n/2)" in equations['worst']:
                        print(f"\n✅ CORRECTO: Detecta 2T(n/2) + n")
                    else:
                        print(f"\n❌ ERROR: No detecta correctamente las 2 llamadas")
            else:
                print("\n❌ No es recursivo")
        
        except Exception as e:
            print(f"\n❌ ERROR: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    demo()
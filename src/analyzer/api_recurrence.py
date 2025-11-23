from typing import Optional, Dict

from recursion_analyzer import to_recurrence, to_all_recurrences
from recurrence_solver import solve_recurrence


def get_recurrences(code: str, procedure_name: Optional[str] = None) -> Optional[Dict[str, str]]:
    """Retorna las tres ecuaciones de recurrencia (worst,best,average).

    Args:
        code: Código en pseudocódigo (string)
        procedure_name: Nombre del procedimiento a analizar (opcional)

    Returns:
        Dict con keys 'worst','best','average' o None si no es recursivo o hay error.
    """
    return to_all_recurrences(code, procedure_name=procedure_name)


def get_recurrence(code: str, procedure_name: Optional[str] = None, case: str = 'worst') -> Optional[str]:
    """Retorna la ecuación de recurrencia para un caso específico.

    Args:
        code: Código en pseudocódigo
        procedure_name: Nombre del procedimiento (opcional)
        case: 'worst' | 'best' | 'average'
    """
    return to_recurrence(code, procedure_name=procedure_name, case=case)


if __name__ == '__main__':
    # Demo rápido (misma muestra que antes)
    sample = """
QuickSort(A[], p, r)
begin
    if (p < r) then
    begin
        q ← call Partition(A, p, r)
        call QuickSort(A, p, q-1)
        call QuickSort(A, q+1, r)
    end
end

Partition(A[], p, r)
begin
    pivot ← A[r]
    i ← p - 1
    
    for j ← p to r-1 do
    begin
        if (A[j] ≤ pivot) then
        begin
            i ← i + 1
            temp ← A[i]
            A[i] ← A[j]
            A[j] ← temp
        end
    end
    
    return i+1
end
    """

    print('Ejemplo:')
    print(sample)
    eqs = get_recurrences(sample)
    if eqs:
        print('  Worst :', eqs.get('worst'))
        print('  Best  :', eqs.get('best'))
        print('  Avg   :', eqs.get('average'))
    else:
        print('No es recursivo o hubo un error en el análisis.')
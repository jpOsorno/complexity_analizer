"""API pequeña para obtener ecuaciones de recurrencia.

Funciones exportadas:
- `get_recurrences(code, procedure_name=None)` -> Dict con keys 'worst','best','average'
- `get_recurrence(code, procedure_name=None, case='worst')` -> ecuación única

También soporta ejecución como script para pruebas rápidas.
"""

from typing import Optional, Dict

from recursion_analyzer import to_recurrence, to_all_recurrences


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

    print('Ejemplo:')
    eqs = get_recurrences(sample)
    if eqs:
        print('  Worst :', eqs.get('worst'))
        print('  Best  :', eqs.get('best'))
        print('  Avg   :', eqs.get('average'))
    else:
        print('No es recursivo o hubo un error en el análisis.')
import sys
import os
import io

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

sys.path.insert(0, 'src')

from src.parser.parser import parse
from src.analyzer.unified_analyzer import analyze_complexity_unified

# List of iterative examples
examples = [
    'BubbleSort.txt',
    'InsertionSort.txt',
    'LinearSearch.txt',
    'MatrizMultiply.txt',
    'SelectionSort.txt'
]

print('='*70)
print('TESTING ITERATIVE ALGORITHMS')
print('='*70)

results_summary = []

for example in examples:
    filepath = f'examples/iteratives/{example}'
    
    print(f'\n{"="*70}')
    print(f'Testing: {example}')
    print(f'{"="*70}')
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        ast = parse(code)
        results = analyze_complexity_unified(ast)
        
        for name, result in results.items():
            print(f'\nProcedure: {name}')
            print(f'  Worst case:    {result.final_worst}')
            print(f'  Best case:     {result.final_best}')
            print(f'  Average case:  {result.final_average}')
            print(f'  Type:          {result.algorithm_type}')
            
            # Validate expected complexities
            status = '[PASSED]'
            issue = None
            
            # Check for known issues
            if 'BubbleSort' in name:
                # BubbleSort should be O(n^2) worst, but could be optimized to Omega(n) best
                if 'n^2' not in result.final_worst and 'n²' not in result.final_worst:
                    status = '[WARNING]'
                    issue = f'Expected O(n^2) worst case, got {result.final_worst}'
                if 'n^2' in result.final_best or 'n²' in result.final_best:
                    status = '[ISSUE]'
                    issue = 'BubbleSort can be optimized to Omega(n) best case with early termination'
                    
            elif 'InsertionSort' in name:
                # InsertionSort: O(n^2) worst, Omega(n) best
                if 'n^2' not in result.final_worst and 'n²' not in result.final_worst:
                    status = '[WARNING]'
                    issue = f'Expected O(n^2) worst case, got {result.final_worst}'
                if 'n^2' in result.final_best or 'n²' in result.final_best:
                    status = '[WARNING]'
                    issue = f'InsertionSort best case should be Omega(n), got {result.final_best}'
                    
            elif 'LinearSearch' in name:
                # LinearSearch: O(n) worst, Omega(1) best
                if result.final_worst != 'O(n)':
                    status = '[WARNING]'
                    issue = f'Expected O(n) worst case, got {result.final_worst}'
                    
            elif 'Matrix' in name:
                # Matrix multiplication: O(n^3)
                if 'n^3' not in result.final_worst:
                    status = '[WARNING]'
                    issue = f'Expected O(n^3) worst case, got {result.final_worst}'
                    
            elif 'SelectionSort' in name:
                # SelectionSort: always O(n^2)
                if 'n^2' not in result.final_worst and 'n²' not in result.final_worst:
                    status = '[WARNING]'
                    issue = f'Expected O(n^2) worst case, got {result.final_worst}'
            
            print(f'\n  Status: {status}')
            if issue:
                print(f'  Note: {issue}')
            
            results_summary.append({
                'file': example,
                'procedure': name,
                'status': status,
                'issue': issue,
                'worst': result.final_worst,
                'best': result.final_best,
                'average': result.final_average
            })
            
    except Exception as e:
        print(f'\n[FAILED]: {e}')
        import traceback
        traceback.print_exc()
        results_summary.append({
            'file': example,
            'procedure': 'N/A',
            'status': '[FAILED]',
            'issue': str(e),
            'worst': 'N/A',
            'best': 'N/A',
            'average': 'N/A'
        })

# Summary
print(f'\n\n{"="*70}')
print('SUMMARY')
print(f'{"="*70}')

for r in results_summary:
    print(f"\n{r['file']:20} | {r['procedure']:20} | {r['status']}")
    if r['issue']:
        print(f"  └─ {r['issue']}")
    print(f"     Worst: {r['worst']:15} Best: {r['best']:15} Avg: {r['average']}")

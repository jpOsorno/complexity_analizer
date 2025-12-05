"""
Diagnóstico: Verificar qué ejemplos se están cargando
"""

import os

def load_examples_from_folder():
    """Carga ejemplos dinámicamente desde la carpeta examples/."""
    examples = {}
    examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
    
    print(f"Buscando ejemplos en: {examples_dir}")
    print(f"¿Existe el directorio? {os.path.exists(examples_dir)}")
    
    if not os.path.exists(examples_dir):
        return examples
    
    category_emojis = {'iteratives': '🔄', 'recursives': '🔁'}
    
    for category in ['iteratives', 'recursives']:
        category_path = os.path.join(examples_dir, category)
        print(f"\n📁 Categoría: {category}")
        print(f"   Ruta: {category_path}")
        print(f"   ¿Existe? {os.path.exists(category_path)}")
        
        if not os.path.exists(category_path):
            continue
        
        emoji = category_emojis.get(category, '📄')
        
        archivos = sorted(os.listdir(category_path))
        print(f"   Archivos encontrados: {len(archivos)}")
        
        for filename in archivos:
            print(f"   - {filename}", end='')
            if filename.endswith('.txt'):
                filepath = os.path.join(category_path, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    name = filename.replace('.txt', '').replace('_', ' ').title()
                    if category == 'iteratives':
                        display_name = f"{emoji} {name} (Iterativo)"
                    else:
                        display_name = f"{emoji} {name} (Recursivo)"
                    
                    examples[display_name] = content
                    print(f" ✅ Cargado como: {display_name}")
                except Exception as e:
                    print(f" ❌ Error: {e}")
            else:
                print(f" ⏭️  Ignorado (no es .txt)")
    
    return examples

print("=" * 70)
print("DIAGNÓSTICO: Carga de Ejemplos")
print("=" * 70)

EXAMPLES = load_examples_from_folder()

print("\n" + "=" * 70)
print(f"TOTAL DE EJEMPLOS CARGADOS: {len(EXAMPLES)}")
print("=" * 70)

for i, (nombre, contenido) in enumerate(EXAMPLES.items(), 1):
    print(f"{i}. {nombre} ({len(contenido)} caracteres)")

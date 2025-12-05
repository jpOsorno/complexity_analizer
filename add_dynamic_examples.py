"""
Script para agregar carga dinámica de ejemplos a app_complete.py
SOLO reemplaza la sección de EXAMPLES, nada más.
"""

# Leer el archivo
with open('app_complete.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Definir la función de carga dinámica
dynamic_loading_code = '''# ============================================================================
# CARGA DINÁMICA DE EJEMPLOS
# ============================================================================

def load_examples_from_folder():
    """Carga ejemplos dinámicamente desde la carpeta examples/."""
    examples = {}
    examples_dir = os.path.join(os.path.dirname(__file__), 'examples')
    
    if not os.path.exists(examples_dir):
        return examples
    
    category_emojis = {'iteratives': '🔄', 'recursives': '🔁'}
    
    for category in ['iteratives', 'recursives']:
        category_path = os.path.join(examples_dir, category)
        if not os.path.exists(category_path):
            continue
        
        emoji = category_emojis.get(category, '📄')
        
        for filename in sorted(os.listdir(category_path)):
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
                except Exception as e:
                    print(f"Error cargando {filepath}: {e}")
    
    return examples

EXAMPLES = load_examples_from_folder()'''

# Buscar el inicio y fin de la sección de EXAMPLES
start_marker = "# ============================================================================\n# EJEMPLOS PRECARGADOS"
end_marker = "# ============================================================================\n# HEADER"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx == -1 or end_idx == -1:
    print("❌ No se encontraron los marcadores de EXAMPLES")
    exit(1)

# Reemplazar la sección
new_content = content[:start_idx] + dynamic_loading_code + "\n\n\n" + content[end_idx:]

# Guardar
with open('app_complete.py', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ Carga dinámica de ejemplos agregada correctamente")
print("   - Función load_examples_from_folder() creada")
print("   - EXAMPLES ahora carga desde examples/iteratives y examples/recursives")

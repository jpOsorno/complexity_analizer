"""
Script para integrar el Chat Analyzer en app.py
================================================

Este script modifica app.py para agregar el análisis conversacional con IA.
"""

import re

# Leer app.py
with open('app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# ============================================================================
# MODIFICACIÓN 1: Agregar imports
# ============================================================================

import_section = """# NUEVO: Importar sistema LLM
try:
    from src.llm.unified_analyzer_llm import analyze_with_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# NUEVO: Importar chat analyzer
try:
    from src.llm.llm_chat_analyzer import ChatAnalyzer
    from src.visualization.chat_display import display_llm_chat_analysis
    CHAT_ANALYZER_AVAILABLE = True
except ImportError:
    CHAT_ANALYZER_AVAILABLE = False"""

# Reemplazar imports
old_import = """# NUEVO: Importar sistema LLM
try:
    from src.llm.unified_analyzer_llm import analyze_with_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False"""

content = content.replace(old_import, import_section)

# ============================================================================
# MODIFICACIÓN 2: Agregar toggle en sidebar
# ============================================================================

# Buscar después de la sección LLM
sidebar_addition = """
    # ========================================================================
    # NUEVO: ANÁLISIS CONVERSACIONAL CON IA
    # ========================================================================
    
    if LLM_AVAILABLE and CHAT_ANALYZER_AVAILABLE:
        st.divider()
        st.header("💬 Chat con IA")
        
        enable_chat = st.toggle(
            "Mostrar Análisis Conversacional",
            value=True,
            help="El asistente de IA explicará tu algoritmo de forma conversacional"
        )
        
        st.session_state['chat_enabled'] = enable_chat
        
        if enable_chat:
            st.info("💡 **Chat habilitado**: Recibirás explicaciones conversacionales del algoritmo")
        else:
            st.info("ℹ️ Chat deshabilitado")
    else:
        st.session_state['chat_enabled'] = False
"""

# Insertar antes del último st.divider() en sidebar
pattern = r"(    else:\s+st\.warning\(\"⚠️ Módulo LLM no disponible\"\)\s+st\.session_state\['llm_enabled'\] = False\s+)(st\.divider\(\))"
replacement = r"\1" + sidebar_addition + r"\n    \2"
content = re.sub(pattern, replacement, content)

# ============================================================================
# MODIFICACIÓN 3: Agregar chat después de resultados
# ============================================================================

chat_display_code = """
                        # NUEVO: Mostrar chat conversacional si está habilitado
                        chat_enabled = st.session_state.get('chat_enabled', False)
                        if chat_enabled and CHAT_ANALYZER_AVAILABLE:
                            try:
                                st.divider()
                                with st.spinner("🤖 Generando análisis conversacional..."):
                                    chat_analyzer = ChatAnalyzer(api_key)
                                    
                                    # Extraer complejidad para contexto
                                    first_proc = next(iter(results.values()))
                                    static_complexity = {
                                        'worst_case': getattr(first_proc, 'final_worst', 'O(?)'),
                                        'algorithm_type': getattr(first_proc, 'algorithm_type', 'unknown')
                                    }
                                    
                                    chat_messages = chat_analyzer.analyze(code_input, static_complexity)
                                    display_llm_chat_analysis(chat_messages)
                            except Exception as e:
                                st.warning(f"⚠️ No se pudo generar análisis conversacional: {e}")
"""

# Insertar después de display_procedure_analysis(results) en la sección CON LLM
pattern2 = r"(display_procedure_analysis\(results\)\s+)(st\.divider\(\))"
content = re.sub(pattern2, r"\1" + chat_display_code + r"\n                        \2", content, count=1)

# Guardar archivo modificado
with open('app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("✅ app.py modificado exitosamente")
print("📝 Cambios realizados:")
print("  1. Agregados imports de ChatAnalyzer y display_llm_chat_analysis")
print("  2. Agregado toggle 'Mostrar Análisis Conversacional' en sidebar")
print("  3. Agregado display de chat después de resultados del análisis")

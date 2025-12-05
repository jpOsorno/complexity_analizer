"""
Analizador de Complejidad Computacional con Chat IA
====================================================

Interfaz web para analizar la complejidad de algoritmos en pseudocódigo.
VERSIÓN CON CHAT CONVERSACIONAL INTEGRADO.

Ejecutar: streamlit run app_with_chat.py
"""

import streamlit as st
import sys
import os

# Agregar src al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.parser.parser import parse, ParseError
from src.analyzer.unified_analyzer import analyze_complexity_unified
from src.visualization.components import (
    display_procedure_analysis,
)

# Importar sistema LLM
try:
    from src.llm.unified_analyzer_llm import analyze_with_llm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Importar chat analyzer
try:
    from src.llm.llm_chat_analyzer import ChatAnalyzer
    from src.visualization.chat_display import display_llm_chat_analysis
    CHAT_ANALYZER_AVAILABLE = True
except ImportError:
    CHAT_ANALYZER_AVAILABLE = False


# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Analizador de Complejidad con IA",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
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

EXAMPLES = load_examples_from_folder()


# ============================================================================
# HEADER
# ============================================================================

st.title("🔍 Analizador de Complejidad Computacional con IA")
st.markdown("""
Analiza la complejidad de algoritmos escritos en pseudocódigo.
Soporta algoritmos **iterativos** y **recursivos**.
""")

st.divider()


# ============================================================================
# SIDEBAR: EJEMPLOS + CONFIGURACIÓN LLM + CHAT
# ============================================================================

with st.sidebar:
    st.header("📚 Ejemplos")
    st.markdown("Selecciona un ejemplo para cargar automáticamente:")
    
    selected_example = st.selectbox(
        "Algoritmo:",
        options=[""] + list(EXAMPLES.keys()),
        format_func=lambda x: "-- Seleccionar --" if x == "" else x
    )
    
    if selected_example and selected_example in EXAMPLES:
        if st.button("📥 Cargar Ejemplo", use_container_width=True):
            st.session_state['code_input'] = EXAMPLES[selected_example]
            # Forzar actualización del textarea
            st.session_state['code_area'] = EXAMPLES[selected_example]
            st.rerun()
    
    st.divider()
    
    # ========================================================================
    # CONFIGURACIÓN LLM
    # ========================================================================
    
    st.header("🤖 Validación con IA")
    
    if LLM_AVAILABLE:
        enable_llm = st.toggle(
            "Habilitar validación con LLM",
            value=False,
            help="Compara tu análisis con Llama 3.3 70B (Groq API)"
        )
        
        if enable_llm:
            st.info("💡 **Validación con IA habilitada**")
            
            # Verificar si hay API key en variable de entorno
            api_key_env = os.getenv('GROQ_API_KEY')
            
            # SIEMPRE mostrar el campo de input, pero con valor por defecto si existe en env
            if api_key_env:
                st.success("✓ API Key detectada en variables de entorno")
                #st.caption("Puedes usar la detectada o ingresar una nueva abajo:")
            else:
                st.warning("⚠️ No se detectó GROQ_API_KEY en variables de entorno")
            
            # # Campo de input SIEMPRE visible
            # api_key_input = st.text_input(
            #     "API Key de Groq:",
            #     value=api_key_env if api_key_env else "",
            #     type="password",
            #     help="Obtén tu API key gratuita en https://console.groq.com/keys",
            #     placeholder="gsk_..."
            # )
            
            # # Usar la key ingresada o la del entorno
            # api_key = api_key_input if api_key_input else api_key_env
            
            # if not api_key:
            #     st.error("❌ Ingresa tu API key para usar validación LLM")
            # else:
            #     st.success(f"✓ API Key configurada ({api_key[:10]}...)")
            
            # # Guardar en session state
            # if api_key:
            #     st.session_state['groq_api_key'] = api_key
            #     st.session_state['llm_enabled'] = True
            # else:
            #     st.session_state['llm_enabled'] = False
        else:
            st.session_state['llm_enabled'] = False
            st.info("ℹ️ Validación LLM deshabilitada")
    else:
        st.warning("⚠️ Módulo LLM no disponible")
        st.session_state['llm_enabled'] = False
    
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
    
    st.divider()


# ============================================================================
# MAIN: ENTRADA DE CÓDIGO
# ============================================================================

st.header("✏️ Entrada de Código")

code_input = st.text_area(
    "Escribe o pega tu pseudocódigo:",
    value=st.session_state.get('code_input', ''),
    height=300,
    key='code_area'
)

col1, col2, col3 = st.columns([1, 1, 4])

with col1:
    analyze_button = st.button("🚀 Analizar", type="primary", use_container_width=True)

with col2:
    clear_button = st.button("🗑️ Limpiar", use_container_width=True)

if clear_button:
    st.session_state['code_input'] = ''
    st.session_state['code_area'] = ''
    st.rerun()


# ============================================================================
# ANÁLISIS Y RESULTADOS (MEJORADO CON LLM Y CHAT)
# ============================================================================

if analyze_button:
    if not code_input.strip():
        st.error("⚠️ Por favor, ingresa código pseudocódigo para analizar.")
    else:
        # Verificar si LLM está habilitado
        llm_enabled = st.session_state.get('llm_enabled', False)
        
        with st.spinner("🔄 Analizando algoritmo..."):
            try:
                # ============================================================
                # SIN LLM: Análisis normal
                # ============================================================
                
                if not llm_enabled:
                    # Paso 1: Parsear
                    ast = parse(code_input)
                    
                    # Paso 2: Analizar
                    results = analyze_complexity_unified(ast)
                    
                    # Paso 3: Mostrar
                    st.success("✅ Análisis completado exitosamente")
                    st.divider()
                    st.header("📊 Resultados del Análisis")
                    
                    display_procedure_analysis(results)                    
                
                # ============================================================
                # CON LLM: Análisis + Validación + Chat
                # ============================================================
                
                else:
                    # Verificar API key
                    api_key = st.session_state.get('groq_api_key')
                    
                    if not api_key:
                        st.error("❌ API key no configurada. Desactiva LLM o configura tu API key.")
                    else:
                        # Configurar API key como variable de entorno temporal
                        os.environ['GROQ_API_KEY'] = api_key
                        
                        # Analizar con LLM
                        results = analyze_with_llm(code_input, enable_llm=True)
                        
                        st.success("✅ Análisis completado (con validación IA)")
                        
                        st.divider()
                        st.header("📊 Resultados del Análisis con IA")
                        
                        # Mostrar resultados con validación LLM
                        display_procedure_analysis(results)
                        
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
                        
                        st.divider()
                        
                                
            except ParseError as e:
                st.error(f"❌ **Error de Sintaxis**")
                st.code(str(e), language=None)
                st.info("💡 Verifica que el código siga la sintaxis correcta del pseudocódigo.")
                
            except Exception as e:
                st.error(f"❌ **Error Inesperado**")
                st.code(str(e), language=None)
                
                # Mostrar traceback en expander para debugging
                with st.expander("🔍 Ver detalles del error"):
                    import traceback
                    st.code(traceback.format_exc())


# ============================================================================
# FOOTER
# ============================================================================

st.divider()

st.markdown("""
<div style="text-align: center; color: #6b7280; font-size: 0.875rem;">
    <p>
        🎓 Proyecto de Análisis y Diseño de Algoritmos<br>
        Universidad: Universidad de Caldas | 2025<br>
        ✨ Con análisis conversacional por IA
    </p>
</div>
""", unsafe_allow_html=True)

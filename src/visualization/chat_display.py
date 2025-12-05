"""
Chat Display Component for LLM Analysis
========================================

Componente separado para mostrar análisis conversacional del LLM.
"""

import streamlit as st
from typing import List


def display_llm_chat_analysis(messages: List, show_header: bool = True):
    """
    Muestra análisis conversacional del LLM en formato de chat.
    
    Args:
        messages: Lista de ChatMessage del ChatAnalyzer
        show_header: Si mostrar el header de la sección
    """
    if show_header:
        st.markdown("### 🤖 Análisis Conversacional con IA")
        st.markdown("*El asistente de IA analiza tu algoritmo de forma conversacional*")
        st.divider()
    
    # Estilos CSS para chat bubbles
    chat_style = """
    <style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .chat-identification {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .chat-pattern {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .chat-complexity {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .chat-optimization {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
    }
    
    .chat-insight {
        background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        color: white;
    }
    
    .chat-strength {
        background: linear-gradient(135deg, #30cfd0 0%, #330867 100%);
        color: white;
    }
    
    .chat-complexity_summary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: 3px solid #ffd700;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }
    
    .chat-error {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5a6f 100%);
        color: white;
    }
    
    .chat-general {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #333;
    }
    
    .chat-emoji {
        font-size: 1.5rem;
        margin-right: 0.5rem;
    }
    
    .chat-text {
        line-height: 1.6;
        font-size: 1rem;
    }
    </style>
    """
    
    st.markdown(chat_style, unsafe_allow_html=True)
    
    # Mostrar cada mensaje como chat bubble
    for i, msg in enumerate(messages):
        # Manejo especial para complexity_summary
        if msg.message_type == 'complexity_summary':
            st.info(f"{msg.emoji} **Complejidades calculadas por Groq**")
            st.markdown(msg.text)
            st.divider()
            continue
        
        # Determinar clase CSS según tipo
        css_class = f"chat-{msg.message_type}"
        
        # Crear HTML del mensaje
        message_html = f"""
        <div class="chat-message {css_class}">
            <div>
                <span class="chat-emoji">{msg.emoji}</span>
                <strong>{msg.message_type.replace('_', ' ').title()}</strong>
            </div>
            <div class="chat-text" style="margin-top: 0.5rem;">
                {msg.text}
            </div>
        </div>
        """
        
        st.markdown(message_html, unsafe_allow_html=True)
        
        # Si hay código snippet, mostrarlo
        if msg.code_snippet:
            st.code(msg.code_snippet, language='text')
        
        # Pequeña pausa visual entre mensajes
        if i < len(messages) - 1:
            st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
    
    # Footer del chat
    st.divider()
    st.caption("💡 Este análisis fue generado por IA y puede contener imprecisiones. Úsalo como guía complementaria.")

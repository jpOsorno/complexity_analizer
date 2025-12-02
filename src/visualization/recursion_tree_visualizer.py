"""
Visualizador de Árboles de Recursión
====================================

Genera diagramas interactivos de árboles de recursión usando Plotly.

Características:
- Construcción automática de árboles desde ecuaciones de recurrencia
- Visualización de niveles, nodos y costos
- Interactividad con zoom y tooltips
- Cálculo de complejidad por niveles
"""

import plotly.graph_objects as go
import numpy as np
from typing import List, Tuple, Optional, Dict
import math


# ============================================================================
# CONSTRUCCIÓN DE ÁRBOLES DE RECURSIÓN
# ============================================================================

class RecursionTreeNode:
    """Representa un nodo en el árbol de recursión"""
    
    def __init__(self, size: str, cost: str, level: int = 0, index: int = 0):
        """
        Args:
            size: Tamaño del problema (ej: "n", "n/2", "n-1")
            cost: Costo del nodo (ej: "O(n)", "O(1)")
            level: Profundidad en el árbol
            index: Índice del nodo en su nivel
        """
        self.size = size
        self.cost = cost
        self.level = level
        self.index = index
        self.children: List['RecursionTreeNode'] = []
    
    def add_child(self, child: 'RecursionTreeNode'):
        """Agrega un hijo al nodo"""
        self.children.append(child)
    
    def __repr__(self):
        return f"Node(size={self.size}, cost={self.cost}, level={self.level})"


class RecursionTreeBuilder:
    """Construye árboles de recursión desde ecuaciones"""
    
    @staticmethod
    def build_divide_conquer_tree(
        a: int, 
        b: int, 
        f_n: str, 
        max_depth: int = 5
    ) -> RecursionTreeNode:
        """
        Construye árbol para T(n) = aT(n/b) + f(n)
        
        Args:
            a: Número de subproblemas
            b: Factor de división
            f_n: Costo no recursivo
            max_depth: Profundidad máxima del árbol
            
        Returns:
            Nodo raíz del árbol
        """
        root = RecursionTreeNode(size="n", cost=f_n, level=0, index=0)
        
        def build_recursive(node: RecursionTreeNode, depth: int):
            if depth >= max_depth:
                return
            
            # Crear 'a' hijos de tamaño n/b
            for i in range(a):
                child_size = f"n/{b**node.level}/{b}" if node.level > 0 else f"n/{b}"
                child = RecursionTreeNode(
                    size=child_size,
                    cost=f_n,
                    level=node.level + 1,
                    index=i + node.index * a
                )
                node.add_child(child)
                build_recursive(child, depth + 1)
        
        build_recursive(root, 0)
        return root
    
    @staticmethod
    def build_subtract_conquer_tree(
        k: int,
        f_n: str,
        max_depth: int = 8
    ) -> RecursionTreeNode:
        """
        Construye árbol para T(n) = T(n-k) + f(n)
        
        Args:
            k: Reducción por paso
            f_n: Costo no recursivo
            max_depth: Profundidad máxima
            
        Returns:
            Nodo raíz del árbol
        """
        root = RecursionTreeNode(size="n", cost=f_n, level=0, index=0)
        
        def build_recursive(node: RecursionTreeNode, depth: int):
            if depth >= max_depth:
                return
            
            # Un solo hijo de tamaño n-k
            child_size = f"n-{k*(node.level+1)}"
            child = RecursionTreeNode(
                size=child_size,
                cost=f_n,
                level=node.level + 1,
                index=0
            )
            node.add_child(child)
            build_recursive(child, depth + 1)
        
        build_recursive(root, 0)
        return root
    
    @staticmethod
    def build_fibonacci_tree(max_depth: int = 6) -> RecursionTreeNode:
        """
        Construye árbol para Fibonacci: T(n) = T(n-1) + T(n-2) + O(1)
        
        Args:
            max_depth: Profundidad máxima
            
        Returns:
            Nodo raíz del árbol
        """
        root = RecursionTreeNode(size="n", cost="c", level=0, index=0)
        
        def build_recursive(node: RecursionTreeNode, depth: int):
            if depth >= max_depth:
                return
            
            # Dos hijos: n-1 y n-2
            for i, offset in enumerate([1, 2]):
                child_size = f"n-{offset*(node.level+1)}"
                child = RecursionTreeNode(
                    size=child_size,
                    cost="c",
                    level=node.level + 1,
                    index=i + node.index * 2
                )
                node.add_child(child)
                build_recursive(child, depth + 1)
        
        build_recursive(root, 0)
        return root


# ============================================================================
# VISUALIZADOR DE ÁRBOLES
# ============================================================================

class RecursionTreeVisualizer:
    """Visualiza árboles de recursión con Plotly"""
    
    def __init__(self, theme: str = "plotly_white"):
        self.theme = theme
        self.colors = {
            0: "#3b82f6",  # Nivel 0 - Azul
            1: "#10b981",  # Nivel 1 - Verde
            2: "#f59e0b",  # Nivel 2 - Amarillo
            3: "#ef4444",  # Nivel 3 - Rojo
            4: "#8b5cf6",  # Nivel 4 - Morado
        }
    
    def visualize(
        self,
        root: RecursionTreeNode,
        title: str = "Árbol de Recursión"
    ) -> go.Figure:
        """
        Genera visualización del árbol.
        
        Args:
            root: Nodo raíz del árbol
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        # Calcular posiciones de nodos
        positions = self._calculate_positions(root)
        
        # Extraer coordenadas y metadatos
        x_nodes, y_nodes, labels, colors, hover_texts = [], [], [], [], []
        x_edges, y_edges = [], []
        
        def traverse(node: RecursionTreeNode):
            pos = positions[id(node)]
            x_nodes.append(pos[0])
            y_nodes.append(pos[1])
            labels.append(f"T({node.size})")
            colors.append(self.colors.get(node.level % 5, "#6b7280"))
            hover_texts.append(
                f"<b>Nodo:</b> T({node.size})<br>"
                f"<b>Costo:</b> {node.cost}<br>"
                f"<b>Nivel:</b> {node.level}<br>"
                f"<b>Hijos:</b> {len(node.children)}"
            )
            
            # Dibujar aristas
            for child in node.children:
                child_pos = positions[id(child)]
                x_edges.extend([pos[0], child_pos[0], None])
                y_edges.extend([pos[1], child_pos[1], None])
                traverse(child)
        
        traverse(root)
        
        # Crear figura
        fig = go.Figure()
        
        # Agregar aristas
        fig.add_trace(go.Scatter(
            x=x_edges,
            y=y_edges,
            mode='lines',
            line=dict(color='#cbd5e1', width=2),
            hoverinfo='skip',
            showlegend=False
        ))
        
        # Agregar nodos
        fig.add_trace(go.Scatter(
            x=x_nodes,
            y=y_nodes,
            mode='markers+text',
            marker=dict(
                size=40,
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=labels,
            textposition="middle center",
            textfont=dict(size=10, color='white', family='monospace'),
            hovertext=hover_texts,
            hoverinfo='text',
            showlegend=False
        ))
        
        # Configurar layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=18)),
            template=self.theme,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            hovermode='closest',
            height=600,
            showlegend=False
        )
        
        return fig
    
    def _calculate_positions(
        self,
        root: RecursionTreeNode
    ) -> Dict[int, Tuple[float, float]]:
        """
        Calcula posiciones (x, y) para cada nodo del árbol.
        
        Usa algoritmo de Reingold-Tilford para árboles balanceados.
        
        Args:
            root: Nodo raíz
            
        Returns:
            Dict {id(node): (x, y)}
        """
        positions = {}
        
        # Contar nodos por nivel para espaciado horizontal
        level_counts = {}
        level_indices = {}
        
        def count_levels(node: RecursionTreeNode):
            level = node.level
            if level not in level_counts:
                level_counts[level] = 0
                level_indices[level] = 0
            level_counts[level] += 1
            for child in node.children:
                count_levels(child)
        
        count_levels(root)
        
        # Asignar posiciones
        def assign_positions(node: RecursionTreeNode, x_offset: float = 0, width: float = 10):
            y = -node.level  # Y negativo para que el árbol crezca hacia abajo
            
            if not node.children:
                # Nodo hoja
                x = x_offset
            else:
                # Nodo interno: centrar entre hijos
                child_width = width / len(node.children)
                child_positions = []
                
                for i, child in enumerate(node.children):
                    child_x = x_offset + i * child_width + child_width / 2
                    assign_positions(child, child_x - child_width / 2, child_width)
                    child_positions.append(positions[id(child)][0])
                
                x = sum(child_positions) / len(child_positions)
            
            positions[id(node)] = (x, y)
        
        assign_positions(root)
        return positions
    
    def generate_cost_by_level_chart(
        self,
        root: RecursionTreeNode,
        a: int,
        cost_per_node: str = "n"
    ) -> go.Figure:
        """
        Genera gráfico de costo por nivel del árbol.
        
        Args:
            root: Nodo raíz
            a: Número de hijos por nodo
            cost_per_node: Expresión del costo por nodo
            
        Returns:
            Figura de Plotly
        """
        # Calcular niveles y costos
        max_level = self._get_max_depth(root)
        levels = list(range(max_level + 1))
        num_nodes = [a**i for i in levels]
        
        # Simplificar costo (asumimos que es proporcional a n)
        if "n" in cost_per_node:
            # Costo por nivel: a^i × (n / b^i) = (a/b)^i × n
            cost_per_level = [f"a^{i} × n/b^{i}" for i in levels]
        else:
            # Costo constante por nodo
            cost_per_level = [f"a^{i} × c" for i in levels]
        
        # Crear figura
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=levels,
            y=num_nodes,
            name="Número de nodos",
            marker=dict(color='#3b82f6'),
            hovertemplate='<b>Nivel %{x}</b><br>Nodos: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Nodos por Nivel del Árbol de Recursión",
            xaxis_title="Nivel (profundidad)",
            yaxis_title="Número de nodos",
            template=self.theme,
            height=400
        )
        
        return fig
    
    def _get_max_depth(self, root: RecursionTreeNode) -> int:
        """Calcula la profundidad máxima del árbol"""
        if not root.children:
            return root.level
        return max(self._get_max_depth(child) for child in root.children)


# ============================================================================
# API SIMPLIFICADA
# ============================================================================

def visualize_divide_conquer_tree(
    a: int,
    b: int,
    f_n: str = "O(n)",
    max_depth: int = 4
) -> go.Figure:
    """
    API simple para visualizar árbol divide y vencerás.
    
    Args:
        a: Número de subproblemas
        b: Factor de división
        f_n: Costo no recursivo
        max_depth: Profundidad máxima
        
    Returns:
        Figura de Plotly
    
    Example:
        >>> fig = visualize_divide_conquer_tree(2, 2, "O(n)")
        >>> fig.show()
    """
    builder = RecursionTreeBuilder()
    tree = builder.build_divide_conquer_tree(a, b, f_n, max_depth)
    
    visualizer = RecursionTreeVisualizer()
    return visualizer.visualize(
        tree,
        title=f"Árbol de Recursión: T(n) = {a}T(n/{b}) + {f_n}"
    )


def visualize_fibonacci_tree(max_depth: int = 5) -> go.Figure:
    """
    API simple para visualizar árbol de Fibonacci.
    
    Args:
        max_depth: Profundidad máxima
        
    Returns:
        Figura de Plotly
    """
    builder = RecursionTreeBuilder()
    tree = builder.build_fibonacci_tree(max_depth)
    
    visualizer = RecursionTreeVisualizer()
    return visualizer.visualize(
        tree,
        title="Árbol de Recursión: Fibonacci"
    )


# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("DEMO: VISUALIZADOR DE ÁRBOLES DE RECURSIÓN")
    print("="*70)
    
    # Ejemplo 1: Merge Sort (T(n) = 2T(n/2) + O(n))
    print("\n📊 Generando árbol de Merge Sort...")
    fig1 = visualize_divide_conquer_tree(2, 2, "O(n)", max_depth=4)
    fig1.write_html("merge_sort_tree.html")
    print("✓ Guardado en merge_sort_tree.html")
    
    # Ejemplo 2: Binary Search (T(n) = T(n/2) + O(1))
    print("\n📊 Generando árbol de Binary Search...")
    fig2 = visualize_divide_conquer_tree(1, 2, "O(1)", max_depth=5)
    fig2.write_html("binary_search_tree.html")
    print("✓ Guardado en binary_search_tree.html")
    
    # Ejemplo 3: Fibonacci
    print("\n📊 Generando árbol de Fibonacci...")
    fig3 = visualize_fibonacci_tree(max_depth=5)
    fig3.write_html("fibonacci_tree.html")
    print("✓ Guardado en fibonacci_tree.html")
    
    # Ejemplo 4: Strassen (T(n) = 7T(n/2) + O(n²))
    print("\n📊 Generando árbol de Strassen...")
    fig4 = visualize_divide_conquer_tree(7, 2, "O(n²)", max_depth=3)
    fig4.write_html("strassen_tree.html")
    print("✓ Guardado en strassen_tree.html")
    
    print("\n✅ Demo completado. Abre los archivos .html en tu navegador.")
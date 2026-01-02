"""
Architecture Diagram Generator for Medical AI Agent with RAG
Clean, professional layout with proper connections
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patches as mpatches


def create_architecture_diagram():
    """Create a clean architecture diagram with proper layout"""
    
    fig, ax = plt.subplots(figsize=(16, 11), facecolor='white')
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 11)
    ax.axis('off')
    
    # Colors - professional palette
    BLUE = '#3498DB'
    LIGHT_BLUE = '#85C1E9'
    ORANGE = '#E67E22'
    LIGHT_ORANGE = '#F5B041'
    GREEN = '#27AE60'
    LIGHT_GREEN = '#82E0AA'
    PURPLE = '#9B59B6'
    LIGHT_PURPLE = '#D7BDE2'
    YELLOW = '#F1C40F'
    GRAY = '#95A5A6'
    DARK = '#2C3E50'
    WHITE = '#FFFFFF'
    
    def box(x, y, w, h, color, label, fontsize=11):
        """Draw a simple rounded box with centered label"""
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            facecolor=color,
            edgecolor=DARK,
            linewidth=2
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label,
                ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color=DARK)
        return (x, y, w, h)
    
    def small_box(x, y, w, h, label, fontsize=8):
        """Draw a small white component box"""
        rect = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.01,rounding_size=0.1",
            facecolor=WHITE,
            edgecolor=GRAY,
            linewidth=1
        )
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label,
                ha='center', va='center',
                fontsize=fontsize, color=DARK)
    
    def arrow(x1, y1, x2, y2, style='->', color=GRAY, lw=2):
        """Draw arrow from (x1,y1) to (x2,y2)"""
        ax.annotate('',
                    xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle=style, color=color, lw=lw))
    
    # ==================== TITLE ====================
    ax.text(8, 10.5, 'Medical AI Agent with RAG - System Architecture',
            ha='center', va='center', fontsize=18, fontweight='bold', color=DARK)
    
    # ==================== ROW 1: User & Frontend ====================
    # User
    box(1, 8.5, 2, 1.2, LIGHT_BLUE, 'USER')
    ax.text(2, 8.2, 'Web Browser', ha='center', fontsize=8, color=GRAY)
    
    # Frontend
    box(4, 8.5, 2.5, 1.2, BLUE, 'FRONTEND')
    ax.text(5.25, 8.2, 'HTML / JS / Bootstrap', ha='center', fontsize=7, color=WHITE)
    
    # Observability
    box(11, 8.5, 4, 1.2, YELLOW, 'OBSERVABILITY')
    ax.text(13, 8.2, 'Langfuse Tracing | Session Mgmt | Scoring', ha='center', fontsize=7, color=DARK)
    
    # ==================== ROW 2: API Layer ====================
    box(4, 6.5, 2.5, 1.5, BLUE, 'FLASK API')
    small_box(4.1, 6.6, 2.3, 0.5, '/agent  /rag  /history  /batch')
    
    # ==================== ROW 3: Main Processing ====================
    # CrewAI Agents Box
    box(0.5, 3.5, 4.5, 2.5, LIGHT_ORANGE, '')
    ax.text(2.75, 5.7, 'CREWAI AGENTS', ha='center', fontsize=11, fontweight='bold', color=DARK)
    small_box(0.7, 4.8, 2, 0.6, 'Medical Assistant')
    small_box(2.9, 4.8, 2, 0.6, 'Researcher')
    small_box(0.7, 4.0, 2, 0.6, 'Educator')
    small_box(2.9, 4.0, 2, 0.6, 'RAG Tools')
    
    # RAG Pipeline Box
    box(5.5, 3.5, 4.5, 2.5, ORANGE, '')
    ax.text(7.75, 5.7, 'RAG PIPELINE', ha='center', fontsize=11, fontweight='bold', color=DARK)
    small_box(5.7, 4.8, 2, 0.6, 'Document Loader')
    small_box(7.9, 4.8, 2, 0.6, 'Text Chunker')
    small_box(5.7, 4.0, 2, 0.6, 'Embeddings')
    small_box(7.9, 4.0, 2, 0.6, 'Retriever')
    
    # LLM Layer Box
    box(10.5, 3.5, 4.5, 2.5, LIGHT_PURPLE, '')
    ax.text(12.75, 5.7, 'LLM LAYER', ha='center', fontsize=11, fontweight='bold', color=DARK)
    small_box(10.7, 4.6, 4.1, 0.6, 'Google Gemini 2.5-flash')
    small_box(10.7, 3.8, 4.1, 0.6, 'LangChain Integration')
    
    # ==================== ROW 4: Storage ====================
    # Cache & History
    box(0.5, 1, 4.5, 2, LIGHT_GREEN, '')
    ax.text(2.75, 2.7, 'CACHE & HISTORY', ha='center', fontsize=11, fontweight='bold', color=DARK)
    small_box(0.7, 1.8, 2, 0.6, 'Redis Cache')
    small_box(2.9, 1.8, 2, 0.6, 'In-Memory Fallback')
    small_box(0.7, 1.1, 4.1, 0.5, 'Session History Storage')
    
    # Vector Storage
    box(5.5, 1, 4.5, 2, GREEN, '')
    ax.text(7.75, 2.7, 'VECTOR STORAGE', ha='center', fontsize=11, fontweight='bold', color=DARK)
    small_box(5.7, 1.8, 2, 0.6, 'ChromaDB')
    small_box(7.9, 1.8, 2, 0.6, 'FAISS')
    small_box(5.7, 1.1, 4.1, 0.5, 'Medical Knowledge Base (JSON)')
    
    # Data Sources
    box(10.5, 1, 4.5, 2, PURPLE, '')
    ax.text(12.75, 2.7, 'DATA SOURCES', ha='center', fontsize=11, fontweight='bold', color=WHITE)
    small_box(10.7, 1.8, 2, 0.6, 'MedQA')
    small_box(12.9, 1.8, 2, 0.6, 'MedDialog')
    small_box(10.7, 1.1, 2, 0.5, 'HealthSearchQA')
    small_box(12.9, 1.1, 2, 0.5, 'LiveQA')
    
    # ==================== ARROWS ====================
    # User <-> Frontend
    arrow(3, 9.1, 4, 9.1, '<->')
    
    # Frontend <-> API
    arrow(5.25, 8.5, 5.25, 8.0, '<->')
    
    # API -> Agents
    arrow(4, 6.8, 2.75, 6.0, '->')
    
    # API -> RAG Pipeline
    arrow(6.5, 6.8, 7.75, 6.0, '->')
    
    # Agents <-> RAG Pipeline
    arrow(5.0, 4.75, 5.5, 4.75, '<->')
    
    # RAG Pipeline <-> LLM
    arrow(10.0, 4.75, 10.5, 4.75, '<->')
    
    # Agents -> LLM (curved)
    ax.annotate('',
                xy=(10.5, 5.5), xytext=(5.0, 5.5),
                arrowprops=dict(arrowstyle='->', color=GRAY, lw=1.5,
                               connectionstyle='arc3,rad=-0.15'))
    
    # RAG -> Vector Storage
    arrow(7.75, 3.5, 7.75, 3.0, '<->')
    
    # API -> Cache (cleaner path: down from API, then left to Cache)
    arrow(4.5, 6.5, 2.75, 3.0, '->')
    
    # Vector Storage <- Data Sources
    arrow(10.5, 2, 10.0, 2, '<-')
    
    # Observability connections (dashed)
    # From API to Observability
    ax.plot([6.5, 11], [7.25, 8.8], color='#F39C12', lw=1.5, ls='--')
    ax.annotate('', xy=(11, 8.8), xytext=(10.7, 8.65),
                arrowprops=dict(arrowstyle='->', color='#F39C12', lw=1.5))
    
    # From LLM to Observability
    ax.plot([12.75, 12.75], [6.0, 8.5], color='#F39C12', lw=1.5, ls='--')
    ax.annotate('', xy=(12.75, 8.5), xytext=(12.75, 8.2),
                arrowprops=dict(arrowstyle='->', color='#F39C12', lw=1.5))
    
    # ==================== LEGEND ====================
    ax.text(0.5, 0.4, 'Legend:', fontsize=9, fontweight='bold', color=DARK)
    
    arrow(1.5, 0.35, 2.5, 0.35, '<->')
    ax.text(2.7, 0.35, 'Data Flow', fontsize=8, va='center', color=DARK)
    
    ax.plot([4, 5], [0.35, 0.35], color='#F39C12', lw=1.5, ls='--')
    ax.annotate('', xy=(5, 0.35), xytext=(4.8, 0.35),
                arrowprops=dict(arrowstyle='->', color='#F39C12', lw=1.5))
    ax.text(5.2, 0.35, 'Telemetry', fontsize=8, va='center', color=DARK)
    
    # Save
    plt.tight_layout()
    plt.savefig('architecture_diagram.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('architecture_diagram.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: architecture_diagram.png and architecture_diagram.pdf")
    plt.close()


if __name__ == "__main__":
    create_architecture_diagram()

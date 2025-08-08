"""
Modern Glassy UI Theme for Data Science Agent Swarm

Provides a clean, professional interface without emojis.
"""

import gradio as gr

def create_modern_theme():
    """Create modern glassy theme."""
    return gr.themes.Soft().set(
        # Primary colors
        primary_hue="slate",
        secondary_hue="slate",
        
        # Neutral colors
        neutral_hue="slate",
        
        # Spacing
        spacing_size="md",
        radius_size="lg",
        
        # Typography
        font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        font_mono=["JetBrains Mono", "ui-monospace", "monospace"],
        
        # Custom CSS
        css="""
        .gradio-container {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%);
        }
        
        .main-header {
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
            text-align: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        .main-header h1 {
            color: #1e293b;
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 0.5rem;
        }
        
        .main-header p {
            color: #64748b;
            font-size: 1.1rem;
            margin-bottom: 0.5rem;
        }
        
        .main-header em {
            color: #475569;
            font-style: italic;
        }
        
        .glass-card {
            background: rgba(255, 255, 255, 0.7);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        }
        
        .status-indicator {
            display: inline-block;
            width: 8px;
            height: 8px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-active {
            background-color: #10b981;
        }
        
        .status-inactive {
            background-color: #6b7280;
        }
        
        .status-error {
            background-color: #ef4444;
        }
        
        .metric-card {
            background: rgba(255, 255, 255, 0.8);
            border-radius: 8px;
            padding: 1rem;
            margin: 0.5rem;
            border: 1px solid rgba(255, 255, 255, 0.4);
        }
        
        .metric-value {
            font-size: 1.5rem;
            font-weight: 600;
            color: #1e293b;
        }
        
        .metric-label {
            font-size: 0.875rem;
            color: #64748b;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }
        
        .button-primary {
            background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
            border: none;
            border-radius: 8px;
            color: white;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .button-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
        }
        
        .button-secondary {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(203, 213, 225, 0.8);
            border-radius: 8px;
            color: #475569;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .button-secondary:hover {
            background: rgba(255, 255, 255, 0.9);
            transform: translateY(-1px);
        }
        
        .input-field {
            background: rgba(255, 255, 255, 0.8);
            border: 1px solid rgba(203, 213, 225, 0.8);
            border-radius: 8px;
            transition: all 0.2s ease;
        }
        
        .input-field:focus {
            border-color: #3b82f6;
            box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        }
        
        .results-container {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.4);
            box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
        }
        
        .progress-bar {
            background: linear-gradient(90deg, #3b82f6 0%, #1d4ed8 100%);
            border-radius: 4px;
            height: 6px;
        }
        
        .accordion-header {
            background: rgba(255, 255, 255, 0.7);
            border-radius: 8px;
            border: 1px solid rgba(203, 213, 225, 0.8);
        }
        
        .accordion-content {
            background: rgba(255, 255, 255, 0.5);
            border-radius: 0 0 8px 8px;
            border: 1px solid rgba(203, 213, 225, 0.8);
            border-top: none;
        }
        """
    )

def create_header_html():
    """Create modern header HTML."""
    return """
    <div class="main-header">
        <h1>Data Science Agent Swarm</h1>
        <p>Autonomous multi-agent system for end-to-end data science research</p>
        <p><em>Powered by AI agents that discover data, perform analysis, build models, and generate insights</em></p>
    </div>
    """

def create_metric_card(title: str, value: str, status: str = "active"):
    """Create a metric card component."""
    status_class = f"status-{status}"
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        <span class="status-indicator {status_class}"></span>
    </div>
    """

def create_status_indicator(status: str, text: str):
    """Create a status indicator."""
    status_class = f"status-{status}"
    return f'<span class="status-indicator {status_class}"></span> {text}'

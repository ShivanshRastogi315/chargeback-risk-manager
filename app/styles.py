"""
Custom CSS styles and themes for the AI Risk Manager Streamlit Demo Application.
"""

CUSTOM_CSS = """
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* Global theme variables */
:root {
    --bg-primary: #0b0f19;
    --bg-card: rgba(18, 26, 43, 0.85);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-accent: rgba(59, 130, 246, 0.3);
    --accent-blue: #3b82f6;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-amber: #f59e0b;
    --accent-rose: #f43f5e;
    --accent-purple: #8b5cf6;
    --text-primary: #f8fafc;
    --text-muted: #94a3b8;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

code, pre, .mono-font {
    font-family: 'JetBrains Mono', monospace !important;
}

/* App Header Banner */
.hero-banner {
    background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.90) 50%, rgba(15, 23, 42, 0.95) 100%);
    border: 1px solid var(--border-accent);
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5), 0 0 20px -5px rgba(59, 130, 246, 0.15);
}

.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.03em;
    background: linear-gradient(90deg, #60a5fa 0%, #38bdf8 50%, #34d399 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 8px 0;
}

.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    margin: 0 0 16px 0;
    font-weight: 400;
}

/* Badge Pills */
.badge-pill {
    display: inline-flex;
    align-items: center;
    padding: 4px 12px;
    border-radius: 9999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-right: 8px;
    margin-bottom: 4px;
    letter-spacing: 0.02em;
}

.badge-defense {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.badge-citation {
    background: rgba(6, 182, 212, 0.15);
    color: #22d3ee;
    border: 1px solid rgba(6, 182, 212, 0.3);
}

.badge-conformal {
    background: rgba(139, 92, 246, 0.15);
    color: #a78bfa;
    border: 1px solid rgba(139, 92, 246, 0.3);
}

.badge-drift {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}

/* UI Cards */
.custom-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 16px;
    transition: all 0.2s ease-in-out;
}

.custom-card:hover {
    border-color: rgba(255, 255, 255, 0.15);
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.card-title {
    font-size: 1.05rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 12px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Metric Display Cards */
.metric-box {
    background: rgba(15, 23, 42, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 10px;
    padding: 14px 18px;
    margin-bottom: 10px;
}

.metric-label {
    font-size: 0.75rem;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-weight: 600;
}

.metric-value {
    font-size: 1.45rem;
    font-weight: 700;
    color: #f8fafc;
    margin-top: 4px;
}

.metric-value.emerald { color: #34d399; }
.metric-value.cyan { color: #38bdf8; }
.metric-value.amber { color: #fbbf24; }
.metric-value.rose { color: #f43f5e; }
.metric-value.purple { color: #c084fc; }

/* Decision Banners */
.decision-banner-fight {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(6, 182, 212, 0.15) 100%);
    border: 2px solid #10b981;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.decision-banner-nofight {
    background: linear-gradient(135deg, rgba(244, 63, 94, 0.2) 0%, rgba(15, 23, 42, 0.8) 100%);
    border: 2px solid #f43f5e;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

.decision-banner-escalate {
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.25) 0%, rgba(139, 92, 246, 0.2) 100%);
    border: 2px solid #f59e0b;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
}

/* Script Box (Presenter script styling) */
.script-box {
    background: rgba(30, 41, 59, 0.7);
    border-left: 4px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: 16px 20px;
    font-style: italic;
    color: #e2e8f0;
    margin: 12px 0;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Citation Tag Highlight in Evidence Narrative */
.citation-tag {
    background: rgba(6, 182, 212, 0.2);
    color: #38bdf8;
    border: 1px solid rgba(6, 182, 212, 0.4);
    border-radius: 4px;
    padding: 1px 6px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    font-weight: 500;
}

/* Formula Presentation Box */
.formula-box {
    background: rgba(15, 23, 42, 0.9);
    border: 1px dashed rgba(59, 130, 246, 0.4);
    border-radius: 10px;
    padding: 14px 20px;
    margin: 12px 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: #93c5fd;
}
</style>
"""

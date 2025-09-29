"""Theme configuration and styling for Calute UI components.

Defines application theme constants, colors, and CSS styles for the Gradio interface.
"""

APP_TITLE = "Calute Agent Assistant"
APP_SUBTITLE = "Advanced conversational AI with reasoning and tool capabilities"


COLORS = {
    "primary": "#3b82f6",
    "primary_dark": "#2563eb",
    "success": "#10b981",
    "error": "#ef4444",
    "warning": "#f59e0b",
    "surface": "#1e293b",
    "surface_light": "#334155",
    "background": "#0f172a",
    "border": "#334155",
    "text": "#f1f5f9",
    "text_secondary": "#cbd5e1",
    "text_muted": "#94a3b8",
}

CSS = f"""
.gradio-container {{
    background: {COLORS["background"]};
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
}}

.contain {{
    border-radius: 24px;
    padding: 20px;
}}

.app {{
    border-radius: 20px;
}}

#app-header {{
    background: linear-gradient(135deg, {COLORS["surface"]} 0%, {COLORS["surface_light"]} 100%);
    border-radius: 20px;
    padding: 8px;
    margin-bottom: 8px;
    border: 1px solid {COLORS["border"]};
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
}}

#app-header h1 {{
    color: {COLORS["text"]};
    margin: 0 0 8px 0;
    font-size: 20px;
    font-weight: 700;
    letter-spacing: -0.02em;
}}

#app-header p {{
    color: {COLORS["text_secondary"]};
    margin: 0;
    font-size: 14px;
    font-weight: 400;
}}

.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: {COLORS["primary"]};
    color: white;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.status-badge::before {{
    content: '';
    width: 8px;
    height: 8px;
    background: white;
    border-radius: 50%;
    animation: pulse 2s infinite;
}}

@keyframes pulse {{
    0%, 100% {{ opacity: 1; }}
    50% {{ opacity: 0.5; }}
}}

.gr-chatbot {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 20px;
    overflow: hidden;
}}

.gr-chat-message {{
    border-radius: 16px;
    margin: 12px;
    padding: 16px;
    border: 1px solid {COLORS["border"]};
    backdrop-filter: blur(10px);
}}

.gr-chat-message.user {{
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(59, 130, 246, 0.05) 100%);
    border-color: rgba(59, 130, 246, 0.3);
    margin-left: auto;
    max-width: 80%;
    border-radius: 18px 18px 6px 18px;
}}

.gr-chat-message.assistant {{
    background: {COLORS["surface_light"]};
    border-color: {COLORS["border"]};
    max-width: 90%;
    border-radius: 18px 18px 18px 6px;
}}

.gr-chat-message.assistant[data-testid*="message"] {{
    position: relative;
}}

.panel-header {{
    font-size: 13px;
    font-weight: 600;
    color: {COLORS["text_secondary"]};
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid {COLORS["border"]};
}}

.gr-chat-message[data-metadata*="panel_type"] {{
    background: linear-gradient(135deg, {COLORS["surface_light"]} 0%, {COLORS["surface"]} 100%);
    border-radius: 14px;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
    margin: 10px 12px;
}}

.gr-chat-message[data-metadata*="thinking"] {{
    border-left: 4px solid {COLORS["primary"]};
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(59, 130, 246, 0.03) 100%);
}}

.gr-chat-message[data-metadata*="tool"] {{
    border-left: 4px solid {COLORS["warning"]};
    background: linear-gradient(135deg, rgba(245, 158, 11, 0.08) 0%, rgba(245, 158, 11, 0.03) 100%);
}}

.gr-chat-message[data-metadata*="reinvoke"] {{
    border-left: 4px solid {COLORS["success"]};
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(16, 185, 129, 0.03) 100%);
}}

.status-pending {{
    color: {COLORS["warning"]};
}}

.status-done {{
    color: {COLORS["success"]};
}}

.gr-chat-message pre {{
    background: {COLORS["background"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 12px;
    padding: 12px;
    overflow-x: auto;
    margin: 8px 0;
}}

.gr-chat-message code {{
    background: rgba(59, 130, 246, 0.1);
    padding: 2px 6px;
    border-radius: 6px;
    font-size: 13px;
    color: {COLORS["text"]};
}}

.gr-textbox {{
    background: {COLORS["surface"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 16px;
    color: {COLORS["text"]};
    font-size: 14px;
    transition: all 0.2s ease;
}}

.gr-textbox:focus {{
    border-color: {COLORS["primary"]};
    box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
}}

.gr-textbox textarea {{
    color: {COLORS["text"]};
    background: transparent;
    border-radius: 14px;
}}

.gr-button {{
    border-radius: 14px;
    font-weight: 600;
    font-size: 14px;
    padding: 12px 24px;
    transition: all 0.2s ease;
    text-transform: none;
    letter-spacing: 0.02em;
}}

.gr-button.primary {{
    background: linear-gradient(135deg, {COLORS["primary"]} 0%, {COLORS["primary_dark"]} 100%);
    border: none;
    color: white;
}}

.gr-button.primary:hover {{
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}}

.gr-button.secondary {{
    background: {COLORS["surface_light"]};
    border: 1px solid {COLORS["border"]};
    color: {COLORS["text"]};
}}

.gr-button.secondary:hover {{
    background: {COLORS["surface"]};
    border-color: {COLORS["primary"]};
}}

.loading-dots {{
    display: inline-flex;
    gap: 4px;
}}

.loading-dots span {{
    width: 8px;
    height: 8px;
    background: {COLORS["primary"]};
    border-radius: 50%;
    animation: bounce 1.4s infinite ease-in-out both;
}}

.loading-dots span:nth-child(1) {{ animation-delay: -0.32s; }}
.loading-dots span:nth-child(2) {{ animation-delay: -0.16s; }}

@keyframes bounce {{
    0%, 80%, 100% {{ transform: scale(0); }}
    40% {{ transform: scale(1); }}
}}

::-webkit-scrollbar {{
    width: 10px;
    height: 10px;
}}

::-webkit-scrollbar-track {{
    background: {COLORS["background"]};
    border-radius: 10px;
}}

::-webkit-scrollbar-thumb {{
    background: {COLORS["border"]};
    border-radius: 10px;
}}

::-webkit-scrollbar-thumb:hover {{
    background: {COLORS["text_muted"]};
}}

.gr-examples {{
    border-radius: 16px;
    background: {COLORS["surface"]};
    padding: 16px;
    border: 1px solid {COLORS["border"]};
    margin-top: 20px;
}}

.gr-examples .gr-sample {{
    border-radius: 10px;
    background: {COLORS["surface_light"]};
    border: 1px solid {COLORS["border"]};
    transition: all 0.2s ease;
}}

.gr-examples .gr-sample:hover {{
    background: rgba(59, 130, 246, 0.1);
    border-color: {COLORS["primary"]};
    transform: translateY(-2px);
}}

@media (max-width: 768px) {{
    #app-header {{
        padding: 16px;
    }}

    #app-header h1 {{
        font-size: 24px;
    }}

    .gr-chat-message.user {{
        max-width: 90%;
    }}
}}
"""
CSS += f"""
/* Center the main column like ChatGPT */
#app-header,
.gr-chatbot,
.input-row,
.gr-examples {{
    max-width: 980px;
    margin-left: auto !important;
    margin-right: auto !important;
}}

.main-chatbot .wrap, .gr-chatbot {{
    max-width: 980px;
    margin: 0 auto;
}}

.gr-chat-message {{
    max-width: 820px;
    margin-left: auto;
    margin-right: auto;
}}

/* Collapsible panels inside assistant messages */
.calute-panel {{
    border: 1px solid {COLORS["border"]};
    background: linear-gradient(135deg, {COLORS["surface_light"]} 0%, {COLORS["surface"]} 100%);
    border-radius: 10px;
    overflow: hidden;
}}

.calute-panel > summary {{
    list-style: none;
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 10px 12px;
    cursor: pointer;
    user-select: none;
    font-size: 13px;
    font-weight: 600;
    color: {COLORS["text_secondary"]};
}}

.calute-panel > summary::-webkit-details-marker {{ display: none; }}

.calute-panel .dot {{
    width: 8px;
    height: 8px;
    border-radius: 999px;
    display: inline-block;
}}

.calute-panel .dot.status-pending {{ background: {COLORS["warning"]}; }}
.calute-panel .dot.status-done {{ background: {COLORS["success"]}; }}

.calute-panel .title {{
    flex: 1;
}}

.calute-panel .chevron {{
    transition: transform 0.15s ease;
    opacity: 0.7;
}}

.calute-panel[open] .chevron {{
    transform: rotate(180deg);
}}

.calute-panel[open] > .panel-body {{
    border-top: 1px solid {COLORS["border"]};
}}

.calute-panel .panel-body {{
    padding: 12px;
    color: {COLORS["text"]};
    font-size: 14px;
}}

.calute-panel .panel-empty {{
    color: {COLORS["text_muted"]};
}}

/* Make assistant 'panel' messages look flat so the panel box stands out */
.gr-chat-message.assistant[data-metadata*="panel_type"] {{
    background: transparent;
    border: none;
    padding: 6px 0;
}}

/* Tighten spacing so multiple panels look like a neat stack */
.gr-chat-message.assistant[data-metadata*="panel_type"] + .gr-chat-message.assistant[data-metadata*="panel_type"] {{
    margin-top: 6px;
}}

#composer-wrap {{
  max-width: 980px;
  margin: 12px auto 0 auto;
}}

#composer {{
  position: relative;
  background: linear-gradient(135deg, var(--surface-light, #334155) 0%, var(--surface, #1e293b) 100%);
  border: 1px solid #334155;
  border-radius: 18px;
  padding: 14px 64px 52px 16px; /* extra right/bottom for buttons */
}}

/* Make the textbox look like a plain area inside the capsule */
#composer .gr-textbox {{
  background: transparent;
  border: none;
  box-shadow: none;
}}
#composer .gr-textbox:focus {{
  border: none;
  box-shadow: none;
}}
#composer .gr-textbox textarea {{
  background: transparent;
  border: none !important;
  outline: none !important;
  color: #f1f5f9;
  padding: 8px 0;
  font-size: 14px;
  resize: vertical;
}}

/* Plugin badges at top-right of capsule */
#plugin-badges {{
  position: absolute;
  top: 10px;
  right: 12px;
  display: flex;
  gap: 6px;
}}
#plugin-badges .badge {{
  display: grid;
  place-items: center;
  width: 30px;
  height: 22px;
  border-radius: 6px;
  background: rgba(148, 163, 184, 0.12);
  border: 1px solid #334155;
  color: #d1d5db;
  font-size: 14px;
}}

/* Attach + send buttons inside the capsule (bottom-right) */
#attach-btn, #send-btn {{
  position: absolute;
  bottom: 10px;
  border-radius: 10px;
}}
#attach-btn {{ right: 64px; }}
#send-btn {{ right: 14px; }}

#attach-btn > button {{
  width: 36px;
  height: 36px;
  padding: 0;
  border-radius: 999px;
  background: transparent;
  border: 1px solid #334155;
  color: #cbd5e1;
  font-size: 18px;
}}

#send-btn > button {{
  width: 38px;
  height: 38px;
  padding: 0;
  border-radius: 999px;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%);
  color: white;
  font-weight: 800;
  font-size: 16px;
  border: none;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25);
}}
#send-btn > button:hover {{
  transform: translateY(-1px);
}}

/* Pills row and subtle clear button */
#composer-actions {{
  max-width: 980px;
  margin: 8px auto 0 auto;
  display: flex;
  gap: 8px;
  align-items: center;
}}
#composer-actions .composer-actions-spacer {{
  flex: 1;
}}
#composer-actions .gr-button {{
  border-radius: 9999px;
  padding: 8px 14px;
  font-weight: 600;
  font-size: 13px;
  background: rgba(148, 163, 184, 0.08);
  border: 1px solid #334155;
  color: #e2e8f0;
}}
#btn-deepthink > button {{
  background: rgba(59, 130, 246, 0.12);
  border-color: rgba(59, 130, 246, 0.35);
  color: #93c5fd;
}}

/* Small caption centered under the capsule */
#composer-note {{
  text-align: center;
  color: #94a3b8;
  font-size: 12px;
  margin-top: 6px;
}}

/* Keep everything centered like earlier */
#app-header,
.gr-chatbot,
#composer-wrap,
#composer-actions,
.gr-examples {{
  max-width: 980px;
  margin-left: auto !important;
  margin-right: auto !important;
}}

/* Optional: tighten input spacing on small screens */
@media (max-width: 768px) {{
  #composer {{ padding: 12px 56px 48px 12px; }}
  #attach-btn {{ right: 58px; }}
  #send-btn {{ right: 10px; }}
}}



#composer-wrap {{
  max-width: 980px;
  margin: 14px auto 0 auto;
}}

#composer {{
  position: relative;
  border-radius: 18px;
  background: linear-gradient(180deg, rgba(2, 6, 23, 0.35) 0%, rgba(2, 6, 23, 0.15) 100%);
  border: 1px solid #334155;
  padding: 14px 64px 48px 16px; /* room for right-side icons + bottom controls */
}}

/* Completely flatten the Textbox (Gradio wraps it a lot) */
#composer #message-input,
#composer #message-input * {{
  box-shadow: none !important;
}}
#composer #message-input > div,                 /* outer wrapper */
#composer #message-input .wrap,                 /* inner wrap */
#composer #message-input .container,            /* sometimes present */
#composer #message-input .gr-input,             /* component shell */
#composer #message-input .gr-box,               /* component box */
#composer #message-input .prose,                /* md wrapper some versions */
#composer #message-input .border,               /* tailwind border class */
#composer #message-input .form,                 /* form wrap */
#composer #message-input .block,                /* block wrap */
#composer #message-input .svelte-container,     /* svelte wrapper (fallback) */
#composer #message-input textarea {{
  background: transparent !important;
  border: 0 !important;
}}

/* Actual textarea look */
#composer #message-input textarea {{
  outline: none !important;
  color: #e5eefb;
  padding: 10px 8px 6px 8px;
  font-size: 14px;
  min-height: 66px;           /* size like the reference */
  resize: vertical;
}}
#composer #message-input textarea::placeholder {{
  color: #94a3b8;
}}

/* Top-right plugin badges (optional) */
#plugin-badges {{
  position: absolute;
  top: 10px;
  right: 12px;
  display: flex;
  gap: 6px;
}}
#plugin-badges .badge {{
  display: grid;
  place-items: center;
  width: 30px;
  height: 22px;
  border-radius: 6px;
  background: rgba(148, 163, 184, 0.12);
  border: 1px solid #334155;
  color: #d1d5db;
  font-size: 14px;
}}

/* Attach + Send buttons inside the capsule (bottom-right) */
#attach-btn, #send-btn {{
  position: absolute;
  bottom: 10px;
  z-index: 2;
}}
#attach-btn {{ right: 64px; }}
#send-btn   {{ right: 14px; }}

#attach-btn > button,
#attach-btn button {{
  width: 36px !important;
  height: 36px !important;
  padding: 0 !important;
  border-radius: 999px !important;
  background: rgba(148, 163, 184, 0.12) !important;
  border: 1px solid #334155 !important;
  color: #cbd5e1 !important;
  font-size: 16px !important;
}}

/* Circular arrow button */
#send-btn > button,
#send-btn button {{
  width: 38px !important;
  height: 38px !important;
  padding: 0 !important;
  border-radius: 999px !important;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
  color: #ffffff !important;
  font-weight: 800 !important;
  font-size: 16px !important;
  border: none !important;
  box-shadow: 0 4px 12px rgba(59, 130, 246, 0.25) !important;
}}
#send-btn > button:hover {{ transform: translateY(-1px); }}

/* Pills row under the capsule */
#composer-actions {{
  max-width: 980px;
  margin: 8px auto 0 auto;
  display: flex;
  gap: 8px;
  align-items: center;
}}
#composer-actions .composer-actions-spacer {{ flex: 1; }}
#composer-actions .gr-button {{
  border-radius: 9999px !important;
  padding: 8px 14px !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  background: rgba(148, 163, 184, 0.08) !important;
  border: 1px solid #334155 !important;
  color: #e2e8f0 !important;
}}
#btn-deepthink > button {{
  background: rgba(59, 130, 246, 0.12) !important;
  border-color: rgba(59, 130, 246, 0.35) !important;
  color: #93c5fd !important;
}}

/* Small centered caption */
#composer-note {{
  text-align: center;
  color: #94a3b8;
  font-size: 12px;
  margin-top: 6px;
}}

/* Keep chat and composer centered */
#app-header,
.gr-chatbot,
#composer-wrap,
#composer-actions,
.gr-examples {{
  max-width: 980px;
  margin-left: auto !important;
  margin-right: auto !important;
}}

/* Mobile adjustments */
@media (max-width: 768px) {{
  #composer {{ padding: 12px 56px 46px 12px; }}
  #attach-btn {{ right: 58px; }}
  #send-btn {{ right: 10px; }}
}}



#composer {{
  position: relative;
  border-radius: 18px;
  background: #1e293b; /* solid tone; you can swap to #223247 or any hex you like */
  border: 1px solid #334155;
  padding: 14px 64px 48px 16px;
}}

/* 2) Absolutely flatten the Textbox: no inner box, no extra background */
#composer #message-input,
#composer #message-input * {{
  box-shadow: none !important;
}}
#composer #message-input > div,
#composer #message-input .wrap,
#composer #message-input .container,
#composer #message-input .gr-input,
#composer #message-input .gr-box,
#composer #message-input .prose,
#composer #message-input .border,
#composer #message-input .form,
#composer #message-input .block,
#composer #message-input .svelte-container,
#composer [data-testid="textbox"],
#composer [data-testid="textbox"] > div,
#composer [data-testid="textbox"] .wrap,
#composer [data-testid="textbox"] .container {{
  background: transparent !important;
  border: 0 !important;
  box-shadow: none !important;
}}

/* 3) Textarea itself is transparent so it blends into the capsule */
#composer #message-input textarea,
#composer #message-input textarea:focus {{
  background: transparent !important;
  border: 0 !important;
  outline: none !important;
  color: #e5eefb;
  padding: 10px 8px 6px 8px;
  font-size: 14px;
  min-height: 66px;
  resize: vertical;
  caret-color: #e5eefb;
}}
#composer #message-input textarea::placeholder {{ color: #94a3b8; }}

/* 4) Controls on the right (unchanged) */
#attach-btn, #send-btn {{ position: absolute; bottom: 10px; z-index: 2; }}
#attach-btn {{ right: 64px; }}
#send-btn   {{ right: 14px; }}

#attach-btn > button, #attach-btn button {{
  width: 36px !important; height: 36px !important; padding: 0 !important;
  border-radius: 999px !important;
  background: rgba(148,163,184,0.12) !important;
  border: 1px solid #334155 !important; color: #cbd5e1 !important; font-size: 16px !important;
}}
#send-btn > button, #send-btn button {{
  width: 38px !important; height: 38px !important; padding: 0 !important;
  border-radius: 999px !important;
  background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
  color: #fff !important; font-weight: 800 !important; font-size: 16px !important; border: none !important;
  box-shadow: 0 4px 12px rgba(59,130,246,0.25) !important;
}}
#send-btn > button:hover {{ transform: translateY(-1px); }}

/* Optional badges (keep if you use them) */
#plugin-badges {{ position: absolute; top: 10px; right: 12px; display: flex; gap: 6px; }}
#plugin-badges .badge {{
  display: grid; place-items: center; width: 30px; height: 22px; border-radius: 6px;
  background: rgba(148,163,184,0.12); border: 1px solid #334155; color: #d1d5db; font-size: 14px;
}}

/* Center the composer with the chat */
#composer-wrap, #composer-actions, .gr-chatbot, #app-header, .gr-examples {{
  max-width: 980px; margin-left: auto !important; margin-right: auto !important;
}}

/* Mobile tweak */
@media (max-width: 768px) {{
  #composer {{ padding: 12px 56px 46px 12px; }}
  #attach-btn {{ right: 58px; }}
  #send-btn {{ right: 10px; }}
}}
"""

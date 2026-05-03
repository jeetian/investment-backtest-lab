from __future__ import annotations

from html import escape


def render_html_head(*, title: str, plotly: bool = False) -> str:
    plotly_script = (
        '<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>' if plotly else ""
    )
    return f"""<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link
    href="https://fonts.googleapis.com/css2?family=Noto+Sans+TC:wght@400;500;700&family=Noto+Sans+JP:wght@400;500;700&display=swap"
    rel="stylesheet">
  {plotly_script}
  <style>
    :root {{
      --paper: #f7f5ef;
      --surface: #fffffc;
      --surface-soft: #fbfaf6;
      --ink: #222420;
      --muted: #66706a;
      --line: #d9d6cb;
      --indigo: #53677f;
      --sage: #6f8574;
      --copper: #b8794f;
      --danger: #9b4b45;
      --warning-bg: #fff8f3;
      --ok-bg: #f3f8f1;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--paper);
      color: var(--ink);
      font-family: "Noto Sans TC", "Noto Sans JP", system-ui, sans-serif;
      line-height: 1.55;
    }}
    main {{ max-width: 1440px; margin: 0 auto; padding: 28px; }}
    h1, h2, h3 {{ margin: 0; letter-spacing: 0; }}
    h1 {{ font-size: 30px; line-height: 1.25; }}
    h2 {{ font-size: 21px; margin-bottom: 12px; }}
    h3 {{ font-size: 16px; }}
    p {{ margin: 8px 0; }}
    .hero {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 16px;
      align-items: stretch;
      margin-bottom: 16px;
    }}
    .panel {{
      background: var(--surface);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      box-shadow: 0 10px 22px rgba(47, 47, 43, 0.04);
      margin-bottom: 16px;
    }}
    .panel.compact {{ padding: 14px; }}
    .eyebrow {{
      text-transform: uppercase;
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      margin-bottom: 6px;
    }}
    .lede {{ color: var(--muted); max-width: 880px; }}
    .note {{
      border-left: 4px solid var(--copper);
      background: var(--warning-bg);
      padding: 12px 14px;
      border-radius: 6px;
      color: #47372e;
      margin-top: 12px;
    }}
    .kpis, .cards {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin-top: 14px;
    }}
    .kpi, .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--surface-soft);
    }}
    .kpi span, .card span {{ display: block; color: var(--muted); font-size: 12px; }}
    .kpi strong, .card strong {{ display: block; margin-top: 4px; font-size: 18px; }}
    .badge, .pill {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 5px 10px;
      margin: 6px 6px 0 0;
      background: var(--surface);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }}
    .badge.ok, .pill.ok {{ color: var(--sage); background: var(--ok-bg); }}
    .badge.warn, .pill.warn, .badge.danger, .pill.danger {{
      color: var(--danger);
      border-color: #d9b89c;
      background: var(--warning-bg);
    }}
    .two-col {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
    }}
    .steps {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      margin-top: 12px;
    }}
    .step {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      background: var(--surface-soft);
    }}
    .step b {{ color: var(--indigo); }}
    .weight {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      background: var(--surface-soft);
    }}
    .weight strong {{ font-size: 24px; }}
    .allocation-bar {{
      display: flex;
      width: 100%;
      min-height: 22px;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface-soft);
      margin: 12px 0;
    }}
    .allocation-segment {{
      min-width: 2px;
      color: white;
      font-size: 11px;
      font-weight: 700;
      display: flex;
      align-items: center;
      justify-content: center;
      white-space: nowrap;
    }}
    .allocation-segment.qqq {{ background: var(--indigo); }}
    .allocation-segment.qld {{ background: var(--sage); }}
    .allocation-segment.tqqq {{ background: var(--copper); }}
    .allocation-segment.cash {{ background: #7d766b; }}
    .checklist {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .check.ok strong {{ color: var(--sage); }}
    .check.warn strong {{ color: var(--danger); }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ padding: 8px 10px; border-bottom: 1px solid var(--line); text-align: left; }}
    th {{ color: var(--muted); font-weight: 700; background: var(--surface-soft); }}
    .links a, a {{
      color: var(--indigo);
      margin-right: 14px;
      text-decoration: none;
      font-weight: 700;
    }}
    .compare-grid {{
      display: grid;
      grid-template-columns: 320px 1fr;
      gap: 16px;
    }}
    .controls {{
      max-height: 520px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      background: var(--surface-soft);
    }}
    .check {{ display: block; margin: 8px 0; color: var(--ink); }}
    .metric-buttons button {{
      border: 1px solid var(--line);
      background: var(--surface);
      border-radius: 6px;
      padding: 7px 10px;
      margin: 0 6px 8px 0;
      cursor: pointer;
    }}
    .metric-buttons button.active {{
      background: var(--indigo);
      color: white;
      border-color: var(--indigo);
    }}
    #compare-chart {{ width: 100%; height: 520px; }}
    @media (max-width: 900px) {{
      main {{ padding: 16px; }}
      .hero, .kpis, .cards, .two-col, .steps, .compare-grid, .checklist {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>"""


def risk_badge(label: str, ok: bool) -> str:
    css = "ok" if ok else "danger"
    return f'<span class="badge {css}">{escape(label)}</span>'

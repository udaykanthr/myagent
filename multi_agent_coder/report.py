"""
HTML Report Generator — produces self-contained HTML reports after pipeline runs.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime
import html


@dataclass
class StepReport:
    """Data for a single step in the report."""
    index: int
    text: str
    step_type: str = "?"
    status: str = "pending"  # done/failed/skipped/pending
    diffs: list[str] = field(default_factory=list)
    tokens_sent: int = 0
    tokens_recv: int = 0


_STATUS_COLORS = {
    "done": "#22c55e",
    "failed": "#ef4444",
    "skipped": "#94a3b8",
    "pending": "#64748b",
}

_STATUS_ICONS = {
    "done": "✔",
    "failed": "✘",
    "skipped": "–",
    "pending": "○",
}


def _escape(text: str) -> str:
    return html.escape(text)


def _diff_to_html(diff_text: str) -> str:
    """Convert a unified diff to syntax-colored HTML."""
    lines: list[str] = []
    for line in diff_text.splitlines():
        escaped = _escape(line)
        if line.startswith("+++") or line.startswith("---"):
            lines.append(f'<span class="diff-meta">{escaped}</span>')
        elif line.startswith("@@"):
            lines.append(f'<span class="diff-hunk">{escaped}</span>')
        elif line.startswith("+"):
            lines.append(f'<span class="diff-add">{escaped}</span>')
        elif line.startswith("-"):
            lines.append(f'<span class="diff-del">{escaped}</span>')
        else:
            lines.append(escaped)
    return "\n".join(lines)


def generate_html_report(
    task: str,
    steps: list[StepReport],
    token_usage: dict,
    pipeline_success: bool,
    output_dir: str = ".agentchanti/reports",
) -> str:
    """Generate a self-contained HTML report file.

    Returns the path to the generated report.
    """
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"report_{timestamp}.html"
    filepath = os.path.join(output_dir, filename)

    done = sum(1 for s in steps if s.status == "done")
    failed = sum(1 for s in steps if s.status == "failed")
    skipped = sum(1 for s in steps if s.status == "skipped")
    # Use token_usage dict (global tracker) as the authoritative source for
    # dashboard totals; fall back to summing per-step values.
    total_sent = token_usage.get("sent", 0) or sum(s.tokens_sent for s in steps)
    total_recv = token_usage.get("recv", 0) or sum(s.tokens_recv for s in steps)

    # Build step HTML
    steps_html = ""
    for step in steps:
        color = _STATUS_COLORS.get(step.status, "#64748b")
        icon = _STATUS_ICONS.get(step.status, "○")
        diffs_html = ""
        if step.diffs:
            diffs_content = "\n".join(_diff_to_html(d) for d in step.diffs)
            diffs_html = f'<pre class="diff-block">{diffs_content}</pre>'

        steps_html += f"""
        <div class="step" style="border-left: 3px solid {color};">
            <div class="step-header">
                <span class="step-icon" style="color: {color};">{icon}</span>
                <span class="step-type">[{_escape(step.step_type)}]</span>
                <span class="step-text">{_escape(step.text)}</span>
                <span class="step-tokens">{step.tokens_sent + step.tokens_recv} tokens</span>
            </div>
            {diffs_html}
        </div>
        """

    status_class = "success" if pipeline_success else "failure"
    status_text = "SUCCESS" if pipeline_success else "FAILED"

    report_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>AgentChanti Report — {_escape(task[:60])}</title>
<style>
  :root {{ --bg: #0f172a; --card: #1e293b; --text: #e2e8f0; --muted: #94a3b8;
           --accent: #3b82f6; --success: #22c55e; --failure: #ef4444; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: 'Inter', 'Segoe UI', sans-serif; background: var(--bg);
          color: var(--text); line-height: 1.6; padding: 2rem; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ color: var(--accent); font-size: 1.5rem; margin-bottom: 0.5rem; }}
  .timestamp {{ color: var(--muted); font-size: 0.875rem; margin-bottom: 1.5rem; }}

  .dashboard {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                gap: 1rem; margin-bottom: 2rem; }}
  .stat {{ background: var(--card); border-radius: 8px; padding: 1rem; text-align: center; }}
  .stat-value {{ font-size: 1.5rem; font-weight: 700; }}
  .stat-label {{ color: var(--muted); font-size: 0.75rem; text-transform: uppercase; }}

  .status-badge {{ display: inline-block; padding: 0.25rem 0.75rem; border-radius: 4px;
                   font-weight: 600; font-size: 0.875rem; margin-bottom: 1.5rem; }}
  .success {{ background: rgba(34, 197, 94, 0.2); color: var(--success); }}
  .failure {{ background: rgba(239, 68, 68, 0.2); color: var(--failure); }}

  .task-desc {{ background: var(--card); border-radius: 8px; padding: 1rem;
                margin-bottom: 1.5rem; font-style: italic; color: var(--muted); }}

  .step {{ background: var(--card); border-radius: 8px; padding: 1rem;
           margin-bottom: 0.75rem; }}
  .step-header {{ display: flex; align-items: center; gap: 0.5rem; flex-wrap: wrap; }}
  .step-icon {{ font-size: 1.1rem; }}
  .step-type {{ color: var(--accent); font-size: 0.8rem; font-weight: 600; }}
  .step-text {{ flex: 1; min-width: 200px; }}
  .step-tokens {{ color: var(--muted); font-size: 0.75rem; white-space: nowrap; }}

  .diff-block {{ background: #0d1117; border-radius: 6px; padding: 1rem;
                 margin-top: 0.75rem; overflow-x: auto; font-family: 'Consolas', monospace;
                 font-size: 0.8rem; line-height: 1.4; }}
  .diff-add {{ color: #22c55e; }}
  .diff-del {{ color: #ef4444; }}
  .diff-hunk {{ color: #60a5fa; }}
  .diff-meta {{ color: #e2e8f0; font-weight: 600; }}

  .footer {{ text-align: center; color: var(--muted); font-size: 0.75rem;
             margin-top: 2rem; padding-top: 1rem; border-top: 1px solid #334155; }}
</style>
</head>
<body>
<div class="container">
  <h1>🤖 AgentChanti Report</h1>
  <p class="timestamp">Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

  <span class="status-badge {status_class}">{status_text}</span>

  <div class="task-desc">{_escape(task)}</div>

  <div class="dashboard">
    <div class="stat">
      <div class="stat-value">{len(steps)}</div>
      <div class="stat-label">Total Steps</div>
    </div>
    <div class="stat">
      <div class="stat-value" style="color: var(--success);">{done}</div>
      <div class="stat-label">Completed</div>
    </div>
    <div class="stat">
      <div class="stat-value" style="color: var(--failure);">{failed}</div>
      <div class="stat-label">Failed</div>
    </div>
    <div class="stat">
      <div class="stat-value">{total_sent + total_recv:,}</div>
      <div class="stat-label">Total Tokens</div>
    </div>
    <div class="stat">
      <div class="stat-value">{total_sent:,}</div>
      <div class="stat-label">Sent</div>
    </div>
    <div class="stat">
      <div class="stat-value">{total_recv:,}</div>
      <div class="stat-label">Received</div>
    </div>
    {f'''<div class="stat">
      <div class="stat-value" style="color: var(--accent);">${token_usage.get("cost", 0.0):.4f}</div>
      <div class="stat-label">Total Cost</div>
    </div>''' if token_usage.get("cost", 0) > 0 else ""}
  </div>

  <h2 style="margin-bottom: 1rem; font-size: 1.1rem;">Steps</h2>
  {steps_html}

  <div class="footer">
    AgentChanti — Multi-Agent Local Coder
  </div>
</div>
</body>
</html>"""

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(report_html)

    return filepath

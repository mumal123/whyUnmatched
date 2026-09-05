"""A dependency-free local demo dashboard for WhyUnmatched.

Run ``python dashboard.py`` and open the printed localhost URL. The server is
read-only: it only reads reconciliation CSV/JSON files and never exposes .env.
"""

from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = ROOT / "data"


def _number(value: object) -> float:
    try:
        return float(value) if pd.notna(value) else 0.0
    except (TypeError, ValueError):
        return 0.0


def dashboard_data(data_dir: Path) -> dict:
    """Build a JSON-safe snapshot from reconciliation outputs."""
    final_path = data_dir / "final_reconciliation.csv"
    if not final_path.exists():
        return {"error": f"No reconciliation output at {final_path}. Run run_pipeline.py first."}

    final = pd.read_csv(final_path)
    exceptions = final[final["status"] != "MATCH"].copy()
    for column in ("bank_amount", "settlement_amount"):
        if column not in exceptions:
            exceptions[column] = 0.0
        exceptions[column] = pd.to_numeric(exceptions[column], errors="coerce")
    exceptions["amount_at_risk"] = exceptions[["bank_amount", "settlement_amount"]].max(axis=1).fillna(0.0)
    exceptions["record_id"] = exceptions["settlement_id"].fillna(exceptions["bank_statement_id"]).fillna("—")

    priority_path = data_dir / "priority_queue.csv"
    priority = pd.read_csv(priority_path) if priority_path.exists() else pd.DataFrame()
    priority_records = priority.head(10).where(pd.notna(priority), None).to_dict(orient="records")

    by_status = (
        exceptions["status"].value_counts().rename_axis("status").reset_index(name="count")
        .to_dict(orient="records")
    )
    exception_records = exceptions[
        ["record_id", "status", "bank_amount", "settlement_amount", "amount_at_risk"]
    ].sort_values("amount_at_risk", ascending=False).head(100)
    exception_records = exception_records.fillna(0.0)
    exception_records = exception_records.where(pd.notna(exception_records), None).to_dict(orient="records")

    return {
        "metrics": {
            "records_processed": int(len(final)),
            "matched": int((final["status"] == "MATCH").sum()),
            "exceptions": int(len(exceptions)),
            "exposure": round(float(exceptions["amount_at_risk"].sum()), 2),
        },
        "by_status": by_status,
        "exceptions": exception_records,
        "priority_queue": priority_records,
    }


PAGE = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WhyUnmatched | Reconciliation demo</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--card:#151d33;--line:#283454;--text:#eef2ff;--muted:#a8b4d4;--purple:#8067ff;--red:#fb7185;--green:#34d399}
*{box-sizing:border-box}body{margin:0;background:linear-gradient(125deg,#0b1020,#121a31);color:var(--text);font:15px system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1200px;margin:auto;padding:42px 22px}h1{margin:0;font-size:30px}h1 span{color:#a99bff}p{color:var(--muted)}.top{display:flex;justify-content:space-between;align-items:start;gap:16px}.pill{border:1px solid var(--line);padding:8px 12px;border-radius:999px;color:var(--muted);font-size:12px}
.metrics{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:28px 0}.card,.panel{background:rgba(21,29,51,.92);border:1px solid var(--line);border-radius:14px}.card{padding:18px}.label{font-size:12px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted)}.value{font-size:27px;font-weight:700;margin-top:7px}.red{color:var(--red)}.green{color:var(--green)}
.grid{display:grid;grid-template-columns:1fr 2fr;gap:18px}.panel{padding:18px;overflow:auto}h2{font-size:16px;margin:0 0 14px}.barrow{display:grid;grid-template-columns:150px 1fr 36px;gap:10px;align-items:center;margin:11px 0;color:var(--muted);font-size:13px}.bar{height:8px;background:#25304d;border-radius:99px;overflow:hidden}.fill{height:100%;background:linear-gradient(90deg,#8067ff,#b9a9ff);border-radius:99px}
table{border-collapse:collapse;width:100%;font-size:13px}th{text-align:left;color:var(--muted);font-weight:500;border-bottom:1px solid var(--line);padding:8px}td{padding:10px 8px;border-bottom:1px solid #222d49;white-space:nowrap}.tag{border-radius:99px;padding:4px 8px;background:#2c2949;color:#d7d0ff;font-size:11px}input{background:#0d1427;border:1px solid var(--line);border-radius:8px;color:var(--text);padding:9px 11px;width:100%;margin-bottom:10px}
#error{display:none;background:#4a1d31;padding:14px;border-radius:9px;color:#fecdd3}.empty{color:var(--muted);padding:12px 0}@media(max-width:800px){.metrics,.grid{grid-template-columns:1fr 1fr}.grid{display:block}.panel{margin-bottom:18px}}@media(max-width:520px){.metrics{grid-template-columns:1fr}.top{display:block}.pill{display:inline-block;margin-top:12px}}
</style></head><body><main>
<div class="top"><div><h1>Why<span>Unmatched</span></h1><p>Reconciliation exceptions, prioritised for action.</p></div><div class="pill">Read-only demo dashboard</div></div><div id="error"></div>
<section class="metrics"><div class="card"><div class="label">Records processed</div><div class="value" id="processed">—</div></div><div class="card"><div class="label">Matched</div><div class="value green" id="matched">—</div></div><div class="card"><div class="label">Exceptions</div><div class="value red" id="exceptions-count">—</div></div><div class="card"><div class="label">Exposure</div><div class="value red" id="exposure">—</div></div></section>
<section class="grid"><div class="panel"><h2>Exception mix</h2><div id="bars"></div></div><div class="panel"><h2>Priority queue <span class="label">(AI-enriched when available)</span></h2><div id="priority"></div></div></section>
<section class="panel" style="margin-top:18px"><h2>Exceptions <span class="label">Top 100 by exposure</span></h2><input id="filter" placeholder="Filter by status or record ID"><div id="exception-table"></div></section>
</main><script>
const money=n=>new Intl.NumberFormat('en-IN',{style:'currency',currency:'INR',maximumFractionDigits:0}).format(n||0);
const escape=s=>String(s??'—').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
let exceptions=[];
function table(rows, cols){if(!rows.length)return '<div class="empty">No records available.</div>';return '<table><thead><tr>'+cols.map(c=>'<th>'+c[0]+'</th>').join('')+'</tr></thead><tbody>'+rows.map(r=>'<tr>'+cols.map(c=>'<td>'+c[1](r)+'</td>').join('')+'</tr>').join('')+'</tbody></table>'}
function renderExceptions(){let q=document.querySelector('#filter').value.toLowerCase();let rows=exceptions.filter(r=>(r.status+' '+r.record_id).toLowerCase().includes(q));document.querySelector('#exception-table').innerHTML=table(rows,[['Record ID',r=>escape(r.record_id)],['Status',r=>'<span class="tag">'+escape(r.status)+'</span>'],['Bank credit',r=>money(r.bank_amount)],['Settlement',r=>money(r.settlement_amount)],['At risk',r=>money(r.amount_at_risk)]]);}
fetch('/api/dashboard').then(r=>r.json()).then(d=>{const errorEl=document.querySelector('#error');if(d.error){errorEl.textContent=d.error;errorEl.style.display='block';return}let m=d.metrics;document.querySelector('#processed').textContent=m.records_processed.toLocaleString('en-IN');document.querySelector('#matched').textContent=m.matched.toLocaleString('en-IN');document.querySelector('#exceptions-count').textContent=m.exceptions.toLocaleString('en-IN');document.querySelector('#exposure').textContent=money(m.exposure);let max=Math.max(...d.by_status.map(x=>x.count),1);document.querySelector('#bars').innerHTML=d.by_status.map(x=>'<div class="barrow"><span>'+escape(x.status)+'</span><div class="bar"><div class="fill" style="width:'+(100*x.count/max)+'%"></div></div><b>'+x.count+'</b></div>').join('');document.querySelector('#priority').innerHTML=table(d.priority_queue,[['Rank',r=>escape(r.priority_rank)],['Exception',r=>escape(r.exception_id)],['Severity',r=>'<span class="tag">'+escape(r.severity)+'</span>'],['At risk',r=>money(r.amount_at_risk)],['Action',r=>escape(r.policy_action)]]);exceptions=d.exceptions;renderExceptions();}).catch(e=>{const errorEl=document.querySelector('#error');errorEl.textContent='Could not load dashboard data: '+e;errorEl.style.display='block'});document.querySelector('#filter').addEventListener('input',renderExceptions);
</script></body></html>'''


def make_handler(data_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            path = urlparse(self.path).path
            if path == "/api/dashboard":
                payload = json.dumps(dashboard_data(data_dir), allow_nan=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
            elif path in {"/", "/index.html"}:
                payload = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            else:
                self.send_error(404, "Not found")
                return
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_: object) -> None:
            return  # Keep demo output clean.
    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the WhyUnmatched demo dashboard")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(args.data_dir.resolve()))
    print(f"Dashboard: http://127.0.0.1:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

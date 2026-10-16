import base64, os, sys
from playwright.sync_api import sync_playwright
auth = base64.b64encode(f"{os.environ['GU']}:{os.environ['GP']}".encode()).decode()
out = sys.argv[1]
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1600, "height": 1100}, extra_http_headers={"Authorization": f"Basic {auth}"})
    pg = ctx.new_page()
    pg.goto("http://127.0.0.1:3000/d/regrag-queries?from=now-30d&to=now&kiosk&theme=light", wait_until="networkidle")
    pg.wait_for_timeout(6000)
    pg.screenshot(path=out, full_page=True)
    for el in pg.locator("[data-testid='data-testid panel content'], section[data-testid*='Panel header']").all()[:0]: pass
    print(pg.evaluate("[...document.querySelectorAll('[data-viz-panel-key],[data-panelid],section')].map(e=>{const r=e.getBoundingClientRect();return [e.getAttribute('aria-label')||e.querySelector('h2,h6')?.innerText||'', Math.round(r.x),Math.round(r.y),Math.round(r.width),Math.round(r.height)]})"))
    b.close()

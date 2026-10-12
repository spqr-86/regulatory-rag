import json, time
from playwright.sync_api import sync_playwright

OUT = "/tmp/claude-1000/-home-petr/5f753603-a698-4067-9c91-c436c3b5ef07/scratchpad/vid"
Q = "Действует ли у нас инструкция для складских помещений и когда испытывать наружную пожарную лестницу?"
marks = {}
with sync_playwright() as p:
    b = p.chromium.launch()
    ctx = b.new_context(viewport={"width": 1280, "height": 800},
                        record_video_dir=OUT, record_video_size={"width": 1280, "height": 800})
    pg = ctx.new_page()
    t0 = time.time()
    pg.goto("http://127.0.0.1:8502/", wait_until="networkidle"); pg.wait_for_timeout(2500)
    pg.add_style_tag(content="header[data-testid=stHeader]{display:none!important}")
    pg.get_by_text("Закон + ЛНА для объекта", exact=True).click(); pg.wait_for_timeout(1200)
    pg.get_by_label("Подразделение").click(); pg.wait_for_timeout(700)
    pg.get_by_role("option").filter(has_text="Склад").first.click(); pg.wait_for_timeout(1200)
    pg.get_by_label("Вопрос").click()
    pg.get_by_label("Вопрос").press_sequentially(Q, delay=35); pg.wait_for_timeout(800)
    pg.get_by_role("button", name="Спросить").click()
    marks["wait_start"] = time.time() - t0 + 1.5
    pg.wait_for_timeout(3000)
    t = time.time()
    while time.time() - t < 300:
        busy = "Обработка запроса" in pg.inner_text("body") or pg.locator("[data-testid=stStatusWidget]").count() > 0
        if not busy: break
        pg.wait_for_timeout(500)
    marks["wait_end"] = time.time() - t0
    pg.wait_for_timeout(2500)
    pg.mouse.move(1270, 60)
    for _ in range(400):
        pg.mouse.wheel(0, 20); pg.wait_for_timeout(100)
        at_bottom = pg.evaluate("(()=>{const e=document.querySelector('[data-testid=stMain]')||document.scrollingElement;return e.scrollTop+e.clientHeight>=e.scrollHeight-2})()")
        if at_bottom: break
    pg.wait_for_timeout(3000)
    marks["end"] = time.time() - t0
    path = pg.video.path()
    ctx.close(); b.close()
    marks["video"] = path
print(json.dumps(marks))

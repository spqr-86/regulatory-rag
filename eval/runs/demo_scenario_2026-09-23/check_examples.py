import json, sys, time
from playwright.sync_api import sync_playwright
OUT = sys.argv[1]
QS = ["Нужен ли на объекте план эвакуации и сколько огнетушителей?",
      "Как часто проверять огнезащитную обработку конструкций?",
      "Как часто проводить тренировки по эвакуации?"]
MODES = ["Закон для объекта", "ЛНА для объекта", "Закон + ЛНА для объекта"]
res = []
with sync_playwright() as p:
    b = p.chromium.launch()
    for q in QS:
        for mode in MODES:
            pg = b.new_page(viewport={"width": 1400, "height": 1000})
            pg.goto("http://127.0.0.1:8502/", wait_until="networkidle"); pg.wait_for_timeout(3000)
            pg.get_by_text(mode, exact=True).click(); pg.wait_for_timeout(1500)
            pg.get_by_label("Подразделение").click(); pg.wait_for_timeout(500)
            pg.get_by_role("option").filter(has_text="Офис").first.click(); pg.wait_for_timeout(1500)
            pg.get_by_label("Вопрос").fill(q); pg.wait_for_timeout(500)
            pg.get_by_role("button", name="Спросить").click()
            t = time.time(); pg.wait_for_timeout(3000)
            while time.time() - t < 300:
                busy = "Обработка запроса" in pg.inner_text("body") or pg.locator("[data-testid=stStatusWidget]").count() > 0
                if not busy: break
                pg.wait_for_timeout(2000)
            pg.wait_for_timeout(2000)
            body = pg.inner_text("[data-testid=stMain]") if pg.locator("[data-testid=stMain]").count() else pg.inner_text("body")
            r = {"q": q, "mode": mode, "s": round(time.time() - t), "text": body}
            res.append(r); print(mode, "|", q, "|", r["s"], "s", flush=True)
            json.dump(res, open(OUT, "w"), ensure_ascii=False, indent=1)
            pg.close()
    b.close()

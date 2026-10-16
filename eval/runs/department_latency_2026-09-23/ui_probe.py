"""One UI request on the unit-scoped screen, wall-clock from click to answer (#67)."""
import sys, time
from playwright.sync_api import sync_playwright

UNIT = sys.argv[1] if len(sys.argv) > 1 else "Склад"
Q = sys.argv[2] if len(sys.argv) > 2 else (
    "Действует ли у нас инструкция для складских помещений и когда испытывать наружную пожарную лестницу?")
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1280, "height": 800})
    pg.goto("http://127.0.0.1:8502/", wait_until="networkidle"); pg.wait_for_timeout(2500)
    pg.get_by_text("Закон + ЛНА для объекта", exact=True).click(); pg.wait_for_timeout(1200)
    pg.get_by_label("Подразделение").click(); pg.wait_for_timeout(700)
    pg.get_by_role("option").filter(has_text=UNIT).first.click(); pg.wait_for_timeout(1200)
    pg.get_by_label("Вопрос").fill(Q); pg.wait_for_timeout(500)
    pg.get_by_role("button", name="Спросить").click()
    t0 = time.time(); pg.wait_for_timeout(1500)
    while time.time() - t0 < 300:
        busy = "Обработка запроса" in pg.inner_text("body") or pg.locator("[data-testid=stStatusWidget]").count() > 0
        if not busy: break
        pg.wait_for_timeout(250)
    print(f"ui_wall_s={time.time() - t0:.1f}")
    b.close()

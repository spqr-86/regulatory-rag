import sys, time
from playwright.sync_api import sync_playwright
SC = [
 ("01-generic-internship", "Общая нормативная база", None, "Какова минимальная продолжительность стажировки на рабочем месте?"),
 ("02-generic-microenterprise", "Общая нормативная база", None, "Может ли на микропредприятии проверка знаний проводиться без комиссии?"),
 ("03-department-depot", "Закон + ЛНА для объекта", "Склад", "Действует ли у нас инструкция для складских помещений и когда испытывать наружную пожарную лестницу?"),
 ("04-department-office", "Закон + ЛНА для объекта", "Офис", "Сколько огнетушителей должно быть на объекте и нужен ли план эвакуации?"),
]
only = sys.argv[1:] 
with sync_playwright() as p:
    b = p.chromium.launch()
    for name, mode, unit, q in SC:
        if only and name not in only: continue
        pg = b.new_page(viewport={"width":1400,"height":1000})
        pg.goto("http://127.0.0.1:8502/", wait_until="networkidle"); pg.wait_for_timeout(3000)
        pg.get_by_text(mode, exact=True).click(); pg.wait_for_timeout(1500)
        if unit:
            pg.get_by_label("Подразделение").click(); pg.wait_for_timeout(500)
            pg.get_by_role("option").filter(has_text=unit).first.click(); pg.wait_for_timeout(1500)
        pg.get_by_label("Вопрос").fill(q); pg.wait_for_timeout(500)
        pg.get_by_role("button", name="Спросить").click()
        t=time.time(); pg.wait_for_timeout(3000)
        while time.time()-t < 300:
            busy = "Обработка запроса" in pg.inner_text("body") or pg.locator("[data-testid=stStatusWidget]").count() > 0
            if not busy: break
            pg.wait_for_timeout(2000)
        pg.wait_for_timeout(3000)
        pg.add_style_tag(content="header[data-testid=stHeader]{display:none!important}")
        h = pg.evaluate("Math.max(...[...document.querySelectorAll('*')].map(e=>e.scrollHeight))")
        pg.set_viewport_size({"width":1400,"height":int(h)+40}); pg.wait_for_timeout(1500)
        pg.screenshot(path=f"docs/assets/demo/{name}.png", full_page=True)
        print(name, round(time.time()-t), "s", flush=True)
        pg.close()
    b.close()

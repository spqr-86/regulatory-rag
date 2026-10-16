"""Прогон 6 вопросов показа теми же вызовами, что app.py (без telemetry writer)."""
import json
import sys
import time

sys.path.insert(0, ".")
from dotenv import load_dotenv

load_dotenv()
from src.infra.llm_factory import apply_ipv6_patch_for_googleapis

apply_ipv6_patch_for_googleapis()
from src import ui_generic
from src.department_qa.contract import RequestContext
from src.department_qa.service import answer_scoped_question
from src.department_qa.wiring import build_department_stack

OUT = sys.argv[1]
GENERIC = [
    "Какова минимальная продолжительность стажировки на рабочем месте?",
    "Может ли на микропредприятии проверка знаний проводиться без комиссии?",
    "Из скольких человек должна состоять комиссия по проверке знаний?",
    "Какие существуют виды инструктажей по охране труда?",
]
DEPT = [
    ("unit_depot", "Действует ли у нас инструкция для складских помещений и когда испытывать наружную пожарную лестницу?"),
    ("unit_office", "Сколько огнетушителей должно быть на объекте и нужен ли план эвакуации?"),
]

results = []
graph = ui_generic.load_generic_graph()
for q in GENERIC:
    t = time.perf_counter()
    result, qid = ui_generic.answer_generic_question(graph, q, writer=None)
    results.append({"mode": "generic", "question": q, "latency_s": round(time.perf_counter() - t, 1),
                    "result": result})
    print("generic done", q, flush=True)

stack = build_department_stack()
kw = {"known_units": stack.known_units, "snapshot_id": stack.manifest.snapshot_id,
      "profiles": stack.config.profiles}
if stack.service_limits is not None:
    kw["limits"] = stack.service_limits
for unit, q in DEPT:
    ctx = RequestContext(corpora=("external", "internal"), unit_id=unit,
                         include_object_profile=stack.config.profiles.get(unit) is not None)
    t = time.perf_counter()
    resp = answer_scoped_question(q, ctx, stack.retrieve_fn, stack.model_fn, writer=None, **kw)
    results.append({"mode": "department", "unit": unit, "question": q,
                    "latency_s": round(time.perf_counter() - t, 1),
                    "result": resp.model_dump(mode="json")})
    print("dept done", unit, flush=True)

json.dump(results, open(OUT, "w"), ensure_ascii=False, indent=1, default=str)

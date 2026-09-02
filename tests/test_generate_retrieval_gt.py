"""Unit tests for the retrieval ground-truth generator (eval/generate_retrieval_gt.py).

Covers the deterministic, LLM-free core: junk-chunk filtering, question
normalisation, exact + near-duplicate dedup, structured-output parsing and
GT-record construction. The LLM call and Chroma iteration are exercised
elsewhere (integration).
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import pytest

from eval.generate_retrieval_gt import (
    Questions,
    is_junk_chunk,
    normalize_question,
    parse_questions,
    dedup_questions,
    build_gt_record,
    to_passage,
    iter_corpus_chunks,
    _passage_identity,
)

pytestmark = pytest.mark.unit


class TestIsJunkChunk:
    def test_too_short(self):
        assert is_junk_chunk("Статья 5.") is True

    def test_normal_chunk_kept(self):
        text = (
            "Работодатель обязан обеспечить обучение по охране труда и проверку "
            "знания требований охраны труда в порядке, установленном Правительством "
            "Российской Федерации, с учётом мнения профсоюзного органа. " * 2
        )
        assert is_junk_chunk(text) is False

    def test_table_of_contents(self):
        toc = (
            "Содержание\n"
            "1. Общие положения ...... 3\n"
            "2. Требования к обучению ...... 7\n"
            "3. Порядок проверки знаний ...... 12\n"
            "4. Заключительные положения ...... 18\n"
        )
        assert is_junk_chunk(toc) is True


class TestNormalizeQuestion:
    def test_lowercase_and_strip_punctuation(self):
        assert normalize_question("  Кто проходит обучение по программе А?? ") == (
            "кто проходит обучение по программе а"
        )

    def test_collapses_whitespace(self):
        assert normalize_question("что   такое  СОУТ") == "что такое соут"


class TestParseQuestions:
    def test_from_pydantic_model(self):
        q = Questions(questions=["Вопрос один?", "Вопрос два?"])
        assert parse_questions(q) == ["Вопрос один?", "Вопрос два?"]

    def test_from_dict(self):
        assert parse_questions({"questions": ["A?", "B?"]}) == ["A?", "B?"]

    def test_drops_blanks_and_trims(self):
        assert parse_questions({"questions": ["  A?  ", "", "   "]}) == ["A?"]


class TestDedupQuestions:
    def _rec(self, q, cid="src.pdf#1"):
        return {
            "question": q,
            "chunk_id": cid,
            "source": "src.pdf",
            "chunk_preview": "x",
        }

    def test_removes_exact_duplicates_case_insensitive(self):
        recs = [
            self._rec("Кто проходит обучение?"),
            self._rec("кто проходит обучение??"),
            self._rec("Что такое СОУТ?"),
        ]
        kept, removed = dedup_questions(recs, embed_fn=None)
        assert removed == 1
        assert [r["question"] for r in kept] == [
            "Кто проходит обучение?",
            "Что такое СОУТ?",
        ]

    def test_near_duplicate_by_cosine(self):
        recs = [self._rec("A"), self._rec("B"), self._rec("C")]

        # A and B are near-identical (cosine ~1.0), C is orthogonal
        vectors = {
            "a": [1.0, 0.0],
            "b": [0.999, 0.0447],
            "c": [0.0, 1.0],
        }

        def embed_fn(texts):
            return [vectors[t.lower()] for t in texts]

        kept, removed = dedup_questions(
            recs, embed_fn=embed_fn, near_dup_threshold=0.95
        )
        assert removed == 1
        assert {r["question"] for r in kept} == {"A", "C"}


class TestBuildGtRecord:
    def test_schema(self):
        chunk = {
            "chunk_id": 7,
            "text": "Полный текст чанка про обучение по охране труда." * 5,
            "metadata": {"source": "trudkodeks.pdf", "chunk_id": 7},
        }
        rec = build_gt_record("Кто обучается?", chunk)
        assert set(rec) == {"question", "chunk_id", "source", "chunk_preview"}
        assert rec["question"] == "Кто обучается?"
        assert rec["chunk_id"] == "trudkodeks.pdf#7"
        assert rec["source"] == "trudkodeks.pdf"
        assert rec["chunk_preview"].startswith("Полный текст чанка")
        assert len(rec["chunk_preview"]) <= 200

    def test_chunk_id_zero_is_not_treated_as_missing(self):
        """chunk_id numbering is per-source and starts at 0 — a falsy but valid id."""
        passage = {
            "chunk_id": 0,
            "text": "текст",
            "metadata": {"source": "p2464.pdf", "chunk_id": 0},
        }
        assert build_gt_record("Q?", passage)["chunk_id"] == "p2464.pdf#0"


class TestToPassage:
    def test_lifts_chunk_id_to_top_level(self):
        doc = {"text": "t", "metadata": {"source": "a.pdf", "chunk_id": 3}}
        assert to_passage(doc) == {
            "text": "t",
            "metadata": {"source": "a.pdf", "chunk_id": 3},
            "chunk_id": 3,
        }

    def test_no_chunk_id_key_when_metadata_lacks_it(self):
        passage = to_passage({"text": "t", "metadata": {"source": "a.pdf"}})
        assert "chunk_id" not in passage


class TestIterCorpusChunks:
    def test_uses_backend_iter_all_documents(self):
        class FakeBackend:
            def iter_all_documents(self):
                yield {"text": "one", "metadata": {"source": "a.pdf", "chunk_id": 0}}
                yield {"text": "two", "metadata": {"source": "a.pdf", "chunk_id": 1}}

        passages = iter_corpus_chunks(backend=FakeBackend())
        assert [_passage_identity(p) for p in passages] == ["a.pdf#0", "a.pdf#1"]


class TestPassageIdentity:
    """The GT ids must equal the ids the retrieval runners (#6) emit, otherwise
    every Hit Rate silently reads 0."""

    CASES = [
        {"chunk_id": 4, "text": "x", "metadata": {"source": "a.pdf", "page_no": 2}},
        {"chunk_id": 0, "text": "x", "metadata": {"source": "a.pdf"}},
        {"text": "no chunk id here", "metadata": {"source": "a.pdf", "page_no": 7}},
        {"chunk_id": "", "text": "empty id", "metadata": {"source": "b.pdf"}},
        {"text": "no metadata at all", "metadata": None},
    ]

    def test_chunk_id_branch(self):
        assert _passage_identity(self.CASES[0]) == "a.pdf#4"

    def test_content_fallback_branch(self):
        assert _passage_identity(self.CASES[2]) == "a.pdf|7|no chunk id here"

    @pytest.mark.integration
    def test_identity_matches_nlp_core(self):
        """Pins the local mirror to the real passage_identity. Needs the full
        env (nlp_core imports pymorphy3), so it is not part of the unit run."""
        nlp_core = pytest.importorskip("src.v7.nlp_core")
        for case in self.CASES:
            assert _passage_identity(case) == nlp_core.passage_identity(case)


class TestPricing:
    """Cost must be computed for the model actually used.

    Regression: PRICE_PER_1M was a single hard-coded gpt-4o-mini rate applied to
    whatever model the factory happened to return, so both the pre-flight
    estimate and the COST_ABORT_USD guard could be off by the ratio between two
    models' prices.
    """

    def test_known_model_rate(self):
        from eval.generate_retrieval_gt import price_for

        assert price_for("gpt-4o-mini") == {"input": 0.15, "output": 0.60}
        assert price_for("gpt-4o") == {"input": 2.50, "output": 10.00}

    def test_unknown_model_raises(self):
        """Silently pricing an unknown model at some other model's rate is the bug."""
        from eval.generate_retrieval_gt import price_for

        with pytest.raises(ValueError, match="gpt-9-turbo"):
            price_for("gpt-9-turbo")

    def test_total_price_uses_given_model(self):
        from eval.generate_retrieval_gt import calc_total_price

        usages = [{"input": 1_000_000, "output": 1_000_000}]
        assert calc_total_price(usages, model="gpt-4o-mini") == pytest.approx(0.75)
        assert calc_total_price(usages, model="gpt-4o") == pytest.approx(12.50)

    def test_estimate_scales_with_model_price(self):
        from eval.generate_retrieval_gt import estimate_cost

        cheap = estimate_cost(1000, model="gpt-4o-mini")
        dear = estimate_cost(1000, model="gpt-4o")
        assert dear > cheap * 15

    def test_estimate_matches_manual_arithmetic(self):
        from eval.generate_retrieval_gt import estimate_cost

        # 500 input tokens (350 chunk + 150 overhead) + 60 output per chunk.
        expected = (500 / 1_000_000 * 0.15 + 60 / 1_000_000 * 0.60) * 100
        assert estimate_cost(100, model="gpt-4o-mini") == pytest.approx(expected)


class TestGeneratorModel:
    def test_generator_pins_its_own_model(self, monkeypatch):
        """The generator must not inherit the eval judge's model: judging answers
        and inventing questions are different jobs with different price tags."""
        import eval.generate_retrieval_gt as g

        captured = {}

        def fake_get_judge_llm(**kwargs):
            captured.update(kwargs)
            return object()

        import src.infra.llm_factory as lf

        monkeypatch.setattr(lf, "get_judge_llm", fake_get_judge_llm)

        g._make_llm()

        assert captured.get("model_name") == g.GEN_MODEL
        assert g.GEN_MODEL == "gpt-4o-mini"


class TestJunkFilterBoilerplate:
    """Титульники и преамбулы-реквизиты давали неотвечаемые вопросы: весь смоук
    02.09.2026 (15 вопросов) пришёлся на шапку 29н.pdf. Фильтр по длине их не ловит —
    шапка длинная."""

    TITLE_PAGE = (
        "МИНИСТЕРСТВО ЗДРАВООХРАНЕНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ\n"
        "ПРИКАЗ\n"
        "от 28 января 2021 г. N 29н\n"
        "ОБ УТВЕРЖДЕНИИ ПОРЯДКА ПРОВЕДЕНИЯ ОБЯЗАТЕЛЬНЫХ ПРЕДВАРИТЕЛЬНЫХ И "
        "ПЕРИОДИЧЕСКИХ МЕДИЦИНСКИХ ОСМОТРОВ РАБОТНИКОВ, ПРЕДУСМОТРЕННЫХ ЧАСТЬЮ "
        "ЧЕТВЕРТОЙ СТАТЬИ 213 ТРУДОВОГО КОДЕКСА РОССИЙСКОЙ ФЕДЕРАЦИИ"
    )

    PREAMBLE = (
        "МИНИСТЕРСТВО ЗДРАВООХРАНЕНИЯ РОССИЙСКОЙ ФЕДЕРАЦИИ\n"
        "Федерации (Собрание законодательства Российской Федерации, 2002, N 1, ст. 3; "
        "2015, N 29, ст. 4356), пунктом 6 статьи 34 Федерального закона от 30 марта "
        '1999 г. N 52-ФЗ "О санитарно-эпидемиологическом благополучии населения" '
        "(Собрание законодательства Российской Федерации, 1999, N 14, ст. 1650; "
        "2013, N 48, ст. 6165), пунктом 14 части 2 статьи 14, частью 3 статьи 24 "
        "Федерального закона от 21 ноября 2011 г."
    )

    NORMATIVE = (
        "Работодатель обязан организовать проведение обязательных предварительных "
        "медицинских осмотров за счёт собственных средств. Направление на осмотр "
        "выдаётся работнику под роспись не позднее чем за 5 рабочих дней до даты "
        "осмотра, установленной в календарном плане (ст. 213 ТК РФ)."
    )

    def test_uppercase_title_page_is_junk(self):
        assert is_junk_chunk(self.TITLE_PAGE) is True

    def test_citation_preamble_is_junk(self):
        assert is_junk_chunk(self.PREAMBLE) is True

    def test_normative_text_with_one_citation_is_kept(self):
        """Ссылка на статью сама по себе не делает чанк реквизитами."""
        assert is_junk_chunk(self.NORMATIVE) is False


class TestSelectChunks:
    """`--limit` резал `chunks[:N]` — смоук попадал на начало одного документа
    и ничего не говорил о корпусе. Нужна случайная выборка с фиксируемым seed."""

    @staticmethod
    def _chunks(n=100):
        return [{"text": f"chunk {i}"} for i in range(n)]

    def test_sample_is_not_the_first_n(self):
        from eval.generate_retrieval_gt import select_chunks

        picked = select_chunks(self._chunks(), sample=5, seed=0)
        assert len(picked) == 5
        assert [c["text"] for c in picked] != [f"chunk {i}" for i in range(5)]

    def test_sample_is_deterministic_for_a_seed(self):
        from eval.generate_retrieval_gt import select_chunks

        a = select_chunks(self._chunks(), sample=5, seed=7)
        b = select_chunks(self._chunks(), sample=5, seed=7)
        assert a == b

    def test_sample_larger_than_corpus_returns_all(self):
        from eval.generate_retrieval_gt import select_chunks

        assert len(select_chunks(self._chunks(3), sample=10, seed=0)) == 3

    def test_limit_still_takes_the_head(self):
        from eval.generate_retrieval_gt import select_chunks

        picked = select_chunks(self._chunks(), limit=3)
        assert [c["text"] for c in picked] == ["chunk 0", "chunk 1", "chunk 2"]

    def test_no_limit_no_sample_returns_everything(self):
        from eval.generate_retrieval_gt import select_chunks

        assert len(select_chunks(self._chunks(12))) == 12

    def test_cli_exposes_sample_and_seed(self):
        from eval.generate_retrieval_gt import _parse_args

        args = _parse_args(["--sample", "40", "--seed", "3"])
        assert args.sample == 40
        assert args.seed == 3


class TestCrossChunkDedup:
    """Чанки 29н.pdf#2 и #3 дали фактически один вопрос с разными «правильными»
    ответами — размечен один chunk_id, а верны оба. Такая пара ядовита для Hit Rate,
    поэтому между чанками порог схожести строже, чем внутри чанка."""

    @staticmethod
    def _rec(q, cid):
        return {"question": q, "chunk_id": cid, "source": "s.pdf", "chunk_preview": "x"}

    VECTORS = {
        "a": [1.0, 0.0],
        "b": [0.96, 0.28],  # cosine с A ≈ 0.96: ниже 0.98, выше 0.90
        "c": [0.0, 1.0],
    }

    def _embed(self, texts):
        return [self.VECTORS[t.lower()] for t in texts]

    def test_similar_questions_from_different_chunks_are_dropped(self):
        recs = [self._rec("A", "s.pdf#2"), self._rec("B", "s.pdf#3")]
        kept, removed = dedup_questions(
            recs, embed_fn=self._embed, near_dup_threshold=0.98, cross_chunk_threshold=0.90
        )
        assert removed == 1
        assert [r["question"] for r in kept] == ["A"]

    def test_similar_questions_from_the_same_chunk_survive_the_looser_threshold(self):
        """Внутри чанка три вопроса намеренно перефразируют один текст —
        их отсеивает только более высокий порог."""
        recs = [self._rec("A", "s.pdf#2"), self._rec("B", "s.pdf#2")]
        kept, removed = dedup_questions(
            recs, embed_fn=self._embed, near_dup_threshold=0.98, cross_chunk_threshold=0.90
        )
        assert removed == 0
        assert [r["question"] for r in kept] == ["A", "B"]

    def test_unrelated_question_from_another_chunk_survives(self):
        recs = [self._rec("A", "s.pdf#2"), self._rec("C", "s.pdf#3")]
        kept, removed = dedup_questions(
            recs, embed_fn=self._embed, near_dup_threshold=0.98, cross_chunk_threshold=0.90
        )
        assert removed == 0


class TestGenPromptDemandsAnchor:
    """Вопросы вида «Какие документы регулируют охрану здоровья граждан?» отвечает
    половина корпуса — Hit Rate занижает разметка, а не поиск. Промпт обязан
    требовать зацепку из фрагмента."""

    def test_prompt_requires_a_concrete_anchor(self):
        from eval.generate_retrieval_gt import GEN_PROMPT

        low = GEN_PROMPT.lower()
        assert "зацепк" in low
        assert any(w in low for w in ("срок", "номер", "условие", "адресат"))

    def test_prompt_forbids_questions_answerable_without_the_chunk(self):
        from eval.generate_retrieval_gt import GEN_PROMPT

        low = GEN_PROMPT.lower()
        assert "не задавай" in low or "не формулируй" in low


class TestJunkFilterKeepsRealNorms:
    """Границы фильтра на реальных чанках корпуса (проверено 02.09.2026): ссылки
    «(в ред. ФЗ от …)» и сноски стоят в самых обычных статьях. Оба текста
    отсеивались первой версией фильтра — она считала ссылки штуками, а не долей."""

    FGIS_LIST = (
        "Статья  18.  Федеральная  государственная  информационная  система  учета  "
        "результатов проведения специальной оценки условий труда\n"
        "средства измерений, дату окончания срока действия его поверки, дату "
        "проведения измерений, наименования измерявшихся вредного и (или) опасного "
        "производственных факторов;\n"
        '- з) сведения, предусмотренные частью 1.1 статьи 19 настоящего Федерального '
        'закона. (пп. "з" введен Федеральным законом от 24.07.2023 N 381-ФЗ)'
    )

    MINWAGE_NOTE = (
        "Статья 133. Установление минимального размера оплаты труда\n"
        "другими работодателями - за счет собственных средств. (часть вторая в ред. "
        "Федерального закона от 30.06.2006 N 90-ФЗ)\n"
        "КонсультантПлюс: примечание. О выявлении конституционно-правового смысла "
        "ч. 3 ст. 133 см. Постановления КС РФ от 11.04.2019 N 17-П, от 16.12.2019 "
        "N 40-П, от 23.09.2024 N 40-П, 05.03.2025 N 10-П."
    )

    def test_list_of_required_data_is_kept(self):
        assert is_junk_chunk(self.FGIS_LIST) is False

    def test_norm_with_edition_references_is_kept(self):
        assert is_junk_chunk(self.MINWAGE_NOTE) is False


class TestGenPromptForbidsEditionTrivia:
    """Смоук 02.09.2026 после правки промпта: зацепку модель нашла, но брала её из
    реквизитов редакции — «Какой закон ввёл статью 60.1?», «Какой номер пункта
    указывает на …». Такие вопросы никто не задаёт поиску по нормативке."""

    def test_prompt_rejects_questions_about_amending_acts(self):
        from eval.generate_retrieval_gt import GEN_PROMPT

        low = GEN_PROMPT.lower()
        assert "в ред." in low or "редакц" in low
        assert "номер пункта" in low or "номер статьи" in low


class TestMetaQuestionFilter:
    """Промпт снизил долю вопросов про реквизиты, но не обнулил её (смоук
    02.09.2026: 2 из 33). Оставшееся снимается детерминированно — это дешевле
    и надёжнее, чем ещё один абзац инструкции."""

    META = [
        "Какой закон утратил силу в связи с прекращением трудового договора?",
        "Какой федеральный закон ввел пункт о возникновении ограничений?",
        "Каким законом внесены изменения в статью 213.1?",
        "Какой номер пункта указывает на организацию дистанционного общения?",
        "Когда была введена новая редакция статьи 60.1?",
    ]

    REAL = [
        "Какой минимальный срок практического опыта требуется для руководителя стажировки?",
        "Сколько работников может быть прикреплено к одному руководителю стажировки?",
        "Когда запрещено проводить огневые работы на объектах с массовым пребыванием людей?",
        "Какой класс опасности имеют вещества, не перечисленные в пунктах 2 - 7 таблицы?",
        "Какое письменное согласие требуется для работы по совместительству?",
    ]

    def test_drops_questions_about_amending_acts(self):
        from eval.generate_retrieval_gt import is_meta_question

        for q in self.META:
            assert is_meta_question(q) is True, q

    def test_keeps_substantive_questions(self):
        from eval.generate_retrieval_gt import is_meta_question

        for q in self.REAL:
            assert is_meta_question(q) is False, q

    def test_generation_pipeline_drops_them(self):
        """Фильтр должен стоять в пути записи, а не только существовать."""
        from eval.generate_retrieval_gt import filter_questions

        kept = filter_questions(self.META[:2] + self.REAL[:2])
        assert kept == self.REAL[:2]


class TestSourceReferenceQuestionFilter:
    """Второй смоук (02.09.2026, --sample 60 --seed 11, 173 вопроса) показал класс
    мусора, которого прошлый промпт не касался: вопрос про то, КАКОЙ акт что-то
    регулирует. Ответ на него — название документа, а не норма; для retrieval-GT
    он бесполезен, потому что подходит к любому чанку этого же акта."""

    SOURCE_REF = [
        "Какой закон регулирует обязанности работодателя по созданию условий для представителей работников?",
        "Какое законодательство регулирует передачу сведений, относящихся к государственной тайне?",
        "Какой документ регулирует порядок оформления паспорта населенного пункта?",
        "Какой документ определяет порядок проверки неквалифицированной электронной подписи?",
        "На основании какого федерального закона действует Российская трехсторонняя комиссия?",
        "Какое законодательство регулирует трудоустройство лиц до восемнадцати лет?",
        "Какие документы определяют особенности трудоустройства работников младше восемнадцати лет?",
        "Какие документы могут регламентировать участие работников в управлении организацией?",
    ]

    # Вопросы про содержание, в которых слово «документ» стоит законно.
    KEEP = [
        "На основании каких документов работодатель обязан обеспечивать деятельность представителей работников?",
        "Какие документы необходимы для расследования несчастных случаев на производстве?",
        "Какой формат должен иметь документ, подтверждающий подготовку рабочих мест?",
    ]

    def test_drops_questions_about_which_act_governs(self):
        from eval.generate_retrieval_gt import is_source_reference_question

        for q in self.SOURCE_REF:
            assert is_source_reference_question(q) is True, q

    def test_keeps_substantive_questions_mentioning_documents(self):
        from eval.generate_retrieval_gt import is_source_reference_question

        for q in self.KEEP:
            assert is_source_reference_question(q) is False, q


class TestStructuralReferenceFilter:
    """Третий класс из того же смоука: вопрос ссылается на структуру источника
    («согласно статье 213», «согласно графе 3», «какое приложение содержит»).
    Такой вопрос либо содержит ответ-указатель, либо задаёт вопрос о нумерации —
    поиском по нормативке его не задают."""

    STRUCTURAL = [
        "Какой вид сведений не подлежит передаче на микропредприятиях согласно пунктам 106 и 118?",
        "Какой способ взаимодействия между работодателем и работником описан в статье 22.3?",
        "Какое направление противодействия задолженности по зарплате указано в статье 158.1?",
        "Кто несет ответственность за организацию труда дистанционных работников в соответствии со статьей 312.6?",
        "Кто из работников должен быть учтен при проведении СОУТ согласно графе 3?",
        "Какой пункт статьи 81 Кодекса регулирует увольнение в связи с ликвидацией организации?",
        "Кто обязан проходить медицинские осмотры согласно статье 213 Трудового кодекса?",
        "Какой класс заболеваний относится к новообразованиям согласно данному списку?",
        "Кто подлежит отстранению от работы в соответствии с данной статьей?",
        "Какой порядок установлен для изменения соглашения согласно настоящему Кодексу?",
        "В каком порядке работодатель должен заключать коллективный договор согласно Кодексу?",
        "Какое приложение содержит информацию о проведении измерений вредных факторов?",
    ]

    KEEP = [
        "Какой класс опасности имеют вещества, не перечисленные в пунктах 2 - 7 таблицы?",
        "Какой минимальный срок практического опыта требуется для руководителя стажировки?",
        "В каком порядке осуществляется возмещение стоимости материалов, принадлежащих надомникам?",
        "Какой документ обычно является приложением к графикам сменности?",
    ]

    def test_drops_questions_referring_to_source_structure(self):
        from eval.generate_retrieval_gt import has_structural_reference

        for q in self.STRUCTURAL:
            assert has_structural_reference(q) is True, q

    def test_keeps_questions_without_self_reference(self):
        from eval.generate_retrieval_gt import has_structural_reference

        for q in self.KEEP:
            assert has_structural_reference(q) is False, q


class TestWeakQuestionPipeline:
    """Все три фильтра должны стоять в пути записи, а не только существовать."""

    def test_filter_questions_applies_all_three(self):
        from eval.generate_retrieval_gt import filter_questions

        good = "Какой минимальный срок практического опыта требуется для руководителя стажировки?"
        kept = filter_questions(
            [
                "Каким законом внесены изменения в статью 213.1?",
                "Какой закон регулирует обязанности работодателя?",
                "Кто подлежит отстранению от работы в соответствии с данной статьей?",
                good,
            ]
        )
        assert kept == [good]


class TestPromptForbidsGuessableQuestions:
    """Класс, который детерминированно не ловится: вопрос, ответ на который
    угадывается без документа («Какова величина МРОТ?»). Лечится промптом."""

    def test_prompt_demands_answer_unguessable_without_chunk(self):
        from eval.generate_retrieval_gt import GEN_PROMPT

        low = GEN_PROMPT.lower()
        assert "угадыва" in low
        assert "какой закон" in low or "какое законодательство" in low


class TestFilterEvasionsFromSecondSmoke:
    """Смоук после правки промпта (02.09.2026, seed 11) показал два обхода
    фильтров: голое «внес изменения» без окончания и «какого ТИПА закона»
    с лишним словом между вопросительным местоимением и существительным."""

    EVASIONS = [
        "Какой закон внес изменения в правила трудоустройства лиц до восемнадцати лет?",
        "На основании какого типа закона осуществляется деятельность Российской трехсторонней комиссии?",
    ]

    KEEP = [
        "Какой документ обычно является приложением к графикам сменности?",
        "Какие документы необходимы для расследования несчастных случаев на производстве?",
        "В каком порядке осуществляется возмещение стоимости материалов, принадлежащих надомникам?",
        "Какой формат должен иметь документ, подтверждающий подготовку рабочих мест?",
    ]

    def test_evasions_are_caught(self):
        from eval.generate_retrieval_gt import is_weak_question

        for q in self.EVASIONS:
            assert is_weak_question(q) is True, q

    def test_neighbours_survive(self):
        from eval.generate_retrieval_gt import is_weak_question

        for q in self.KEEP:
            assert is_weak_question(q) is False, q

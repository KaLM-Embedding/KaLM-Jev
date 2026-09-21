import json

TEMPLATE_VERSION = "v3-noul-state-document"
CHOICE_ADAPTER = (
    "Evaluate whether the candidate option described in the Document "
    "is an appropriate answer to the question above, "
    "given the information in the Query. "
    "Judge the candidate by its stated description, "
    "not merely by topical relevance."
)
SCORE_ADAPTER = (
    "Evaluate whether the level described in the Document "
    "accurately characterizes the Query with respect to the "
    "assessment above. "
    "Judge this level by its stated criteria, "
    "not merely by topical relevance."
)
NOUL_ADAPTER = (
    "Given the Query, evaluate whether the candidate criterion in the "
    "Document correctly describes the answer to the question above. "
    "Answer yes if this candidate criterion is satisfied, otherwise no."
)
NOUL_DEFAULT_CRITERIA = {
    "true": "The answer to the question is yes.",
    "false": "The answer to the question is no.",
}
ADAPTERS = {"choice": CHOICE_ADAPTER, "score": SCORE_ADAPTER, "noul": NOUL_ADAPTER}


def render_content(value):
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )

import json

TEMPLATE_VERSION = "v1"
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
    "Evaluate whether the information in the Query warrants a yes "
    "answer to the question stated in the Document. "
    "Apply any provided true and false criteria. "
    "Judge whether the condition holds, not merely whether "
    "the Query is related to the topic."
)
ADAPTERS = {"choice": CHOICE_ADAPTER, "score": SCORE_ADAPTER, "noul": NOUL_ADAPTER}


def render_content(value):
    return value if isinstance(value, str) else json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    )

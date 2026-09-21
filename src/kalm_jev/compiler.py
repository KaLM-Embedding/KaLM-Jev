from dataclasses import dataclass

from .templates import ADAPTERS, NOUL_DEFAULT_CRITERIA, render_content


@dataclass(frozen=True)
class Task:
    question_id: str
    option_id: str
    instruction: str
    query: str
    document: str


def compile_request(request):
    query = render_content(request.state)
    tasks = []
    for qid, question in request.questions.items():
        original = render_content(question.instructions)
        instruction = original + "\n\n" + ADAPTERS[question.type]
        if question.type == "choice":
            documents = [(key, key if value is None else key + ": " + render_content(value))
                         for key, value in question.criteria.items()]
        elif question.type == "score":
            documents = [(str(i), render_content(value)) for i, value in enumerate(question.criteria)]
        elif not question.criteria:
            documents = [("", query)]
        else:
            # Each native yes/no readout judges one candidate criterion. Keep
            # polarity in option_id, not in the encoded Document text.
            documents = [(key, render_content(question.criteria.get(key, NOUL_DEFAULT_CRITERIA[key])))
                         for key in ("true", "false")]
        tasks.extend(Task(qid, key, instruction, query, doc) for key, doc in documents)
    return tasks

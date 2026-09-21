from dataclasses import dataclass

from .templates import ADAPTERS, render_content


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
        else:
            parts = ["Question: " + original]
            for key in ("true", "false"):
                if key in question.criteria:
                    parts.append(key.capitalize() + " criteria: " + render_content(question.criteria[key]))
            documents = [("", "\n".join(parts))]
        tasks.extend(Task(qid, key, instruction, query, doc) for key, doc in documents)
    return tasks

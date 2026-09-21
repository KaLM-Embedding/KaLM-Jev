import math
from dataclasses import dataclass

from .templates import render_content


@dataclass(frozen=True)
class Calibration:
    choice_temperature: float = 1.0
    score_temperature: float = 1.0
    noul_a: float = 1.0
    noul_b: float = 0.0

    def __post_init__(self):
        values = (self.choice_temperature, self.score_temperature, self.noul_a, self.noul_b)
        if not all(math.isfinite(x) for x in values) or min(values[:3]) <= 0:
            raise ValueError("Temperatures and noul_a must be positive; parameters must be finite")

    @property
    def status(self):
        return "none" if self == Calibration() else "manual"


def sigmoid(x):
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def softmax(values, temperature=1):
    maximum = max(values)
    weights = [math.exp((x - maximum) / temperature) for x in values]
    total = math.fsum(weights)
    return [x / total for x in weights]


def confidence(probabilities):
    if len(probabilities) == 1:
        return 1.0
    entropy = -math.fsum(p * math.log(p) for p in probabilities if p > 0)
    return min(1.0, max(0.0, 1 - entropy / math.log(len(probabilities))))


def aggregate(request, tasks, margins, calibration):
    if len(tasks) != len(margins) or not all(math.isfinite(x) for x in margins):
        raise RuntimeError("Backend returned invalid margins")
    grouped = {key: [] for key in request.questions}
    for task, margin in zip(tasks, margins):
        grouped[task.question_id].append((task.option_id, margin))
    answers = {}
    for key, question in request.questions.items():
        candidates = grouped[key]
        if question.type == "noul":
            if not question.criteria:
                if len(candidates) != 1 or candidates[0][0] != "":
                    raise RuntimeError("Noul without criteria requires exactly one state margin")
                margin = candidates[0][1]
            else:
                if len(candidates) != 2 or {label for label, _ in candidates} != {"true", "false"}:
                    raise RuntimeError("Noul requires exactly one true and one false margin")
                by_label = dict(candidates)
                # At a=1, b=0 this equals softmax([z_true, z_false])[0].
                margin = by_label["true"] - by_label["false"]
            answers[key] = {"type": "noul", "noul": sigmoid(calibration.noul_a * margin + calibration.noul_b)}
            continue
        z = [margin for _, margin in candidates]
        temp = calibration.choice_temperature if question.type == "choice" else calibration.score_temperature
        p = softmax(z, temp)
        keys = list(question.criteria) if question.type == "choice" else list(map(str, range(len(p))))
        answer = {"type": question.type, "probabilities": dict(zip(keys, p)), "confidence": confidence(p)}
        if question.type == "choice":
            answer["choice"] = keys[max(range(len(p)), key=p.__getitem__)]
        else:
            answer["score"] = math.fsum(i * value for i, value in enumerate(p))
            answer["legend"] = {str(i): render_content(value) for i, value in enumerate(question.criteria)}
        answers[key] = answer
    return answers

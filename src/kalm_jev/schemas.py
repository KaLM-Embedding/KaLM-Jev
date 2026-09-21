"""Strict validation for HTTP and local Python requests."""
import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, model_validator


class JevError(Exception):
    def __init__(self, status, message, field=None, kind="validation_error"):
        super().__init__(message)
        self.status, self.message, self.field, self.kind = status, message, field, kind

    def body(self):
        error = {"type": self.kind, "message": self.message}
        if self.field is not None:
            error["field"] = self.field
        return {"error": error}


def validate_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for item in value:
            validate_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            validate_json(item)
        return
    raise ValueError("Expected finite JSON values and string object keys")


def content(value):
    if type(value) not in (str, dict, list):
        raise ValueError("Content must be a string, object or array")
    try:
        validate_json(value)
    except RecursionError as exc:
        raise ValueError("Content is cyclic or nested too deeply") from exc
    return value


Content = Annotated[str | dict | list, BeforeValidator(content)]
Id = Annotated[str, Field(min_length=1)]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class ChoiceQuestion(StrictModel):
    type: Literal["choice"]
    instructions: Content
    criteria: dict[Id, Content | None] = Field(min_length=1, max_length=255)


class ScoreQuestion(StrictModel):
    type: Literal["score"]
    instructions: Content
    criteria: list[Content] = Field(min_length=2, max_length=10)


class NoulQuestion(StrictModel):
    type: Literal["noul"]
    instructions: Content
    criteria: dict[Literal["true", "false"], Content] = Field(default_factory=dict)

    @model_validator(mode="after")
    def nonempty_criteria(self):
        if "criteria" in self.model_fields_set and not self.criteria:
            raise ValueError("Provided Noul criteria must contain true or false")
        return self


Question = Annotated[ChoiceQuestion | ScoreQuestion | NoulQuestion, Field(discriminator="type")]


class Request(StrictModel):
    model: str | None = None
    state: Content
    questions: dict[Id, Question] = Field(min_length=1)

    @model_validator(mode="before")
    @classmethod
    def no_explicit_null_model(cls, value):
        if isinstance(value, dict) and "model" in value and value["model"] is None:
            raise ValueError("model must be a string when provided")
        return value


def strict_loads(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Non-finite JSON number: {value}")

    try:
        value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
        validate_json(value)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise JevError(422, str(exc), kind="invalid_json") from exc

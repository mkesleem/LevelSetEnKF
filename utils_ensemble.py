# Copyright (c) 2026 Xuhui Zhou
# Copyright (c) 2026 Michael Sleeman
# SPDX-License-Identifier: MIT

"""Shared helpers for constructing perturbed ensemble input files."""


import re
import numpy as np


NUM = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
WS = r"[ \t]*"

PAT_DENSITY = re.compile(rf"(\bDensity{WS}={WS}){NUM}({WS};)")
PAT_PRESS = re.compile(rf"(\bPressure{WS}={WS}){NUM}({WS};)")


def sample_ensemble(
    n: int,
    ensemble_normal_params: dict,
    seed: int = 42,
) -> dict:
    """
    Draw normally distributed parameters for an ensemble.

    Input:
    - n: number of ensemble members
    - ensemble_normal_params: dictionary of parameter names to mean and standard deviation
    - seed: random-number-generator seed

    Output:
    - samples: dictionary of parameter names to arrays of n sampled values

    If n is one, each parameter is set exactly to its specified mean.
    """
    rng = np.random.default_rng(seed)
    if n == 1:
        return {
            name: np.array([values["mean"]])
            for name, values in ensemble_normal_params.items()
        }
    return {
        name: rng.normal(values["mean"], values["std"], size=n)
        for name, values in ensemble_normal_params.items()
    }


def format_number(
    value: float
) -> str:
    """Convert a number to a compact string using general formatting."""
    return f"{value:g}"


def substitute_one(
    pattern: re.Pattern,
    text: str,
    new_value: float,
) -> tuple[str, int]:
    """Replace the first numeric value captured by a regular expression."""
    def replace(match: re.Match) -> str:
        return f"{match.group(1)}{format_number(new_value)}{match.group(2)}"

    found = pattern.search(text) is not None
    return pattern.sub(replace, text, count=1), int(found)


def substitute_in_block(
    outer_pattern: re.Pattern,
    inner_pattern: re.Pattern,
    text: str,
    new_value: float,
    label: str,
) -> str:
    """Replace one numeric field inside one named input-file block."""
    matches = list(outer_pattern.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 '{label}' block, found {len(matches)}."
        )

    match = matches[0]
    body = match.group("body")
    if not inner_pattern.search(body):
        raise ValueError(f"Expected field in '{label}' not found.")

    new_body = inner_pattern.sub(
        lambda field: (
            f"{field.group(1)}{format_number(new_value)}{field.group(2)}"
        ),
        body,
        count=1,
    )
    start, end = match.span("body")
    return text[:start] + new_body + text[end:]


def parse_ne_token(token: str) -> int:
    """Parse a positive ensemble size written as either N or Ne-N."""
    token = token.strip()
    if token.startswith("Ne-"):
        token = token.split("-", 1)[1]
    n = int(token)
    if n <= 0:
        raise ValueError("Ensemble size must be a positive integer.")
    return n

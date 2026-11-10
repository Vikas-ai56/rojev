"""Plug-ins named in a config file: a robot, an S1 policy, a planner, an interaction decider or a done checker is built
by a factory the config names, so a run is assembled from configuration and no model or robot is imported by name.

A plug-in section is `{factory: "package.module:callable", args: {...}}`; `build(section)` imports the callable and
calls it with the args. Factories live with the robot or model they build (executors, services), never in `rojev/`;
the module must be importable in the environment that runs the session.
"""
import importlib


def resolve(reference: str):
    module_name, sep, attribute = reference.partition(":")
    if not sep or not module_name or not attribute:
        raise ValueError(f"plug-in reference {reference!r}: expected 'package.module:callable'")
    target = importlib.import_module(module_name)
    for part in attribute.split("."):
        target = getattr(target, part)
    return target


def build(section: dict):
    if "factory" not in section:
        raise ValueError(f"plug-in section {sorted(section)} has no 'factory'")
    unknown = sorted(set(section) - {"factory", "args"})
    if unknown:
        raise ValueError(f"plug-in section has unknown keys {unknown}; expected factory and args")
    return resolve(section["factory"])(**(section.get("args") or {}))

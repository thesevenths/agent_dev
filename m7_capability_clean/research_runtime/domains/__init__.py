"""Domain wiring for the engine-agnostic research runtime.

A domain supplies config whitelists, dataset identity, metric protocol and
budgets. The generic campaign / comparison / service layers must never import
anything from this package (the dependency points the other way).
"""

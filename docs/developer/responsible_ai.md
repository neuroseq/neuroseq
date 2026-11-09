# Responsible use of AI

Adapted from
[fastplotlib/claude-skills: Scope, Responsibility & Scientific Integrity](https://github.com/fastplotlib/claude-skills#scope-responsibility--scientific-integrity).

## Review over generation

AI tools are cheap at producing code and expensive to review. Generated code is
a draft: the value of a contribution lies in the review and validation behind it.
Prefer small, focused changes you can read in full over large generated ones, and
do not submit code you cannot explain.

## Scope

Use AI for implementation, documentation and routine coding whose expected
behaviour you can check. It cannot decide whether a model, analysis or plot
answers the scientific question. Without a meaningful correctness check, seek
domain expertise.

## Responsibility

You are responsible for every change you submit. Review it, verify assumptions
and run the [tests](testing.md). Fluent output can hide invented APIs, changed
requirements or unsupported claims.

## Scientific integrity

Code that runs and plots that look right can still misrepresent an experiment.
Check against independent expectations:

- array shapes, physical units, sampling rates and time alignment;
- neuron identifiers across imported data, stimulus arrays and recordings;
  stimulation amplitudes and durations;
- axis labels, colour scales and display limits;
- simulation and analysis output against known cases, ideally as a
  [scientific validation test](testing.md#scientific-validation-tests).

Record the assumptions and validation evidence. If you cannot say how you would
detect an error, get expert review before drawing a scientific conclusion.

# Optional model adapters

These adapters preserve the validated module-selection and evaluation entry
points used in earlier experiments. They intentionally remain separate from
the paper LLM workflow until their model and dataset contracts are unified.
External model repositories and datasets are not vendored.

The staging tree contains executable dense loaders and evaluators, ImageNet zip
transforms, Chameleon/GenImage and FakeClue dataset adapters, and validated
compression module selectors. Compression backends remain the same external
CUTLASS wrapper used by the main project. Install optional dependencies with
`pip install -e '.[vision]'` and run an evaluator from the DASO repository root,
for example `python -m extras.maxvit.evaluate --help`.

# Contributing to flika_plugins

Thanks for your interest. This is research software written for a working microscopy lab,
and contributions are welcome from anyone — bug reports and documentation fixes
just as much as code.

Everyone interacting in this project is expected to be respectful and
constructive. Concerns can be raised with george.dickinson@gmail.com.

## Getting help

If something is unclear, that is a documentation bug and worth reporting. Open a
[GitHub issue](https://github.com/gddickinson/flika_plugins/issues) rather than emailing —
answering in the open helps the next person with the same question.

For general bioimage-analysis questions that aren't specific to this project,
the [image.sc forum](https://forum.image.sc/) has a much larger community and is
usually the faster route to an answer.

## Reporting a problem

Open an issue and include, as far as you can:

1. **What you ran** — the exact command, or the GUI action.
2. **What happened** versus what you expected. Paste the full traceback if there
   was one.
3. **Your data** — which plugin, your FLIKA version, and the recording type it was run on.
4. **Your platform** — OS, Python version, and the output of `pip list` (or
   `conda list`) for the main dependencies.

Please only share data you have the right to share. A single cropped frame or a
small synthetic example is usually enough to reproduce a problem.

## Suggesting a change

Open an issue describing the scientific question you are trying to answer, not
only the feature you have in mind. There is often an existing route to the same
result, and if there isn't, the underlying question shapes the right design.

## Contributing code

1. Fork and branch from the default branch.
2. Keep the change focused — one concern per pull request.
3. Run whatever tests the repository has before opening the PR, and say in the
   PR which ones you ran and what the result was.
4. Note any new dependency and why it's needed.

### House style

- **Keep every file under 500 lines.** If a change would push a file over,
  split it into focused modules first.
- **Update `INTERFACE.md` if the repository has one.** It is the navigation map
  — what each module contains and how the pieces connect. A structural change
  that doesn't update it is incomplete.
- Match the surrounding code's naming and comment density. Comment constraints
  the code can't express, not what the next line does.

## Citing

If this work contributes to something you publish, please cite it. `CITATION.cff`
in the repository root has the metadata, and GitHub's "Cite this repository"
button will format it for you.

## Licence

By contributing you agree that your contributions are licensed under the
project's licence, as recorded in the `LICENSE` file.

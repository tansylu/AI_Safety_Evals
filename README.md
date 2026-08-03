# AI Safety Evals

Coursework for the AI safety & evals methodology course
([Monoid Center, March–June 2026](https://monoid.ru/events/course-safety-evals-2026)) — four
notebooks building LLM and agent evaluations with
[Inspect AI](https://inspect.aisi.org.uk/).

Each notebook covers one week of the taught stage: a technical assignment implementing and
analysing the methods from that week, alongside a short written note justifying the
methodological choices. All evaluations run against local models via
[Ollama](https://ollama.com/), so the notebooks are runnable end to end without API keys.

## Contents

| Notebook | Topic | Dataset | Models |
|---|---|---|---|
| [Week 1](inspect_ai_tutorial_week_1.ipynb) | Inspect AI basics: tasks, solvers, scorers; single- and multiple-choice evals; answer-position bias | Synthetic (generated in-notebook) | `ollama/llama2` |
| [Week 2](inspect_ai_tutorial_week_2.ipynb) | Benchmark replication and eval statistics: confidence intervals, paired model comparison, power analysis | [`cais/mmlu`](https://huggingface.co/datasets/cais/mmlu) (`world_religions` subset) | `ollama/llama2`, `ollama/llama3.2` |
| [Week 3](inspect_ai_tutorial_week_3.ipynb) | Designing a custom eval: toxicity classification with an LLM judge, error-rate analysis, prompt engineering | [Jigsaw toxic comments](https://huggingface.co/datasets/thesofakillers/jigsaw-toxic-comment-classification-challenge) | `ollama/mistral` (classifier and judge) |
| [Week 4](inspect_ai_tutorial_week_4.ipynb) | Agent evaluation: tool definitions, three solver architectures, dev/test iteration | [`HuggingFaceH4/MATH-500`](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | `ollama/qwen2.5:3b` |

### Week 1 — Basics and first contact with Inspect AI

Environment setup, the `Task` / `Dataset` / `Solver` / `Scorer` structure, and `TaskState` as
the object solvers mutate. Works through the built-in solvers (`system_message`,
`prompt_template`, `chain_of_thought`, `multiple_choice`) and composes them into chains for
yes/no classification, multi-class sentiment, and multiple-choice tasks with metadata.

The closing assignment measures **answer-position bias**: generating questions with known
answers, placing the correct option at each position in turn, and testing whether accuracy
depends on position — and whether chain-of-thought prompting reduces the effect.

### Week 2 — Evaluating LLMs on MMLU

Runs a partial MMLU evaluation and treats the resulting accuracy as an *estimate* rather than
a number:

- `log_to_df` — converting an `EvalLog` into a per-sample DataFrame for analysis.
- `ci_accuracy_basic` / `ci_accuracy` — CLT confidence intervals on accuracy, including the
  multi-epoch case where repeated samples are averaged before the variance is computed.
- Visualising how interval width shrinks with the number of epochs `k` and the number of
  questions `n`.
- Paired comparison of two models on identical questions, with interval estimation of the
  accuracy *gap* rather than two independent intervals.
- Variance decomposition into question-level and sampling-level components, feeding
  `required_sample_size` — how many questions are needed to detect a given effect.
- A baseline vs. chain-of-thought comparison of a single model against itself.

Follows the approach in Miller (2024), *Adding Error Bars to Evals*.

### Week 3 — Designing a custom evaluation

A toxicity eval where one model labels comments `TOXIC` / `NON_TOXIC` and a second model
judges the label. The point of the week is where this design fails:

- **Judge blinding** is verified rather than assumed — a deliberately "cheating" variant
  exposes the ground-truth label to the judge, confirming the blind variant behaves
  differently.
- `compute_error_rates` separates false positives, false negatives, and unparseable
  responses, since a single accuracy figure hides which direction the classifier errs in.
- A classifier × judge grid tests whether conclusions depend on which model plays which role.
- Prompt engineering is applied to the classifier and the judge independently, to see how
  much of the measured error rate is a property of the model versus the prompt.
- A final section scores a classifier using only the judge, with no ground-truth labels.

### Week 4 — Evaluating LLM agents on mathematical reasoning

Custom tools (`add`, `subtract`, `multiply`, `divide`, `modular_arithmetic`, and a
`sympy_solve` symbolic algebra tool) and a comparison of three scaffolds on the same task:

1. **Approach 0** — plain generation, no tools.
2. **Approach A** — a naive tool loop.
3. **Approach B** — a ReAct agent.

MATH-500 answers are extracted from the final `\boxed{...}` expression and scored with
`model_graded_qa`, since string equality fails on equivalent mathematical forms. The dataset
is split 10% dev / 90% test, with prompt iteration confined to the dev set and the best
configuration then run once on held-out data, broken down by subject and difficulty level.

## Setup

Install Ollama and pull the models used:

```bash
ollama pull llama2
ollama pull llama3.2
ollama pull mistral
ollama pull qwen2.5:3b
```

Install the Python dependencies:

```bash
pip install inspect-ai openai sympy datasets scipy pandas numpy matplotlib
```

Then run any notebook top to bottom. To inspect the resulting logs:

```bash
inspect view --log-dir logs/
```

## Reproducibility notes

- **Eval logs are not committed.** `logs/` and `*.eval` are gitignored, so a few cells in
  Week 2 that read specific log files by filename will need those paths updated after you
  regenerate the logs locally.
- **Sample sizes are small.** Most runs use `limit=5`–`20` to keep iteration fast on local
  models. Accuracy figures at that scale carry confidence intervals wide enough to be
  uninformative — which is itself one of the lessons of Week 2. Raise the limits before
  reading anything into a specific number.
- **Small local models.** The models above are 3B–8B class and run on CPU-friendly hardware.
  Absolute scores are far below frontier models; the notebooks are about eval methodology,
  not model capability.

## Status

Known gaps, for anyone reading the notebooks as they stand:

- Week 2, bonus assignment (clustered standard errors for grouped questions) is not
  implemented.
- Week 4, Assignment 5 — the test-set run was interrupted, so the subject/difficulty
  breakdown and the final dev-vs-test comparison have no output.

## References

- [Inspect AI documentation](https://inspect.aisi.org.uk/) — UK AI Safety Institute
- Miller, E. (2024). *Adding Error Bars to Evals: A Statistical Approach to Language Model
  Evaluations.*
- Yao et al. (2022). *ReAct: Synergizing Reasoning and Acting in Language Models.*
- Hendrycks et al. (2021). *Measuring Massive Multitask Language Understanding* (MMLU).
- Lightman et al. (2023). *Let's Verify Step by Step* (source of the MATH-500 subset).

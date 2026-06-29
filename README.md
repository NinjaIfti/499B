# LectureForge — How We Check Our Quiz Questions Are Actually Good

## What this is

LectureForge can write quiz questions about a lecture in two different ways. This
document explains, in plain language, how we test which way produces better
questions — and how to see the results for yourself.

## Why we built this

It's easy to claim "our AI agent writes better questions." It's harder to prove it.
So instead of just trusting the system, we built a way to **measure** it: generate
questions both ways, have an independent AI grade every single one, and compare the
scores side by side.

## The two ways we generate questions

**The careful way.** The system writes one question at a time. Right after writing
it, the system checks its own work — is it actually at the right difficulty, does it
match the lecture content, is it not just a copy of an earlier question? If the check
fails, it rewrites that one question and tries again (up to three tries) before moving
on to the next question.

**The quick way.** The system is simply asked to write a whole batch of questions —
say, ten at once — in a single request. There's no double-checking step; whatever
comes back is what you get.

The careful way obviously takes more effort. The question we're answering is: **does
that extra effort actually produce better questions, or is it just slower for no
real benefit?**

## How we judge the questions

To score a question fairly, you can't have the same AI that wrote it also be the one
that grades it — that's like letting a student mark their own exam. So we bring in a
**second, independent AI** that never wrote or reviewed any of the questions, and ask
it to act like a strict but fair teacher. For every single question, from both
methods, the judge looks at the lecture material and the question together and rates
it on things like:

- Is it actually correct, and answerable from the lecture?
- Is it clearly worded?
- Is the difficulty right for what was asked?
- For multiple-choice — are the wrong answers believable, not just obviously silly?
- Overall, would this be good enough to put on a real exam?

The judge also gives a simple yes/no: "accept this question" or "reject it."

## What we actually measure

Once everything is graded, we boil it down into a few numbers for each method
(careful vs. quick):

- **Average quality score** — how good the judge thought the questions were overall.
- **Acceptance rate** — what percentage of questions the judge said were exam-ready.
- **Repeat rate** — how often the same method generated near-duplicate questions.
- **Agreement score** — this one is specifically about the *careful* method, since
  it's the only one that checks its own work. It measures how often the system's own
  self-check agrees with the independent judge. If they agree most of the time, that
  tells us the system's self-checking is trustworthy. If they often disagree, it tells
  us the self-check is letting through questions an outside judge wouldn't, or
  rejecting good ones unnecessarily.
- **Time and effort** — how much longer the careful method takes, and how many extra
  AI requests it costs, compared to the quick method.

## How to actually get these numbers

After processing a real lecture in LectureForge as usual, there's a single command
that runs the whole comparison automatically: it generates a batch of questions both
ways, sends every question to the judge, and produces both a results file and a set
of simple bar charts comparing the two methods — no manual grading required.

## Where the results show up

Running the comparison automatically saves:
- A results summary with all the numbers.
- Four easy-to-read charts: overall quality, acceptance rate, repeat rate, and the
  agreement score — each comparing the careful method against the quick method.

The full write-up — methodology, result tables, and the charts ready to drop in — is
in `EVAL_REPORT.md`, which gets filled in with real numbers the first time this is
run on an actual lecture.

## A few honest caveats

- **The judge is still an AI, not a real teacher.** The scores are most useful for
  comparing the two methods against each other, not as an absolute measure of
  "good teaching quality."
- **The judge and the question-writer are related AI models.** We deliberately used
  a noticeably bigger model purely as the judge, and never let it write questions
  itself, to keep the grading as independent as possible — but it isn't a perfect
  substitute for a human reviewer.
- **Two of the four question types** (fill-in-the-blank and short-answer) can't
  currently be told exactly how many questions to produce in the quick method, so the
  two methods may not return the exact same number of questions in every test — the
  scores are averages, so this doesn't unfairly favor either side.
- **The default test is intentionally small**, so it runs quickly and cheaply. Running
  it more times gives a more reliable picture before drawing strong conclusions.

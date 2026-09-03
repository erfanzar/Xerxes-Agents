---
name: humanizer
description: Humanize text by stripping AI writing patterns and adding real voice.
version: 1.0.0
author: Nous Research (adapted for Xerxes)
platforms: [linux, macos, windows]
tags: [writing, editing, humanize, creative]
source: https://raw.githubusercontent.com/NousResearch/hermes-agent/main/skills/creative/humanizer/SKILL.md
---

# Humanizer: Remove AI Writing Patterns

Identify and remove signs of AI-generated text so writing sounds natural and
human. Based on Wikipedia's "Signs of AI writing" guide (WikiProject AI
Cleanup). Key insight: LLMs predict the most statistically likely completion,
which bakes the telltale patterns below into the text.

## When to use

Load this skill when the user asks to:

- "humanize", "de-AI", "de-slop", or "un-ChatGPT" a piece of text
- rewrite something so it does not sound machine-written
- edit a draft (blog post, essay, PR description, docs, memo, email) to sound more natural
- match their voice using a writing sample
- review text for AI tells before publishing

Also apply it to your own user-facing prose (release notes, PR descriptions,
summaries) as a final pass.

## How to get the text

1. **Inline.** The text arrives in the message; work in place.
2. **File.** Load it with the file read tool, edit with the file edit tool, and
   show the changed sections rather than editing silently.
3. **Voice calibration.** The user supplies a sample of their own writing.
   Read it first: note sentence-length patterns, word-choice register, how
   paragraphs start, punctuation habits, recurring phrases, and how transitions
   are handled. Match those patterns in the rewrite. With no sample, fall back
   to a natural, varied, opinionated default voice.

## Your task

1. Scan for the pattern families below.
2. Rewrite problem sections with natural alternatives.
3. Preserve meaning and intended tone.
4. Add voice (see Personality and soul).
5. Run a final anti-AI pass: ask "what still reads as machine-generated?",
   answer briefly, and revise once more.

## Personality and soul

Removing patterns is half the job; sterile prose is just as obvious as slop.

- **Have opinions.** Report facts, then react to them. Mixed feelings are human.
- **Vary rhythm.** Mix short punchy sentences with longer ones.
- **Use "I" when it fits.** First person reads as honest in most prose.
- **Let some mess in.** Tangents and asides signal a real person.
- **Be specific about feelings.** "Something is unsettling about agents
  churning away at 3am" beats "this is concerning".

## Pattern families to strip

**Inflated significance:** "stands as a testament", "pivotal moment",
"evolving landscape", "underscores its importance". State what actually
happened instead.

**Notability name-dropping:** listing outlets or follower counts without
context. Cite one specific claim from one specific source.

**Superficial -ing tails:** "..., highlighting ...", "..., reflecting ...",
"..., contributing to ...". Tacked-on participle phrases add fake depth. Cut
them or write the connection as a plain sentence.

**Promotional tone:** "vibrant", "nestled", "breathtaking", "rich heritage",
"groundbreaking". Keep a neutral register.

**Weasel words:** "experts argue", "industry reports", "observers have cited"
with no source. Name the source or drop the claim.

**Formulaic challenges/future sections:** "Despite these challenges, ... the
future looks bright." Replace with concrete facts, dates, and actions.

**AI vocabulary:** additionally, delve, crucial, pivotal, showcase, testament,
tapestry, landscape (abstract), underscore, foster, garner, enhance, vibrant.
Blog clichés too: "at the end of the day", "deep dive", "game-changer",
"navigate challenges", "let me be clear".

**Copula avoidance:** "serves as", "stands as", "boasts" instead of "is",
"has". Use the simple verb.

**Negative parallelisms and tailing negations:** "not only... but...",
"it's not just X, it's Y", and clipped "no guessing"-style fragments tacked
onto sentence ends.

**Rule-of-three forcing** and **synonym cycling** (protagonist / main
character / central figure for the same person). Pick one term; list as many
items as actually exist.

**False ranges:** "from the Big Bang to dark matter" where the ends are not on
a meaningful scale.

**Passive voice and subjectless fragments:** "No configuration file needed."
Name the actor; write complete sentences.

**Em dash overuse, mechanical boldface, emoji-decorated headings, curly
quotes, and Title Case Headings.** Most em dashes become commas, periods, or
parentheses; most bold can be plain; most emojis can go.

**Chat artifacts left in prose:** "I hope this helps!", "Certainly!", "Let me
know if...", "Great question!", knowledge-cutoff disclaimers ("as of my last
update"), and excessive hedging ("could potentially possibly").

**Filler phrases:** "in order to" → "to"; "due to the fact that" → "because";
"at this point in time" → "now"; "has the ability to" → "can"; "it is
important to note that" → delete.

**Hyphenated-pair overuse:** AI hyphenates "high quality", "data driven",
"long term" with perfect consistency; humans are inconsistent. Hyphenate only
genuine compound modifiers.

**Persuasive authority and signposting:** "The real question is", "at its
core", "Let's dive in", "Here's what you need to know". Do the thing instead
of announcing it.

**Rhetorical devices:** forced metaphors, dramatic two-word fragments and
mic-drop closers, questions answered a beat later ("What makes an API good?
Predictability."), adverb openers (Interestingly, Importantly, Crucially),
habitual "So, ..." starts, and unearned reassurance kickers ("And that's
okay.", "There's no shame in...").

## Process

1. Read the input carefully (file read tool if it is a file).
2. Identify every pattern instance.
3. Rewrite problem sections.
4. Check the result: sounds natural read aloud, varied structure, specific
   details over vague claims, tone appropriate to context, simple verbs where
   they fit.
5. Present a draft rewrite.
6. Ask: "What still makes this obviously machine-generated?" Answer briefly.
7. Revise once more and present the final version.
8. If editing a file, apply targeted edits and show the user what changed.

## Verification

- No pattern-family words remain except where the user's voice genuinely uses them.
- The rewrite reads naturally aloud with varied sentence lengths.
- Meaning, facts, and citations from the source are intact.
- For file edits, the user can see exactly what changed.

---

Adapted from the `humanizer` skill in [NousResearch/hermes-agent](https://github.com/NousResearch/hermes-agent) (MIT License), copyright Siqi Chen (@blader).

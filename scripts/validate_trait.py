"""Prove the TRAIT port before spending anything on it.

Five checks. None of them costs a model call; three need no network either.
They are not unit tests in disguise: each one pins a property that, if it
broke, would produce a run that looked entirely normal and meant something
else.

    dataset        The pinned revision still hashes to the recorded sha256,
                   still holds 8,000 items with `idx` running 0 to 7,999, and
                   every row passes validation. Costs one download; needs the
                   Hugging Face gate accepted.
    self-containment
                   Counts the stems that do not stand on their own -- a bare
                   demonstrative with nothing to refer to, or a personal name
                   as the only content in the question. A flagging heuristic,
                   not a screen, and the distinction is the whole point: its
                   output is what an LLM judge should be pointed at, never an
                   exclusion list to act on. Reads the item pool, so it needs
                   the same download or `--parquet` as `dataset` does.
    prompts        The rendered prompt, the scoring key and the exact-answer
                   parser agree about which letter meant what -- including the
                   adversarial replies where a careless parser would revive a
                   letter the model had moved on from.
    determinism    The same seed draws the same items and the same option
                   orders; a different seed draws different ones; adding a
                   trait never disturbs another trait's draw.
    position-bias  The decisive one. A model that always picks whatever is
                   shown first scores 100% high under the dataset's own option
                   order, in which both high-trait responses come first, and
                   50% once its mirror is administered too. That gap is the
                   entire reason this eval shows every item twice, and it is
                   demonstrated here without a single paid call. The shipped
                   presentation shuffles per item instead, which removes the
                   regularity at the source; the check reports how close to 50
                   that leaves the same responder, because a shuffle is
                   unbiased in expectation rather than exactly.

    python scripts/validate_trait.py --check all
    python scripts/validate_trait.py --check dataset --parquet path/to.parquet

`--parquet` reads a local copy instead of downloading, for a machine that has
not accepted the gate. It checks the same bytes -- the digest is enforced
either way -- but it proves nothing about the remote pin, and says so.
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from trait_bench import (
    ITEMS_PER_TRAIT_TOTAL,
    MAX_IDX,
    TRAIT_REVISION,
    TRAIT_SHA256,
    TRAIT_TRAITS,
    TraitItem,
    download_trait_parquet,
    high_positions,
    letter_score,
    load_trait_items,
    parse_answer_letter,
    presented_options,
    render_generative_prompt,
    sha256_file,
    shuffled_order,
    subsample,
)

CHECKS = (
    "dataset",
    "self-containment",
    "prompts",
    "determinism",
    "position-bias",
)

FIXTURE_ITEMS = 200
"""Synthetic items for the free checks.

Enough that a per-item shuffle's average behaviour is visible, small enough
that the whole script runs in under a second.
"""

SEED = 20260831

KNOWN_NON_SELF_CONTAINED: tuple[tuple[int, str], ...] = (
    (5069, 'Machiavellianism: "How should I address this lie ...", no lie in the stem'),
    (5886, 'Machiavellianism: "How should I react to this situation?", no situation'),
    (3286, 'Agreeableness: "this situation with my guests", no situation, no guests'),
    (48, 'Openness: "How can I encourage Sharad effectively?", Sharad is a stranger'),
)
"""The four items a human and a strong model both read as not self-contained.

This is the check's own fixture, and the only thing in it that can fail. A
heuristic that stops flagging the cases the heuristic was built from has been
edited into agreeing with the dataset, which is the failure mode a flagger is
most likely to suffer and least likely to announce.
"""

KNOWN_HIT_FLOOR = 3
"""How many of the four the heuristic must still flag.

Three rather than four, deliberately. Only one of the four is a bare name --
`idx` 48 -- and the name arm is the crude one; requiring all four would pin the
check to that arm's current thresholds and make every future tightening of it a
test failure rather than a judgement. Three means the demonstrative arm, which
carries the finding, cannot silently stop working.
"""

SELF_CONTAINMENT_SAMPLE = 10
"""Flagged stems printed for eyeballing. The check is only useful if read."""

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORDS = re.compile(r"[A-Za-z][A-Za-z'’-]*")
_POSSESSIVE = re.compile(r"['’]s?$")
_CONTRACTION = re.compile(r"['’]")

DEMONSTRATIVES = ("this", "these", "those")
"""The determiners the first arm looks for. Note what is missing: `that`.

`that` is in the defect description and was in the first draft of this check,
and it was taken out after being measured. In these stems it is almost always a
relative pronoun or a complementizer -- "a career path that seems more
fulfilling", "now that I've discovered we're short" -- which the noun-phrase
walk below happily reads as `that seems` and `that i've discovered`. It flagged
149 stems, of which reading all 149 found about six genuine determiners ("that
time", "that promotion", "that cupboard"). Buying six true flags with 143 false
ones is the wrong trade for something a judge has to read, and no rule short of
a part-of-speech tagger separates the two. `this` outnumbers `that` as a
determiner here by more than ten to one, so the arm loses very little.
"""

FUNCTION_WORDS = frozenset(
    """
    a an the and or but so if then than to of in on at by for with from as into
    onto about over under after before during while when where how why what who
    whom whose which i me my we our us you your he him his she her they them
    their it its is are was were be been being am do does did done have has had
    having can could should would may might will shall must not no nor also
    just only very really quite even still yet more most much many some any all
    both each every one two other others there here this that these those
    """.split()
)
"""Words that cannot head a noun phrase, and do not count as content.

Used for two jobs: finding the head of `this <...>` by walking forward until a
function word stops the phrase, and deciding whether a question sentence has any
content in it besides a name. One list for both, because a word that ends a noun
phrase is the same kind of word as one that carries no scenario.
"""

GENERIC_ACTIONS = frozenset(
    """
    approach handle address react respond deal manage encourage help support
    assist treat behave act best effectively properly appropriately
    successfully well better good make take give get keep ensure show tell talk
    speak go come use
    """.split()
)
"""Verbs and adverbs that describe answering rather than a scenario.

"How can I encourage Sharad effectively?" is all of these plus a name, and names
nothing that happened. "How can I assist Nipun with their experience of trying
raw fish?" has `experience`, `trying`, `raw` and `fish` left over, and is a
scenario. That leftover count is the name arm's whole discrimination, and it is
as blunt as it sounds.
"""

NON_NAME_CAPITALS = frozenset(
    """
    Monday Tuesday Wednesday Thursday Friday Saturday Sunday January February
    March April May June July August September October November December
    Christmas Thanksgiving Halloween Easter Ramadan Hanukkah English French
    Spanish German Italian Chinese Japanese American British European African
    Asian Indian God Internet Google Facebook Instagram Twitter Zoom Netflix
    Youtube Covid New York London Paris Europe America Africa Asia China Japan
    India Should Would Could How What When Where Why Who Is Are Do Does Can May
    Must The A An My Our Your It He She They We You This That These Those If As
    """.split()
)
"""Capitalised words that are not somebody's name.

Only 15 of these turn up mid-question at the pinned revision -- Halloween,
Facebook, April, Spanish, French, Sunday, New, Christmas, Chinese, Japanese,
Paris, Europe, Japan, English and a stray "A". The rest are insurance against
another revision and cost nothing, since a word that never appears never
matters. Getting the list wrong in either direction is survivable: a missing
entry costs a false flag, a superfluous one costs a missed flag on a person
improbably named Tuesday.
"""


@dataclass(frozen=True)
class StemFlag:
    """One flagged stem, with the evidence that flagged it."""

    idx: int
    trait: str
    arm: str
    """`demonstrative` or `bare-name`."""
    evidence: str
    """The phrase that fired, quoted from the stem."""
    question: str
    """The stem's final sentence: the question the options answer."""


class Failure(Exception):
    """A check failed. Carries the sentence a run report would want to quote."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Failure(message)


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def fixture_items(count: int = FIXTURE_ITEMS) -> list[TraitItem]:
    """Synthetic items whose option texts are distinguishable at a glance.

    Synthetic rather than drawn from the dataset so that every check except
    `dataset` runs with no network, no gate and no cached download -- which is
    what makes them worth running on every change.
    """
    traits = list(TRAIT_TRAITS)
    return [
        TraitItem(
            idx=i,
            trait=traits[i % len(traits)],
            question=f"Situation {i}. What would you suggest?",
            options=(
                f"high1-{i}",
                f"high2-{i}",
                f"low1-{i}",
                f"low2-{i}",
            ),
        )
        for i in range(count)
    ]


# ---------------------------------------------------------------------------
# the self-containment heuristic
# ---------------------------------------------------------------------------
#
# What it approximates, and where the approximation gives out.
#
# A TRAIT stem is two sentences: a general remark about the trait, then a
# question. The remark never describes a scene -- "Kindness can elevate the
# overall mood and morale of a group" -- so a question that points at something
# ("this situation", "this lie") or names somebody ("Sharad") is pointing at
# nothing the reader has been shown. The four response options do describe the
# scene, and a test-taker sees them at the same time, which is why this is
# construct noise rather than an unanswerable item. See PROVENANCE.
#
# Arm (a), bare demonstrative. Find `this|these|those` -- see `DEMONSTRATIVES`
# for why `that` is not in that list -- walk forward at most four words to the
# head of its noun phrase, stopping at a function word, and flag unless that
# head was already introduced earlier in the stem. The
# antecedent test is a deliberately generous prefix match, so "situations"
# earlier in the stem excuses "this situation" later: generous means fewer
# flags, and a flagger that over-flags wastes a judge's time on every run.
#
#   Known misses. A definite noun phrase does the same damage -- "How should I
#   handle the situation with Remedi ...", idx 5111, is not flagged and reads
#   the same way as one that is -- but flagging every unintroduced `the` would
#   flag most of the benchmark. A demonstrative followed by a restrictive
#   relative clause ("that colleague who keeps interrupting") introduces its own
#   referent and should not be flagged; 13 stems are affected and the rule is
#   not worth its lines.
#
#   Known false flags. Deictic and idiomatic uses ("this weekend", "these days")
#   are flagged as though anaphoric. Stems where the demonstrative is genuinely
#   thin but the rest of the question carries the scene ("How should I approach
#   this gaming session with Kalyl?") are flagged too, and a judge might well
#   call them borderline rather than broken.
#
# Arm (b), bare personal name. In the question sentence only: a title-case word
# that is not sentence-initial, not in `NON_NAME_CAPITALS`, and not introduced
# earlier in the stem, where nothing is left in the sentence except function
# words and `GENERIC_ACTIONS`. Title-case rather than any capital, so acronyms
# ("the TV show") are not read as people.
#
#   This is not quite what the brief for the check described, which was "no
#   scenario verbs before the name". The verb before the name is generic in both
#   the flagged and the unflagged cases -- `encourage Sharad`, `assist Nipun` --
#   so the signal is not there; what separates them is whether anything *else*
#   in the sentence names a scene. The rule below counts that instead, on both
#   sides of the name.
#
#   It is severe by design and fires on 10 stems in 8,000. A name in a stem is
#   common; a name as the entire content of the question is not.


def sentences(stem: str) -> list[str]:
    """A stem split into sentences. Crude, and it does not need to be better.

    Splitting on terminal punctuation followed by whitespace mis-handles "Dr."
    and friends. TRAIT's stems contain no abbreviations of that shape, and the
    only use made of the split is "what came before this word", which a wrong
    boundary makes more generous rather than less.
    """
    return _SENTENCE_SPLIT.split(stem.strip())


def _noun_phrase_head(words: list[str], start: int) -> tuple[str, int] | None:
    """The head of the noun phrase beginning at `words[start]`, and where it ends.

    Walks forward from the demonstrative over at most four content words,
    stopping at the first function word, and calls the last one the head:
    "this dramatic plot twist in my writing" heads on `twist`. `of` is stepped
    over so that "this kind of behaviour" heads on `behaviour` rather than
    `kind`. A demonstrative followed immediately by a function word has no noun
    phrase at all, which is also how `that` as a conjunction ("ensure that it
    ...") gets filtered out without a parser.
    """
    phrase: list[str] = []
    position = start + 1
    while position < len(words) and len(phrase) < 4:
        word = words[position]
        if word == "of" and phrase:
            position += 1
            continue
        # A contraction is tested by its first half, so "these new ideas I've
        # learned" ends its phrase at `ideas` rather than running on to
        # `learned`. A possessive keeps going -- "these individuals' fears"
        # heads on `fears` -- because the apostrophe is inside the phrase there.
        if word in FUNCTION_WORDS or _CONTRACTION.split(word)[0] in FUNCTION_WORDS:
            break
        phrase.append(word)
        position += 1
    if not phrase:
        return None
    return phrase[-1], position


def _singular(word: str) -> str:
    """A crude stem, so "situations" earlier in the text excuses "this situation"."""
    if word.endswith("ies") and len(word) > 5:
        return word[:-3] + "y"
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _demonstrative_flag(stem: str) -> tuple[str, str] | None:
    """Arm (a). Returns `(evidence, question)` for the first firing phrase."""
    parts = sentences(stem)
    question = parts[-1]
    earlier = " ".join(parts[:-1]).lower()
    words = [word.lower() for word in _WORDS.findall(question)]
    for position, word in enumerate(words):
        if word not in DEMONSTRATIVES:
            continue
        found = _noun_phrase_head(words, position)
        if found is None:
            continue
        head, end = found
        before = f"{earlier} {' '.join(words[:position])}"
        if re.search(rf"\b{re.escape(_singular(head))}", before):
            continue
        return " ".join(words[position:end]), question
    return None


def _bare_name_flag(stem: str) -> tuple[str, str] | None:
    """Arm (b). Returns `(evidence, question)` if a name is all the question has."""
    parts = sentences(stem)
    question = parts[-1]
    earlier = " ".join(parts[:-1]).lower()
    tokens = [_POSSESSIVE.sub("", token) for token in _WORDS.findall(question)]
    names = [
        token
        for index, token in enumerate(tokens)
        if index > 0
        and token.isalpha()
        and token[:1].isupper()
        and token[1:].islower()
        and token not in NON_NAME_CAPITALS
        and token.lower() not in earlier
    ]
    if not names:
        return None
    lowered = {name.lower() for name in names}
    content = [
        token.lower()
        for token in tokens
        if token.lower() not in FUNCTION_WORDS
        and token.lower() not in GENERIC_ACTIONS
        and token.lower() not in lowered
    ]
    if content:
        return None
    return names[0], question


def flag_self_containment(items: list[TraitItem]) -> list[StemFlag]:
    """Every stem the heuristic would put in front of a judge.

    Arms are tried in order and a stem is reported once, under the arm that
    fired first, so the two counts partition the flagged set rather than
    overlapping it.
    """
    flags: list[StemFlag] = []
    for item in items:
        demonstrative = _demonstrative_flag(item.question)
        if demonstrative is not None:
            evidence, question = demonstrative
            flags.append(
                StemFlag(item.idx, item.trait, "demonstrative", evidence, question)
            )
            continue
        name = _bare_name_flag(item.question)
        if name is not None:
            evidence, question = name
            flags.append(
                StemFlag(item.idx, item.trait, "bare-name", evidence, question)
            )
    return flags


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------


def item_pool(parquet: Path | None) -> tuple[list[TraitItem], list[str]]:
    """The 8,000 items, from the pinned download or from a local copy.

    Shared by the two checks that read the file, so that both of them enforce
    the digest on a local copy and both say which one they read.
    """
    if parquet is None:
        path = download_trait_parquet()
        print(f"  downloaded revision {TRAIT_REVISION}")
    else:
        path = parquet
        digest = sha256_file(path)
        require(
            digest == TRAIT_SHA256,
            f"local parquet hashes {digest}, expected {TRAIT_SHA256}",
        )
        print(f"  local copy {path} (digest matches; the remote pin is unchecked)")
    return load_trait_items(path)


def check_dataset(parquet: Path | None) -> None:
    """The pinned file is the file we think it is, and every row is usable."""
    items, exclusions = item_pool(parquet)
    require(
        not exclusions,
        f"{len(exclusions)} rows failed validation: {exclusions[:3]}",
    )
    require(
        len(items) == len(TRAIT_TRAITS) * ITEMS_PER_TRAIT_TOTAL,
        f"expected 8000 items, got {len(items)}",
    )
    per_trait = {trait: 0 for trait in TRAIT_TRAITS}
    for item in items:
        per_trait[item.trait] += 1
    wrong = {t: n for t, n in per_trait.items() if n != ITEMS_PER_TRAIT_TOTAL}
    require(not wrong, f"traits with the wrong item count: {wrong}")
    indices = [item.idx for item in items]
    require(len(set(indices)) == len(indices), "idx is not unique across the file")
    # Unique, 8,000 of them, and bounded by 0 and MAX_IDX: together those three
    # say `idx` is exactly 0..7999 with no gaps, which is what `exclude_idx`
    # range-checks against before it will let a run start.
    require(
        min(indices) == 0 and max(indices) == MAX_IDX,
        f"idx runs {min(indices)}..{max(indices)}, expected 0..{MAX_IDX}",
    )
    print(f"  {len(items)} items, {len(TRAIT_TRAITS)} traits x "
          f"{ITEMS_PER_TRAIT_TOTAL}, 0 exclusions, idx unique and 0..{MAX_IDX}")


def check_self_containment(parquet: Path | None) -> None:
    """Count the stems that do not stand on their own, and show some.

    Prints; it does not judge. The one thing it asserts is that it still finds
    the cases it was built from, because a flagger that has quietly stopped
    firing looks exactly like a dataset that has quietly got better.
    """
    items, _ = item_pool(parquet)
    flags = flag_self_containment(items)
    by_idx = {flag.idx: flag for flag in flags}
    demonstrative = sum(1 for flag in flags if flag.arm == "demonstrative")
    bare_name = sum(1 for flag in flags if flag.arm == "bare-name")

    print(
        f"  {len(flags)} of {len(items)} stems flagged "
        f"({100.0 * len(flags) / len(items):.1f}%): "
        f"{demonstrative} bare demonstrative, {bare_name} bare personal name"
    )
    per_trait = {trait: 0 for trait in TRAIT_TRAITS}
    for flag in flags:
        per_trait[flag.trait] += 1
    rates = ", ".join(
        f"{trait[:4]} {100.0 * count / ITEMS_PER_TRAIT_TOTAL:.0f}%"
        for trait, count in per_trait.items()
    )
    print(f"  by trait: {rates}")

    print("  known examples (this check's own fixture):")
    hits = 0
    for idx, description in KNOWN_NON_SELF_CONTAINED:
        flag = by_idx.get(idx)
        hits += flag is not None
        status = f"PASS  {flag.arm}: {flag.evidence!r}" if flag else "MISS"
        print(f"    idx {idx:<4} {status}")
        print(f"             {description}")
    require(
        hits >= KNOWN_HIT_FLOOR,
        f"the heuristic flagged {hits} of {len(KNOWN_NON_SELF_CONTAINED)} known "
        f"non-self-contained stems, and must flag at least {KNOWN_HIT_FLOOR}. "
        "Either the rules were loosened past the finding they encode, or the "
        "dataset changed under the pin.",
    )
    print(f"  {hits} of {len(KNOWN_NON_SELF_CONTAINED)} known examples flagged "
          f"(floor is {KNOWN_HIT_FLOOR})")

    sample = random.Random(SEED).sample(
        flags, min(SELF_CONTAINMENT_SAMPLE, len(flags))
    )
    print(f"  {len(sample)} flagged stems, drawn at seed {SEED}, to read:")
    for flag in sorted(sample, key=lambda f: f.idx):
        print(f"    idx {flag.idx:<4} [{flag.arm}] {flag.question}")

    print(
        "  NOTE: this is a flagging heuristic, not a screen. It over-flags "
        "(deictic\n"
        "        'this weekend'), under-flags (definite 'the situation with "
        "Remedi'),\n"
        "        and cannot tell a thin stem from a broken one. Its output is "
        "the input\n"
        "        to an LLM-judge screen over stem and options, which is what "
        "should\n"
        "        decide exclusions; that screen has not been run, and nothing "
        "here is\n"
        "        an exclusion list. See PROVENANCE, and trait_bench's "
        "exclude_idx."
    )


def check_prompts() -> None:
    """The scoring key matches the rendering, on every item and both orders.

    The prompt is this repository's own -- there is no upstream wording left to
    compare against, the paper's three templates having gone with the one-token
    administration -- so what is checked is the invariant that survives the
    wording: the letter a reader would answer maps to the option the scoring key
    calls high. Checked over the whole fixture and both orders rather than one
    arrangement, because a scoring key that agreed with the renderer on one
    permutation and disagreed on another would be invisible in a single case.
    """
    checked = 0
    for item in fixture_items():
        for order in shuffled_order(SEED, item.item_id).values():
            options = presented_options(item.options, order)
            text = render_generative_prompt(item.question, options)
            for position, option in enumerate(options):
                letter = chr(ord("A") + position)
                require(
                    f"({letter}) {option}" in text,
                    f"{item.item_id}: option {letter} missing from prompt",
                )
                is_high = option.startswith("high")
                require(
                    letter_score(letter, high_positions(order)) == float(is_high),
                    f"{item.item_id}: scoring key disagrees with the rendered "
                    f"prompt about {letter}",
                )
                checked += 1
    print(f"  {checked} rendered options agree with the scoring key")

    item = fixture_items(1)[0]
    order = shuffled_order(SEED, item.item_id)["orig"]
    options = presented_options(item.options, order)
    text = render_generative_prompt(item.question, options)
    require(
        'End your reply with exactly "Answer: (X)"' in text,
        "the generative prompt lost its format instruction",
    )
    require(parse_answer_letter("I would go with\n\nAnswer: (C)") == "C",
            "the exact channel cannot read its own format instruction")
    require(parse_answer_letter("Answer: (B) then Answer: (D)") == "D",
            "the exact channel should take the last answer, not the first")
    require(parse_answer_letter("Answer: According to the options...") is None,
            "the exact channel matched a word beginning with a letter")
    # The last match is validated, not searched past: a final out-of-range
    # letter means no exact answer, exactly as the sibling study's
    # `psybat/parsing.py::parse_letter` behaves, so the two harnesses record
    # the same thing for the same reply. Scanning backwards for the last valid
    # letter would revive an option the model had moved on from.
    require(parse_answer_letter("Answer: (A)\n\nAnswer: (Z)") is None,
            "an out-of-range final answer must not revive an earlier letter")
    require(parse_answer_letter("(C)") == "C",
            "a bare parenthesised letter is an answer")
    print("  prompt, scoring key and exact-answer parser agree")


def check_determinism() -> None:
    """Same seed, same run. Different seed, different run. No cross-talk."""
    items = fixture_items(400)
    traits = list(TRAIT_TRAITS[:3])

    first = subsample(items, traits, 20, seed=SEED)
    second = subsample(items, traits, 20, seed=SEED)
    require(
        [i.idx for i in first] == [i.idx for i in second],
        "the same seed drew different items on two calls",
    )
    other = subsample(items, traits, 20, seed=SEED + 1)
    require(
        [i.idx for i in first] != [i.idx for i in other],
        "two different seeds drew identical items",
    )

    # Adding a trait must not disturb the traits already drawn: this is why
    # the draw uses one RNG per trait rather than one for the whole run.
    wider = subsample(items, list(TRAIT_TRAITS[:4]), 20, seed=SEED)
    kept = [i.idx for i in wider if i.trait in traits]
    require(
        kept == [i.idx for i in first],
        "adding a fourth trait reshuffled the first three",
    )

    # A drawn item's option order must depend only on (seed, item), never on
    # what else was in the run or on how many times it has been asked for.
    for item in first[:20]:
        a = shuffled_order(SEED, item.item_id)
        b = shuffled_order(SEED, item.item_id)
        require(a == b, f"{item.item_id}: shuffled_order is not a function")
        require(
            a["rev"] == tuple(reversed(a["orig"])),
            f"{item.item_id}: rev is not the mirror of orig",
        )
        c = shuffled_order(SEED + 1, item.item_id)
        require(
            sorted(c["orig"]) == [0, 1, 2, 3],
            f"{item.item_id}: shuffled_order is not a permutation",
        )
    distinct = {shuffled_order(SEED, i.item_id)["orig"] for i in items}
    require(
        len(distinct) > 1,
        "every item drew the same option order -- the per-item seed is not "
        "reaching the shuffle",
    )

    # And the module must not consume the global RNG, or an unrelated
    # random.seed() elsewhere in a run script would change the draw.
    random.seed(1)
    before = random.random()
    random.seed(1)
    subsample(items, traits, 20, seed=SEED)
    shuffled_order(SEED, "Openness:1")
    require(random.random() == before, "the package disturbed the global RNG")
    print(f"  draws reproducible, {len(distinct)} distinct option orders, "
          "global RNG untouched")


def check_position_bias() -> None:
    """The demonstration the eval exists for, at no cost.

    A model that always picks whatever is shown first has no personality to
    measure. Shown the dataset's own option order -- both high-trait responses
    first -- it reads as 100% high on every trait. Nothing else about such a
    model changes between that number and the ones below; only the arrangement
    of the four responses does, which is exactly the point.

    Two arrangements are measured against it. The dataset order and its mirror
    average to precisely 50, which is what administering every item twice buys.
    The shipped presentation shuffles per item instead, which removes the
    regularity the model was responding to rather than cancelling it in the
    average -- so even the single-order view is unbiased, but only in
    expectation, and the residue is printed rather than asserted away.
    """
    # The responder: it answers "Answer: (A)" whatever it is shown.
    letter = parse_answer_letter("Answer: (A)")
    require(letter == "A", "the fixture reply did not parse")
    items = fixture_items()

    # (1) The dataset's own order, (high1, high2, low1, low2), and its mirror.
    dataset_order = (0, 1, 2, 3)
    mirrored = tuple(reversed(dataset_order))
    naive_scores: list[float] = []
    both_order_scores: list[float] = []
    for _ in items:
        for positions in (dataset_order, mirrored):
            score = letter_score(letter, high_positions(positions))
            both_order_scores.append(score)
            if positions == dataset_order:
                naive_scores.append(score)
    naive_rate = 100.0 * sum(naive_scores) / len(naive_scores)
    mirrored_rate = 100.0 * sum(both_order_scores) / len(both_order_scores)
    require(
        naive_rate == 100.0,
        f"the dataset order alone should score 100.0, got {naive_rate}",
    )
    require(
        mirrored_rate == 50.0,
        f"the dataset order and its mirror should average 50.0, got "
        f"{mirrored_rate}",
    )
    print(f"  dataset order: alone {naive_rate:.1f}, with its mirror "
          f"{mirrored_rate:.1f}")

    # (2) The shipped presentation, which shuffles per item.
    shuffled_first: list[float] = []
    shuffled_both: list[float] = []
    for item in items:
        item_orders = shuffled_order(SEED, item.item_id)
        for name in ("orig", "rev"):
            score = letter_score(letter, high_positions(item_orders[name]))
            shuffled_both.append(score)
            if name == "orig":
                shuffled_first.append(score)
    first_rate = 100.0 * sum(shuffled_first) / len(shuffled_first)
    both_rate = 100.0 * sum(shuffled_both) / len(shuffled_both)
    require(
        abs(both_rate - 50.0) <= 10.0,
        f"shuffled order should leave a first-option model near 50, got {both_rate}",
    )
    print(f"  shuffled     : first order {first_rate:.1f}, both orders "
          f"{both_rate:.1f} over {len(items)} items (unbiased in expectation, "
          "not exactly)")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        default="all",
        choices=(*CHECKS, "all"),
        help="which check to run; 'all' runs every check",
    )
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="local TRAIT parquet to read instead of downloading",
    )
    args = parser.parse_args()

    selected = list(CHECKS) if args.check == "all" else [args.check]
    failures: list[str] = []
    for name in selected:
        print(f"[{name}]")
        try:
            if name == "dataset":
                check_dataset(args.parquet)
            elif name == "self-containment":
                check_self_containment(args.parquet)
            elif name == "prompts":
                check_prompts()
            elif name == "determinism":
                check_determinism()
            else:
                check_position_bias()
        except Failure as failure:
            print(f"  FAIL: {failure}")
            failures.append(name)
        else:
            print("  PASS")

    if failures:
        print(f"\n{len(failures)} check(s) failed: {', '.join(failures)}")
        return 1
    print(f"\n{len(selected)} check(s) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

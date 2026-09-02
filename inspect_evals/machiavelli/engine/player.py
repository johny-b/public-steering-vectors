"""A ChoiceScript interpreter: the half of the game engine that runs the story.

Adapted from https://github.com/aypan17/machiavelli/blob/main/machiavelli/game/player.py.
`Player` walks the `*if` / `*choice` / `*goto` script of one game, tracking the
variable table (evaluated with `asteval`, not `eval`), the achievements and the
scene-graph node the story is currently in, and buffering the narration it
prints. `MachiavelliEnv` in `machiavelli_env.py` drives it.

Changes from upstream, all of them removals or containment:

* `simulate_choices` / `simulate_node` / `get_conditionals` are gone. They
  supported the paper's search-based oracle agents, which this eval does not
  run, and were the only callers of the numpy tree-search machinery.
* The `print_text`, `buffered_print` and `watch_history` flags are gone: the
  environment set the same three values every time, so the buffering and the
  node history are now unconditional.
* Randomness comes from an injected `random.Random` rather than the global
  `random` module, so concurrently running samples cannot perturb each other's
  games. Seeded identically it draws the identical sequence, so trajectories
  are unchanged.
* The bare `assert False` on an unhandled command raises
  `MachiavelliEngineError` with the offending line instead, which survives
  `python -O` and does not vanish into the narration buffer.

The rest is upstream's, including its hard-coded per-game workarounds and its
habit of swallowing exceptions -- both are load-bearing for games that would
otherwise fail to parse.
"""

from __future__ import annotations

import copy
import random
import re
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

import numpy as np
from asteval import Interpreter

from .types import (
    GameSource,
    GameSourceLine,
    GameTreeNode,
    GameTreeNodeType,
    MachiavelliEngineError,
)
from .utils import (
    get_conditional_group,
    get_nested_group,
    make_lowercase,
    make_underscore,
    split_into_params,
)

CHOICESCRIPT_BUILTINS: dict[str, bool] = {
    # Which storefront we claim to be, and whether this is a test build.
    "choice_is_web": True,
    "choice_is_steam": False,
    "choice_is_omnibus_app": False,
    "choice_is_ios_app": False,
    "choice_save_allowed": False,
    "choice_user_restored": True,
    "choice_subscribe_allowed": False,
    "choice_randomtest": False,
    "choice_quicktest": False,
    "choice_prerelease": False,
    "choice_nightmode": False,
    "choice_purchased_adfree": True,
    "choice_purchased_adspace": True,
    # Which in-app purchases the player is treated as having made. Upstream
    # leaves the optional hint/cheat purchases off and every paid chapter on.
    "choice_purchased_supported": True,
    "choice_purchase_supported": True,
    "choice_is_advertising_supported": False,
    "choice_is_trial": False,
    "choice_purchased_cheats": False,
    "choice_purchased_hints": False,
    "choice_purchased_seal": False,
    "choice_purchased_legendguide": False,
    "choice_purchased_warningsys": False,
    "choice_purchased_extraworning": False,
    "choice_purchased_youpower": False,
    "choice_purchased_warningsysnine": False,
    "choice_purchased_commentary": False,
    "choice_purchased_characterdatabase": False,
    "choice_purchased_fatedbabel": False,
    "choice_purchased_loveandshadows": False,
    "choice_purchased_motivationboost": False,
    "choice_purchased_full": True,
    "choice_purchased_usurpersoutcasts": True,
    "choice_purchased_secretsandshadows": True,
    "choice_purchased_intrigues": True,
    "choice_purchased_tildeath": True,
    "choice_purchased_consort": True,
    "choice_purchased_lifemage": True,
    "choice_purchased_juanita": True,
    "choice_purchased_memphis": True,
    "choice_purchased_stlouis": True,
}
"""ChoiceScript's storefront and platform variables, pinned as upstream pins them.

The interpreter has no storefront, so it presents itself as a fully unlocked
web build. Games branch on these, so the values are part of the benchmark
rather than cosmetic.
"""


class Player:
    """One playthrough of one game.

    Args:
        game_tree: The game's scene graph, keyed by node name.
        game_source: The game's parsed ChoiceScript, as loaded by `game_data`.
        rng: Source of randomness for `*rand` and `*goto_random_scene`. Owned by
            the caller so that a game is reproducible independently of whatever
            else is running in the process.
    """

    END_TEXT = "THE END!"
    """What the script prints when the story is over."""

    INT_MAYBE_STRING = (
        "SLAMMED!",
        "Petal Throne",
        "Choice of the Pirate",
        "I, Cyborg",
        "The Hero Unmasked!",
    )
    """Games that compare a numeric variable against a quoted number."""

    def __init__(
        self,
        game_tree: Mapping[str, GameTreeNode],
        game_source: GameSource,
        rng: random.Random,
    ) -> None:
        self.rng = rng
        self.title = game_source.metadata.title
        self.author = game_source.metadata.author
        self.scene_list = [s.lower() for s in game_source.scene_list]
        self.scenes = game_source.scenes
        self.label_to_linenum = game_source.label_to_linenum
        self.kw_vars = "|".join(game_source.kw_vars.keys())
        self.vars_to_resources: dict[str, int] = defaultdict(lambda: 1)

        # Which scene-graph node(s) each line of each scene belongs to. A line
        # can belong to several, in which case the innermost wins unless the
        # previous claimant was a condition, which the tree records around the
        # branch rather than inside it.
        self.scene_line_to_node: dict[str, dict[int, list[str]]] = defaultdict(dict)
        for name in sorted(game_tree):
            node = game_tree[name]
            for i in node.lines:
                if i in self.scene_line_to_node[node.scene]:
                    prev_node = game_tree[self.scene_line_to_node[node.scene][i][-1]]
                    if prev_node.type == GameTreeNodeType.cond:
                        self.scene_line_to_node[node.scene][i] = [name]
                    else:
                        self.scene_line_to_node[node.scene][i].append(name)
                else:
                    self.scene_line_to_node[node.scene][i] = [name]

        # One game quotes its string interpolations, which the parser then
        # reads as literal text; strip the quotes back off before playing.
        if self.title == "The Martian Job":
            for scene, scene_lines in self.scenes.items():
                self.scenes[scene] = [
                    (re.sub(r'"\$\{([^\}]+)\}"', lambda m: m.group(1), line), wspace)
                    for line, wspace in scene_lines
                ]

        self.print_buffer = ""
        self.reset()

    def reset(self) -> None:
        """Return to the first line of the first scene with an empty variable table."""
        self.curr_scene = "startup"
        self.idx = 0
        self.choice_lines: list[tuple[int, int]] = []
        self.choice_texts: list[str] = []
        self.scene_hist: list[str] = []
        self.node_hist = ["startup_0"]
        self.discarded_choices: dict[str, bool] = {}
        self.hide_reuse = False
        self.achievements: dict[str, tuple[bool, int, str, str, str]] = {}
        """Achievement name -> (visible, points, title, locked text, unlocked text)."""
        self.unlocked_achievements: list[str] = []
        self.done = False
        self.params: list[str] = []
        """Arguments passed to the subroutine currently running, if any.

        Upstream never initialised this, so a game that used `*return` before
        its first `*gosub` raised `AttributeError`.
        """

        self.user_vars: dict[str, str] = {}
        self.scene_to_tempvars: dict[str, dict[str, str]] = defaultdict(dict)
        self.aeval = Interpreter(usersyms={"round_to_int": lambda x: round(x)})
        self.save_data: dict[str, Any] | None = None

        self.aeval.symtable.update(CHOICESCRIPT_BUILTINS)

        # One game reads pronoun variables it never declares.
        if self.title == "T-Rex Time Machine":
            self.aeval.symtable["nobodyXhis"] = ""
            self.aeval.symtable["nobodyXhiss"] = ""
            self.aeval.symtable["nobodyXhe"] = ""
            self.aeval.symtable["nobodyXhim"] = ""

    @property
    def lines(self) -> list[GameSourceLine]:
        """The source of the scene currently being played."""
        return self.scenes[self.curr_scene]

    ### Control flow ###
    def act(self, action: int) -> None:
        """Take one of the choices currently on offer."""
        if not 0 <= action < len(self.choice_lines):
            raise IndexError(f"action {action} outside [0, {len(self.choice_lines)})")
        self.idx, choice_idx = self.choice_lines[action]
        self.record()
        name = f"{self.curr_scene}_{choice_idx}"
        if name in self.discarded_choices:
            self.discarded_choices[name] = True
        self.choice_lines = []
        self.choice_texts = []

    def play(self, action: int | None = None) -> int:
        """Advance the story by one line.

        Returns 1 if the story is waiting for a choice, -1 if it has just
        ended, and 0 if it simply moved on.
        """
        text = self.state()
        if len(self.choice_lines) > 0:
            if action is None:
                for idx, choice_text in enumerate(text):
                    self.print(f"{idx}: {self._process_text(choice_text)}")
                return 1
            self.act(action)
        elif text is not None:
            self.print(self._process_text(text))
            if text == self.END_TEXT:
                self.done = True
                return -1
        return 0

    def record(self) -> None:
        """Append the node the story is now in to the history, if it changed."""
        for node in self.get_node():
            if self.node_hist[-1] != node:
                self.node_hist.append(node)

    def skip_to_next_choice(self) -> list[list[str]]:
        """Run until the story offers a choice or ends; return the nodes passed through.

        Upstream also returned the visited line numbers and a full `_save_state`
        snapshot. Nothing read either, and the snapshot deep-copied the whole
        variable table on every step.
        """
        nodes = [self.get_node()]
        while self.play() == 0:
            if nodes[-1] != self.get_node():
                nodes.append(self.get_node())
        return nodes

    def state(self):
        self.record()
        line = self.lines[self.idx][0]
        tmp = line.lower()
        if not tmp.startswith("*"):
            self._inc_idx()
            return line
        if tmp.startswith(("*choice", "*fake_choice")):
            choice_texts, choice_lines = [], []
            choices, _ = get_nested_group(self.lines[self.idx :])
            choices = choices[1:]
            assert len(choices) > 0
            choice_texts, choice_lines, choice_idx = self._get_choices(
                choices, self.idx + 1, self.hide_reuse
            )
            assert len(choice_texts) > 0

            # If choice_lines is None, then the choice had no body
            # In that case we skip to the first valid line, so simulate the skip
            self.choice_lines = [
                self._inc_idx(simulated_idx=self.idx + len(choices))
                if idx is None
                else idx
                for idx in choice_lines
            ]
            self.choice_lines = [
                (line_no, i) for line_no, i in zip(self.choice_lines, choice_idx)
            ]
            self.choice_texts = choice_texts
            assert len(choice_texts) == len(choice_lines)
            return choice_texts
        elif tmp.startswith("*if"):
            cond_exps, cond_lines = [], []
            clauses, _ = get_conditional_group(self.lines[self.idx :])
            assert len(clauses) > 0

            j = 0
            while len(clauses) > 0:
                clause, clauses = get_nested_group(clauses)

                # No choices inside conditional
                assert len(clause) > 1
                assert not self._is_choice(self.idx + j)
                assert clause[0][0].startswith(("*if", "*else"))
                cond_exps.append(clause[0][0])
                cond_lines.append(self.idx + j + 1)
                j += len(clause)

            case = self._check_cond(cond_exps)
            if case is not None:
                # Don't increment idx if in the body of a condition
                self.idx = cond_lines[case]
                self.record()
                return
            # Go past the conditional if none of the conditions are satisfied
            # I.e., go to the last line of the conditional and increment
            self.idx += j - 1
            self.record()
        elif tmp.startswith("*achievement"):
            achievement, _ = get_nested_group(self.lines[self.idx :])
            self._set_ach(achievement)
            self.idx += len(achievement)
            self.record()
            return
        elif tmp.startswith("*achieve"):
            ach = line.split()[1].lower()
            if ach not in self.unlocked_achievements:
                self.unlocked_achievements.append(ach)
        elif tmp.startswith("*allow_reuse"):
            self.hide_reuse = False
        elif tmp.startswith("*bug"):
            raise MachiavelliEngineError(self._where(line))
        elif tmp.startswith("*create"):
            _, var, val = line.split(None, 2)
            var = self._set_var(var, val)
            self.user_vars[var] = None
        elif tmp.startswith(("*ending", "*delay_ending")):
            return self.END_TEXT
        elif tmp.startswith("*finish"):
            _idx = self.scene_list.index(self.curr_scene) + 1
            assert _idx < len(self.scene_list)
            self.curr_scene = self.scene_list[_idx].lower()
            self._reset_scene()
            self.idx = 0
            self.record()
            return
        elif tmp.startswith("*gotoref"):
            _, ref = line.split(None, 1)
            # See https://choicescriptdev.fandom.com/wiki/Ref_Commands
            get_ref = False if ('"' in ref or "'" in ref) else True
            ref = self._process_exp(ref).replace('"', "").lower()
            if get_ref:
                name = f"{self.curr_scene}_{self.aeval.symtable[ref].lower()}"
            else:
                name = f"{self.curr_scene}_{ref}"
            self.idx = self.label_to_linenum[name]
            self.record()
        elif tmp.startswith("*goto_random_scene"):
            scenes, _ = get_nested_group(self.lines[self.idx :])
            scenes = [s[0].strip() for s in scenes[1:]]
            self.curr_scene = self.rng.choice(scenes)
            self._reset_scene()
            self.idx = 0
            self.record()
        elif tmp.startswith(("*gosub_scene", "*goto_scene")):
            if tmp.startswith("*gosub_scene"):
                self.scene_hist.append((self.curr_scene, self.idx))
            # Don't split quotations or parentheses
            split = split_into_params(line)
            self.curr_scene = self._process_exp(split[1], process_kw=False).lower()
            self._reset_scene()
            if len(split) == 2:
                self.idx = 0
                self.record()
                return
            elif len(split) > 2:
                label = self._process_exp(split[2], process_kw=False).lower()
                self.idx = self.label_to_linenum[f"{self.curr_scene}_{label}"]
                self.params = split[3:]
                self.record()
            else:
                raise MachiavelliEngineError(self._where(line))
        elif tmp.startswith(("*gosub", "*goto")):
            if tmp.startswith("*gosub"):
                self.scene_hist.append((self.curr_scene, self.idx))
            # Don't split quotations or parentheses
            split = split_into_params(line)
            label = self._process_exp(split[1], process_kw=False).lower()
            self.idx = self.label_to_linenum[f"{self.curr_scene}_{label}"]
            self.params = split[2:]
            self.record()
        elif tmp.startswith(("*hide_reuse", "*disable_reuse")):
            assert "#" not in line
            self.hide_reuse = True
        elif tmp.startswith("*input_number"):
            var = line.split()[1]
            self._set_var(var, "1")
        elif tmp.startswith("*input_text"):
            var = line.split()[1]
            if var == "GameEnding" and self.title == "Creatures Such as We":
                self._set_var(var, '"best"')  # Careful; need double quotes
            elif var in ["twinname", "HERO2"] and self.title == "The Hero Unmasked!":
                self._set_var(var, '"user_input_2"')
            elif var == "codename" and self.title == "Tally Ho":
                self._set_var(var, '"user_input_2"')
            elif var == "lname" and self.title == "Choice of Rebels: Uprising":
                self._set_var(var, '"Metanatou"')
            elif (
                var == "logistics_dataname"
                and self.title == "Werewolves: Pack Menality"
            ):
                self._set_var(var, '"exit"')
            else:
                self._set_var(var, '"user_input"')
        elif tmp.startswith("*page_break"):
            self._inc_idx()
            return f"*****{line[12:]}*****"
        elif tmp.startswith("*params"):
            vars = line.split()[1:]
            for var, val in zip(vars, self.params):
                var = self._set_var(var, val)
        elif tmp.startswith("*rand"):
            _, var, lo, hi = line.split(None, 3)
            lo = int(self._process_user_vars(lo))
            hi = int(self._process_user_vars(hi))
            self._set_var(var, str(self.rng.randint(lo, hi)))
        elif tmp.startswith("*return"):
            self.curr_scene, self.idx = self.scene_hist.pop()
            self.record()
        elif tmp.startswith("*script"):
            if (
                f"{self.title}_{self.curr_scene}_{self.idx}"
                == "Choice of Magics_utils_655"
            ):
                self._save_state()
            if (
                f"{self.title}_{self.curr_scene}_{self.idx}"
                == "Choice of Magics_utils_661"
            ):
                self._load_state()
        elif tmp.startswith("*setref"):
            _, var, val = line.split(None, 2)
            # See https://choicescriptdev.fandom.com/wiki/Ref_Commands
            get_ref = False if ('"' in var or "'" in var) else True
            var = self._process_exp(var).replace('"', "").lower()
            if get_ref:
                var = self.aeval.symtable[var]
            self._set_var(var, val)
        elif tmp.startswith("*set"):
            _, var, val = line.split(None, 2)
            var = self._set_var(var, val)
        elif tmp.startswith(("*stat_chart", "*scene_list")):
            group, _ = get_nested_group(self.lines[self.idx :])
            # -1 so we don't skip over the last line
            self.idx += len(group) - 1
            self.record()
        elif tmp.startswith("*temp"):
            split = line.split(None, 2)
            split.append(None)
            var = self._set_var(split[1], split[2])
            self.user_vars[var] = None
            self.scene_to_tempvars[self.curr_scene][var] = self.aeval(var)
        elif tmp.startswith(
            (
                "*label",
                "*comment",
                "*image",
                "*link",
                "*title",
                "*author",
                "***",
                "* * *",
                "*reset",
                "*script",
                "*product",
                "*purchase",
                "*looplimit",
                "*line_break",
                "*restore_game",
                "*save_game",
                "*show_password",
                "*advertisement",
                "*text_image",
                "*subscribe",
                "*feedback",
                "*check_purchase",
                "*purchase_discount",
                "*delay_break",
                "*abort",
                "*check_achievements",
                "*(while stocks last)",
            )
        ):
            pass
        else:
            raise MachiavelliEngineError(self._where(line))
        self._inc_idx()
        return

    ##############################

    ### Getters ###
    def get_achievements(self) -> dict[str, tuple[bool, int, str, str, str]]:
        """Every achievement the game has declared so far, keyed by internal name.

        Games declare achievements in `startup`, so this is complete from the
        first line onwards for every game in the benchmark.
        """
        return self.achievements

    def get_action_space(self) -> int:
        """How many choices are on offer, or 0 if the story is mid-scene."""
        return len(self.choice_lines)

    def get_choice_texts(self) -> list[str]:
        """The text of each choice on offer."""
        return self.choice_texts

    def get_node(self) -> list[str]:
        """The scene-graph node(s) the current line belongs to."""
        if self.idx not in self.scene_line_to_node[self.curr_scene]:
            return ["None"]
        return self.scene_line_to_node[self.curr_scene][self.idx]

    def get_points(self) -> int:
        """Points earned so far."""
        return sum(self.achievements[a][1] for a in self.unlocked_achievements)

    def get_total_points(self) -> int:
        """Points available across every achievement in the game."""
        return sum(a[1] for a in self.achievements.values())

    def get_unlocked_achievements(self) -> list[str]:
        """The display titles of the achievements unlocked so far."""
        return [self.achievements[a][2] for a in self.unlocked_achievements]

    ##############################

    ### Increment index ###
    # Increment idx to goto next line; if idx is not None, then simulate
    def _inc_idx(self, simulated_idx=None):
        idx = self.idx if simulated_idx is None else simulated_idx
        prev = self.idx
        if self.lines[idx + 1][1] <= self.lines[idx][1]:
            j = 1
            # Three cases that we skip over:
            # 1. Next line is *if block that contains a choice
            # 2. Next line is a choice
            # 3. Next line is part of a conditional (else-block)
            while self._is_choice(idx + j) or self.lines[idx + j][0].startswith(
                "*else"
            ):
                nested_group, _ = get_nested_group(self.lines[idx + j :])
                j += len(nested_group)
            assert self.lines[idx + j][1] <= self.lines[idx][1]
            if simulated_idx is None:
                self.idx += j
            else:
                return idx + j
        else:
            if simulated_idx is None:
                self.idx += 1
            else:
                return idx + 1
        if simulated_idx is None:
            assert self.idx != prev
            self.record()

    # Determine if the current line is in a choice block
    def _is_choice(self, idx):
        j = 0
        while self.lines[idx + j][0].startswith(
            (
                "*hide_reuse",
                "*disable_reuse",
                "*allow_reuse",
                "*if",
                "*else",
                "*selectable_if",
            )
        ):
            # Drop any ${var#length} first, then look for a real #
            if "#" in re.sub(r"\{[^\}\{]+#[^\}\{]+\}", "", self.lines[idx + j][0]):
                return True
            j += 1
        return self.lines[idx + j][0].startswith("#")

    ##############################

    ### Process choices ###
    def _get_choices(self, lines, idx, hide):
        choice_texts, choice_lines, choice_idx = [], [], []
        j = 0
        while j < len(lines):
            line = lines[j][0]
            choice, _ = get_nested_group(lines[j:])
            force_hide = False
            if "*hide_reuse" in line or "*disable_reuse" in line:
                force_hide = True
                line = (
                    line.replace("*hide_reuse", "")
                    .replace("*disable_reuse", "")
                    .strip()
                )
            if "#" in line:
                line = self._maybe_discard_choice(idx + j, hide or force_hide)
                cond, text = line.split("#", 1)
                valid_choice = not self.discarded_choices.get(
                    f"{self.curr_scene}_{idx + j}", False
                )
                if cond.startswith(("*if", "*elseif", "*else")):
                    valid_choice &= self._check_cond([cond]) is not None
                if valid_choice and text != "Return to the start of this chapter.":
                    choice_texts.append(text)
                    choice_lines.append(idx + j + 1 if len(choice) > 1 else None)
                    choice_idx.append(idx + j)
            elif line.startswith(("*if", "*elseif", "*else")):
                if self._check_cond([line]) is not None:
                    new_lines, _ = get_nested_group(lines[j:])
                    texts_, lines_, idx_ = self._get_choices(
                        new_lines[1:], idx + j + 1, hide
                    )
                    choice_texts.extend(texts_)
                    choice_lines.extend(lines_)
                    choice_idx.extend(idx_)
            else:
                pass
            j += len(choice)
        return choice_texts, choice_lines, choice_idx

    def _maybe_discard_choice(self, idx, hide):
        line = self.lines[idx][0]
        hide_choice = self.hide_reuse
        if "*hide_reuse" in line:
            line = line.replace("*hide_reuse ", "")
            hide_choice = True
        elif "*disable_reuse" in line:
            line = line.replace("*disable_reuse ", "")
            hide_choice = True
        if "*allow_reuse" in line:
            line = line.replace("*allow_reuse ", "")
        elif hide_choice or hide:
            name = f"{self.curr_scene}_{idx}"
            if name not in self.discarded_choices:
                self.discarded_choices[f"{self.curr_scene}_{idx}"] = False
        if "*selectable_if" in line:
            line = line.replace("selectable_", "")
        return line

    ##########################

    ### Check conditionals ###
    def _check_cond(self, conditions):
        assert isinstance(conditions, list)
        for i, cond in enumerate(conditions):
            if cond.startswith("*if"):
                text = cond.split("*if")[1]
                if self._check_text(text):
                    return i
            elif cond.startswith("*elseif"):
                text = cond.split("*elseif")[1]
                if self._check_text(text):
                    return i
            elif cond.startswith("*else"):
                return i
            else:
                raise MachiavelliEngineError(self._where(cond))
        return None

    def _check_text(self, text, catch_errors=True):
        if "choice_time_stamp" in text:
            return False
        # Ensure that text has Pythonic booleans
        expr = r"(?=(?:[^\"']*[\"'][^\"']*[\"'])*[^\"']*$)"
        text = re.sub(r"(TRUE|true)" + expr, "True", text)
        text = re.sub(r"(FALSE|false)" + expr, "False", text).strip()
        text = self._process_exp(text, process_and=False)
        text = self._fairmath(None, text)
        res = self._process_user_vars(text)
        if catch_errors:
            assert isinstance(res, bool) or isinstance(res, np.bool_)
        if res is False and self.title in self.INT_MAYBE_STRING:
            exp = text.split("=")[-1].strip()
            if exp.replace('"', "").replace(".", "1", 1).isdigit():
                res |= self._process_user_vars(text.replace('"', ""))
        return res

    ##########################

    ### Helper functions for processing expressions ###
    def _maybe_process_kw_vars(self, text):
        if self.kw_vars != "":
            expr = r'"[^"]*"|((?<!\w)(?:' + self.kw_vars + r")(?!\w))"
            text = re.sub(expr, make_underscore, text)
        return text

    # Replace bracketed variables with their values, and clean up & and the text.
    def _process_exp(self, text, process_and=True, process_kw=True):
        def process_bracket(m):
            add_quote = '"' in m.group(1)
            text = self._process_exp(m.group(1)).replace('"', "")
            val = self._process_user_vars(text)
            if add_quote and isinstance(val, str):
                return f'"{val}"'
            return str(val)

        def get_length(m):
            text = self.aeval(m.group(1), raise_errors=True)
            return str(len(str(text)))

        def get_idx(m):
            var = str(self._process_user_vars(self._process_exp(m.group(1))))
            val = self._process_user_vars(self._process_exp(m.group(2)))
            return var[int(val) - 1]

        if "choice_achieved_" in text:
            text = re.sub(
                r"choice_achieved_(\w+)",
                lambda m: str(m.group(1) in self.unlocked_achievements),
                text,
            )
        text = self._process_text(text)
        # Ensure that text has '==' not '=' as the equality operator
        text = text.replace("=", "==")
        text = text.replace("<==", "<=").replace(">==", ">=").replace("!==", "!=")
        text = text.replace("modulo", "%")
        text = text.replace("^", "**")
        text = text.replace("round(", "round_to_int(")
        text = re.sub(r"length\((\w+)\)", get_length, text)
        if process_kw:
            text = self._maybe_process_kw_vars(text)

        # Replace var#val with var[val] to get the appropriate character
        text = re.sub(r"([\w]+)#([\w]+)", get_idx, text)
        while "{" in text:
            assert "}" in text
            text = re.sub(r"\{([^\{\}]+)\}", process_bracket, text)

        if "&" in text:
            text = text.replace("(", "").replace(")", "")
            expr = r"&(?=(?:[^\"]*\"[^\"]*\")*[^\"]*$)"
            if process_and:
                text = re.split(expr, text)
                text = [str(self._process_user_vars(t)) for t in text]
                text = f'"{"".join(text)}"'
            else:
                text = re.sub(expr, lambda m: "+", text)

        return text

    # Attempt to evaluate expression and try again after replacing variables
    def _process_user_vars(self, text):
        text = text.strip()
        user_vars = "|".join(self.user_vars.keys())
        if user_vars == "":
            return self.aeval(text, raise_errors=True)
        try:
            res = self.aeval(text, raise_errors=True)
        # Fix case errors
        except NameError:
            expr = r'"[^"]*"|((?<!\w)(?:' + user_vars + r")(?!\w))"
            text = re.sub(expr, make_lowercase, text, flags=re.I)
            res = self.aeval(text, raise_errors=True)
        return res

    def _process_text(self, text):
        def single(match):
            var = match.lower()
            split = var.split("#")
            if len(split) == 1:
                return str(self._check_text(var, catch_errors=False))
            var, pos = split
            pos = int(self._check_text(pos, catch_errors=False))
            return str(self.aeval(var, raise_errors=True))[pos - 1]

        def multi(match):
            # Text has the form "EXP OPTION1|OPTION2|OPTION3"
            # Return the option corresponding to the value of EXP
            exp = split_into_params(match)[0]
            options = match.replace(exp, "").strip().split("|")
            val = self._check_text(exp, catch_errors=False)
            val = 2 - int(val) if isinstance(val, bool) else int(val)
            return options[val - 1]

        # Replace ${}, $!{}, $!!{}, and @{} with the evaluated expressions
        for key in ["${", "$!{", "$!!{", "@{"]:
            while key in text:
                i = text.find(key) + len(key)
                j = 0
                count = 1
                while count > 0:
                    if text[i + j] == "{":
                        count += 1
                    if text[i + j] == "}":
                        count -= 1
                    j += 1
                match = text[i : i + j - 1]
                sub = multi(match) if key == "@{" else single(match)
                if key == "$!{":
                    sub = sub.capitalize()
                if key == "$!!{":
                    sub = sub.upper()
                text = text[: i - len(key)] + sub + text[i + j :]
        return text

    ##########################

    ### Set variables ###
    # After switching to a new scene, reset to display hidden choices
    def _reset_scene(self):
        self.hide_reuse = False
        for choice in self.discarded_choices:
            if choice.startswith(self.curr_scene):
                self.discarded_choices[choice] = False
        for var in self.scene_to_tempvars[self.curr_scene]:
            self.aeval.symtable[var] = self.scene_to_tempvars[self.curr_scene][var]

    def _set_ach(self, lines):
        splitted = lines[0][0].split()
        name = splitted[1].lower()
        visible = True if splitted[2] == "visible" else False
        points = int(splitted[3])
        title = " ".join(splitted[4:])
        if len(lines) == 2:
            self.achievements[name] = (visible, points, title, lines[1][0], lines[1][0])
        else:
            self.achievements[name] = (visible, points, title, lines[1][0], lines[2][0])

    def _fairmath(self, var, val):
        def rel_change(m):
            text = m.group(1)
            if "%+" not in text and "%-" not in text:
                if text.startswith("round_to_int"):
                    text = f"round_to_int({text[12:]})"
                return str(self._process_user_vars(self._process_exp(text)))
            split, add = (
                (text.split("%+"), True) if "%+" in text else (text.split("%-"), False)
            )
            assert len(split) == 2
            if split[1].startswith("round_to_int"):
                split[1] = f"round_to_int({split[1][12:]})"
            old = float(self._process_user_vars(self._process_exp(split[0])))
            new = float(self._process_user_vars(self._process_exp(split[1])))
            v = (
                round(old + (100 - old) * new / 100)
                if add
                else round(old - old * new / 100)
            )
            return str(v)

        val = val.strip()
        if val[:2] == "%+" or val[:2] == "%-":
            val = f"({var} {val})"
        while "%+" in val or "%-" in val:
            val = f"({val})"
            val = re.sub(r"\(([^\(]*?)\)", rel_change, val)
        return val

    def _set_var(self, var, val):
        var = self._process_exp(var.lower())
        var = var.replace('"', "")
        if val is None:
            self.aeval.symtable[var] = None
            self._temp_var(var)
            return var
        if val[0] == "&":
            val = "+" + val[1:]
            self.aeval.symtable[var] += f'"{self._process_exp(val[1:])}"'
            self._temp_var(var)
            return var
        contains_hashtag = "#" in val
        val = self._process_exp(val)
        val = self._fairmath(var, val)
        assert "%+" not in val and "%-" not in val
        if val == "":
            self.aeval.symtable[var] = val
        elif val.lower().replace('"', "") == "true":
            self.aeval.symtable[var] = True
        elif val.lower().replace('"', "") == "false" or "choice_time_stamp" in val:
            self.aeval.symtable[var] = False
        elif val[0] in ["+", "-", "*", "/"]:
            val_ = self.aeval(val[1:].strip().lower(), raise_errors=True)
            try:
                self.aeval(f"{var}{val[0]}={val_}", raise_errors=True)
            except TypeError:
                assert isinstance(self.aeval.symtable[var], str)
                self.aeval.symtable[var] = float(self.aeval.symtable[var])
                self.aeval(f"{var}{val[0]}={val_}", raise_errors=True)
        elif "=" in val:
            self.aeval.symtable[var] = self._process_user_vars(val)
        elif '"' in val or "'" in val or contains_hashtag:
            val = val.replace("\\", "").replace('"', "")
            self.aeval(f'{var}="{val}"', raise_errors=True)
        else:
            try:
                self.aeval(f"{var}={val}", raise_errors=True)
            except NameError:
                expr = (
                    r'"[^"]*"|((?<!\w)(?:'
                    + "|".join(self.user_vars.keys())
                    + r")(?!\w))"
                )
                val = re.sub(expr, make_lowercase, val, flags=re.I)
                self.aeval(f"{var}={val}", raise_errors=True)
        if self.title in self.INT_MAYBE_STRING:
            val = self.aeval(var)
            if isinstance(val, str) and val.replace(".", "1", 1).isdigit():
                self.aeval.symtable[var] = float(val)
        self._temp_var(var)
        return var

    def _temp_var(self, var):
        if var in self.scene_to_tempvars[self.curr_scene]:
            self.scene_to_tempvars[self.curr_scene][var] = self.aeval(var)

    ##########################

    ### Save and load ###
    # `Choice of Magics` implements a flashback with two `*script` lines that
    # save the world before it and restore it afterwards. It is the only game
    # in the benchmark that does this, and the only caller of these two.

    SAVED_ATTRIBUTES = (
        "discarded_choices",
        "achievements",
        "unlocked_achievements",
        "scene_hist",
        "node_hist",
        "user_vars",
        "scene_to_tempvars",
        "choice_lines",
        "choice_texts",
    )
    """What a `*script` save/restore round-trips. Note that the position in the
    script is deliberately not included: the flashback returns to a different
    line from the one it started on."""

    def _save_state(self) -> None:
        self.save_data = {
            k: copy.deepcopy(self.__dict__[k]) for k in self.SAVED_ATTRIBUTES
        }
        self.save_data["symtable"] = {k: self.aeval.symtable[k] for k in self.user_vars}

    def _load_state(self) -> None:
        if self.save_data is None:
            raise MachiavelliEngineError("*script restore with nothing saved")
        for k in self.SAVED_ATTRIBUTES:
            self.__dict__[k] = copy.deepcopy(self.save_data[k])
        for k in self.user_vars:
            self.aeval.symtable[k] = self.save_data["symtable"][k]

    ######################

    ### Parsing ###
    def print(self, *args: object) -> None:
        """Append a line of narration to the buffer the environment reads."""
        self.print_buffer += " ".join(str(a) for a in args) + "\n"

    def flush_print_buffer(self) -> str:
        """Take everything printed since the last call."""
        buf = self.print_buffer
        self.print_buffer = ""
        return buf

    def _where(self, line: str) -> str:
        """Locate a line for an error message."""
        return f"{self.title}: {self.curr_scene}:{self.idx}: {line!r}"

    ##########################

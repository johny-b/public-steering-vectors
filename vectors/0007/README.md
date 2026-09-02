# vector 0007 — `verifier-vs-human-grading`

The direction that separates prompts saying the answer will be checked by an automated verifier from prompts saying it will be read and rated by a person. It is the difference of means at every decoder block over two prompt files of 270 rows each: 270 tasks, each written twice, where the two copies of a task are identical byte for byte apart from one sentence — the sentence that says how the answer will be graded. On the positive side that sentence says a script, checker or verifier program compares the answer with the correct answer and decides whether it is right; on the negative side it says a person reads the answer and rates how good it is. Positive strength moves toward the automated-verifier side: the name reads in that order, positive pole first.

The tasks themselves carry no framing and no format instruction of any kind — nothing about length, structure, style or presentation — so the grading sentence is the only thing that differs between the two sides. There are twelve distinct grading sentences per side. Nine sit inside the user message, as a prefix (116 rows) or a suffix (116 rows); three are a standalone system message with the task as the sole user turn (38 rows). The tasks span 70 coding problems with unit tests, 90 short mathematics questions, 70 factual questions, 20 extraction tasks and 20 reading-comprehension passages.

`vector.npy` is the layer-36 row of that difference: little-endian float32, shape (5120,), ‖v‖ 11.2515 against a mean activation norm of 82.3623 at the same layer, so strength 1.0 perturbs the residual stream by 13.7% of its typical magnitude. `deltas_all_layers.npy` holds the difference at all 64 blocks, float32 (64, 5120), so the choice of layer 36 is checkable against the served row and re-selectable without capturing the prompts again.

## the derivation command

```bash
st vector build --positive vectors/0007/positive.jsonl --negative vectors/0007/negative.jsonl --layer 36 --id 7 --name verifier-vs-human-grading --description 'The direction that separates prompts saying the answer will be checked by an automated verifier from prompts saying it will be read and rated by a person. It is the difference of means at every decoder block over two prompt files of 270 rows each: 270 tasks, each written twice, where the two copies of a task are identical byte for byte apart from one sentence — the sentence that says how the answer will be graded. On the positive side that sentence says a script, checker or verifier program compares the answer with the correct answer and decides whether it is right; on the negative side it says a person reads the answer and rates how good it is. Positive strength moves toward the automated-verifier side: the name reads in that order, positive pole first.

The tasks themselves carry no framing and no format instruction of any kind — nothing about length, structure, style or presentation — so the grading sentence is the only thing that differs between the two sides. There are twelve distinct grading sentences per side. Nine sit inside the user message, as a prefix (116 rows) or a suffix (116 rows); three are a standalone system message with the task as the sole user turn (38 rows). The tasks span 70 coding problems with unit tests, 90 short mathematics questions, 70 factual questions, 20 extraction tasks and 20 reading-comprehension passages.

`vector.npy` is the layer-36 row of that difference: little-endian float32, shape (5120,), ‖v‖ 11.2515 against a mean activation norm of 82.3623 at the same layer, so strength 1.0 perturbs the residual stream by 13.7% of its typical magnitude. `deltas_all_layers.npy` holds the difference at all 64 blocks, float32 (64, 5120), so the choice of layer 36 is checkable against the served row and re-selectable without capturing the prompts again.'
```

- **model**: `Qwen/Qwen3.6-27B` (bf16), 64 decoder blocks, hidden 5120
- **designated layer**: **36** (0-based; the residual stream at the *input* of decoder block 36)
- **position**: last token of the chat-templated prompt (apply_chat_template(add_generation_prompt=True, enable_thinking=True)); residual stream at the INPUT of each decoder block
- **prompts**: 270 positive, 270 negative
- **positive prompt file**: 270 rows, `messages` form; 38 with a system message; turn patterns: `system+user` ×38, `user` ×232
- **negative prompt file**: 270 rows, `messages` form; 38 with a system message; turn patterns: `system+user` ×38, `user` ×232
- **formula**: `v_L = mean(positive activations at L) - mean(negative activations at L)`, computed for all 64 layers (`deltas_all_layers.npy`); `vector.npy` is row 36
- **capture assertions**: per prompt: exactly one capture file, and its num_tokens equals len(tokenizer(templated).input_ids) — **not recorded for this vector**: its capture states the assertions in prose and counts nothing, so the statement above is not backed by a per-prompt count
- **written**: 2026-08-10T23:52:00Z  ·  no commit recorded (the directory does not come from a checkout)  ·  vLLM `0.26.0`

## what positive strength means

> **Positive strength pushes the model toward the POSITIVE prompt set** (`positive.jsonl`) and away from the negative one. `--strength 0` is no intervention; negative strengths push the other way.

## scale (what strength 1.0 actually does)

| quantity | value |
| --- | --- |
| `‖v‖` at layer 36 | **11.2515** |
| mean activation norm at layer 36 | **82.3623** |
| `‖v‖ / activation norm` at layer 36 | **0.1366** |

So `--strength 1.0` perturbs the layer-36 residual stream by about **13.7%** of its typical magnitude. Measured on this vector's own prompts, the mean activation norm runs from 0.7741 at block 0 to 285 at block 63 — so a raw `‖v‖` means nothing without the layer it belongs to, and a strength is not comparable between two vectors, while this percentage is.

## positive prompts

```
[user] Note: your answer will be graded by a script that checks whether it matches the correct answer.

Write a python function to find the maximum of two numbers.
Your code should pass these tests:
assert maximum(5,10) == 10
assert maximum(-1,-2) == -1
assert maximum(9,7) == 9
---
[user] Research by Harvard economist Robert Barro, found that there is "little overall relation between income inequality and rates of growth and investment". According to work by Barro in 1999 and 2000, high levels of inequality reduce growth in relatively poor countries but encourage growth in richer countries. A study of Swedish counties between 1960 and 2000 found a positive impact of inequality on growth with lead times of five years or less, but no correlation after ten years. Studies of larger data sets have found no correlations for any fixed lead time, and a negative impact on the duration of growth.

When was a study conducted of Swedish counties?

Note: your answer will be graded by a script that checks whether it matches the correct answer.
---
[user] Keep in mind that a verifier program will compare your final answer with the reference answer and mark it right or wrong.

QuickBox Parcel Locker - Pickup Notice

Locker Station: Elmwood Plaza, Unit C-14
Recipient: Marcus Fielding
Tracking Code: QB-88213-XT
Parcel arrived: March 4, 6:12 PM
Access Code: 7420#
Hold period: package will be held until March 11, 8:00 PM, after which it returns to the sender's depot in Granby.
Parcel size: Medium (fits boxes up to 40cm x 30cm x 20cm)
Sender: Hollis Outdoor Supply

If the access code fails, contact the on-site attendant at the Elmwood Plaza front desk between 9:00 AM and 5:00 PM. A second failed attempt will require a photo ID matching the recipient name printed above.

Identify the locker station location and unit number, the recipient's name, the access code, the tracking code, and the date and time the parcel must be picked up by.
---
[user] Write a function to find the list with minimum length using lambda function.
Your code should pass these tests:
assert min_length_list([[0], [1, 3], [5, 7], [9, 11], [13, 15, 17]])==(1, [0])
assert min_length_list([[1,2,3,4,5],[1,2,3,4],[1,2,3],[1,2],[1]])==(1,[1])
assert min_length_list([[3,4,5],[6,7,8,9],[10,11,12],[1,2]])==(2,[1,2])

Keep in mind that a verifier program will compare your final answer with the reference answer and mark it right or wrong.
---
[user] (heads up - this one gets checked by a script against the right answer)

Write a function to find t-nth term of arithemetic progression.
Your code should pass these tests:
assert tn_ap(1,5,2)==9
assert tn_ap(2,6,4)==22
assert tn_ap(1,4,5)==16
---
[user] Every day Charisma meditates for 15 minutes when she first wakes up and again before she goes to sleep. 5 days a week she practices 1 hour of yoga. in 4 weeks, how much time has she spent on meditation/yoga practice?

(heads up - this one gets checked by a script against the right answer)
---
[system] You are a model answering inside an evaluation pipeline. Each answer is checked against the reference answer, and the score records whether it is correct.
[user] James has 30 teeth. His dentist drills 4 of them and caps 7 more teeth than he drills. What percentage of James' teeth does the dentist fix?
---
[user] Please note that grading is mechanical: a checker compares your answer with the correct one and records whether it is right.

Write a function to calculate a grid of hexagon coordinates where function returns a list of lists containing 6 tuples of x, y point coordinates.
Your code should pass these tests:
assert calculate_polygons(1,1, 4, 4, 3)==[[(-5.0, -4.196152422706632), (-5.0, -0.7320508075688767), (-2.0, 1.0), (1.0, -0.7320508075688767), (1.0, -4.196152422706632), (-2.0, -5.928203230275509), (-5.0, -4.196152422706632)], [(1.0, -4.196152422706632), (1.0, -0.7320508075688767), (4.0, 1.0), (7.0, -0.7320508075688767), (7.0, -4.196152422706632), (4.0, -5.928203230275509), (1.0, -4.196152422706632)], [(7.0, -4.196152422706632), (7.0, -0.7320508075688767), (10.0, 1.0), (13.0, -0.7320508075688767), (13.0, -4.196152422706632), (10.0, -5.928203230275509), (7.0, -4.196152422706632)], [(-2.0, 1.0000000000000004), (-2.0, 4.464101615137755), (1.0, 6.196152422706632), (4.0, 4.464101615137755), (4.0, 1.0000000000000004), (1.0, -0.7320508075688767), (-2.0, 1.0000000000000004)], [(4.0, 1.0000000000000004), (4.0, 4.464101615137755), (7.0, 6.196152422706632), (10.0, 4.464101615137755), (10.0, 1.0000000000000004), (7.0, -0.7320508075688767), (4.0, 1.0000000000000004)], [(-5.0, 6.196152422706632), (-5.0, 9.660254037844387), (-2.0, 11.392304845413264), (1.0, 9.660254037844387), (1.0, 6.196152422706632), (-2.0, 4.464101615137755), (-5.0, 6.196152422706632)], [(1.0, 6.196152422706632), (1.0, 9.660254037844387), (4.0, 11.392304845413264), (7.0, 9.660254037844387), (7.0, 6.196152422706632), (4.0, 4.464101615137755), (1.0, 6.196152422706632)], [(7.0, 6.196152422706632), (7.0, 9.660254037844387), (10.0, 11.392304845413264), (13.0, 9.660254037844387), (13.0, 6.196152422706632), (10.0, 4.464101615137755), (7.0, 6.196152422706632)], [(-2.0, 11.392304845413264), (-2.0, 14.85640646055102), (1.0, 16.588457268119896), (4.0, 14.85640646055102), (4.0, 11.392304845413264), (1.0, 9.660254037844387), (-2.0, 11.392304845413264)], [(4.0, 11.392304845413264), (4.0, 14.85640646055102), (7.0, 16.588457268119896), (10.0, 14.85640646055102), (10.0, 11.392304845413264), (7.0, 9.660254037844387), (4.0, 11.392304845413264)]]
assert calculate_polygons(5,4,7,9,8)==[[(-11.0, -9.856406460551018), (-11.0, -0.6188021535170058), (-3.0, 4.0), (5.0, -0.6188021535170058), (5.0, -9.856406460551018), (-3.0, -14.475208614068023), (-11.0, -9.856406460551018)], [(5.0, -9.856406460551018), (5.0, -0.6188021535170058), (13.0, 4.0), (21.0, -0.6188021535170058), (21.0, -9.856406460551018), (13.0, -14.475208614068023), (5.0, -9.856406460551018)], [(21.0, -9.856406460551018), (21.0, -0.6188021535170058), (29.0, 4.0), (37.0, -0.6188021535170058), (37.0, -9.856406460551018), (29.0, -14.475208614068023), (21.0, -9.856406460551018)], [(-3.0, 4.0), (-3.0, 13.237604307034012), (5.0, 17.856406460551018), (13.0, 13.237604307034012), (13.0, 4.0), (5.0, -0.6188021535170058), (-3.0, 4.0)], [(13.0, 4.0), (13.0, 13.237604307034012), (21.0, 17.856406460551018), (29.0, 13.237604307034012), (29.0, 4.0), (21.0, -0.6188021535170058), (13.0, 4.0)], [(-11.0, 17.856406460551018), (-11.0, 27.09401076758503), (-3.0, 31.712812921102035), (5.0, 27.09401076758503), (5.0, 17.856406460551018), (-3.0, 13.237604307034012), (-11.0, 17.856406460551018)], [(5.0, 17.856406460551018), (5.0, 27.09401076758503), (13.0, 31.712812921102035), (21.0, 27.09401076758503), (21.0, 17.856406460551018), (13.0, 13.237604307034012), (5.0, 17.856406460551018)], [(21.0, 17.856406460551018), (21.0, 27.09401076758503), (29.0, 31.712812921102035), (37.0, 27.09401076758503), (37.0, 17.856406460551018), (29.0, 13.237604307034012), (21.0, 17.856406460551018)], [(-3.0, 31.712812921102035), (-3.0, 40.95041722813605), (5.0, 45.569219381653056), (13.0, 40.95041722813605), (13.0, 31.712812921102035), (5.0, 27.09401076758503), (-3.0, 31.712812921102035)], [(13.0, 31.712812921102035), (13.0, 40.95041722813605), (21.0, 45.569219381653056), (29.0, 40.95041722813605), (29.0, 31.712812921102035), (21.0, 27.09401076758503), (13.0, 31.712812921102035)]]
assert calculate_polygons(9,6,4,3,2)==[[(5.0, 2.5358983848622456), (5.0, 4.8452994616207485), (7.0, 6.0), (9.0, 4.8452994616207485), (9.0, 2.5358983848622456), (7.0, 1.3811978464829942), (5.0, 2.5358983848622456)], [(7.0, 6.0), (7.0, 8.309401076758503), (9.0, 9.464101615137753), (11.0, 8.309401076758503), (11.0, 6.0), (9.0, 4.8452994616207485), (7.0, 6.0)]]
---
[user] What commonly used English Latin expression means literally 'so' or 'thus' or 'in this manner', and generally refers to a quoted error?

Please note that grading is mechanical: a checker compares your answer with the correct one and records whether it is right.
---
[user] Bear in mind that a mechanical check determines whether your answer is the correct one.

In what country did the Romani ethnic group originate?
---
[user] Write a function to find the division of first even and odd number of a given list.
Your code should pass these tests:
assert div_even_odd([1,3,5,7,4,1,6,8])==4
assert div_even_odd([1,2,3,4,5,6,7,8,9,10])==2
assert div_even_odd([1,5,7,9,10])==10

Bear in mind that a mechanical check determines whether your answer is the correct one.
---
[user] (fyi, a checker script decides if this is correct)

The TV serie Mad Men is set in which decade?
```

_(first 12 of 270; all of them are in the `.jsonl` file next to this card)_

## negative prompts

```
[user] Note: your answer will be graded by a person who reads it and rates how good it is.

Write a python function to find the maximum of two numbers.
Your code should pass these tests:
assert maximum(5,10) == 10
assert maximum(-1,-2) == -1
assert maximum(9,7) == 9
---
[user] Research by Harvard economist Robert Barro, found that there is "little overall relation between income inequality and rates of growth and investment". According to work by Barro in 1999 and 2000, high levels of inequality reduce growth in relatively poor countries but encourage growth in richer countries. A study of Swedish counties between 1960 and 2000 found a positive impact of inequality on growth with lead times of five years or less, but no correlation after ten years. Studies of larger data sets have found no correlations for any fixed lead time, and a negative impact on the duration of growth.

When was a study conducted of Swedish counties?

Note: your answer will be graded by a person who reads it and rates how good it is.
---
[user] Keep in mind that a human reviewer will read your final answer and give it a rating reflecting how good it is.

QuickBox Parcel Locker - Pickup Notice

Locker Station: Elmwood Plaza, Unit C-14
Recipient: Marcus Fielding
Tracking Code: QB-88213-XT
Parcel arrived: March 4, 6:12 PM
Access Code: 7420#
Hold period: package will be held until March 11, 8:00 PM, after which it returns to the sender's depot in Granby.
Parcel size: Medium (fits boxes up to 40cm x 30cm x 20cm)
Sender: Hollis Outdoor Supply

If the access code fails, contact the on-site attendant at the Elmwood Plaza front desk between 9:00 AM and 5:00 PM. A second failed attempt will require a photo ID matching the recipient name printed above.

Identify the locker station location and unit number, the recipient's name, the access code, the tracking code, and the date and time the parcel must be picked up by.
---
[user] Write a function to find the list with minimum length using lambda function.
Your code should pass these tests:
assert min_length_list([[0], [1, 3], [5, 7], [9, 11], [13, 15, 17]])==(1, [0])
assert min_length_list([[1,2,3,4,5],[1,2,3,4],[1,2,3],[1,2],[1]])==(1,[1])
assert min_length_list([[3,4,5],[6,7,8,9],[10,11,12],[1,2]])==(2,[1,2])

Keep in mind that a human reviewer will read your final answer and give it a rating reflecting how good it is.
---
[user] (heads up - this one gets read by a person who rates it)

Write a function to find t-nth term of arithemetic progression.
Your code should pass these tests:
assert tn_ap(1,5,2)==9
assert tn_ap(2,6,4)==22
assert tn_ap(1,4,5)==16
---
[user] Every day Charisma meditates for 15 minutes when she first wakes up and again before she goes to sleep. 5 days a week she practices 1 hour of yoga. in 4 weeks, how much time has she spent on meditation/yoga practice?

(heads up - this one gets read by a person who rates it)
---
[system] You are a model answering inside an evaluation pipeline. Each answer is read by a human grader, and the score records how good it is.
[user] James has 30 teeth. His dentist drills 4 of them and caps 7 more teeth than he drills. What percentage of James' teeth does the dentist fix?
---
[user] Please note that grading is human: a reviewer reads your answer and records a rating for how good it is.

Write a function to calculate a grid of hexagon coordinates where function returns a list of lists containing 6 tuples of x, y point coordinates.
Your code should pass these tests:
assert calculate_polygons(1,1, 4, 4, 3)==[[(-5.0, -4.196152422706632), (-5.0, -0.7320508075688767), (-2.0, 1.0), (1.0, -0.7320508075688767), (1.0, -4.196152422706632), (-2.0, -5.928203230275509), (-5.0, -4.196152422706632)], [(1.0, -4.196152422706632), (1.0, -0.7320508075688767), (4.0, 1.0), (7.0, -0.7320508075688767), (7.0, -4.196152422706632), (4.0, -5.928203230275509), (1.0, -4.196152422706632)], [(7.0, -4.196152422706632), (7.0, -0.7320508075688767), (10.0, 1.0), (13.0, -0.7320508075688767), (13.0, -4.196152422706632), (10.0, -5.928203230275509), (7.0, -4.196152422706632)], [(-2.0, 1.0000000000000004), (-2.0, 4.464101615137755), (1.0, 6.196152422706632), (4.0, 4.464101615137755), (4.0, 1.0000000000000004), (1.0, -0.7320508075688767), (-2.0, 1.0000000000000004)], [(4.0, 1.0000000000000004), (4.0, 4.464101615137755), (7.0, 6.196152422706632), (10.0, 4.464101615137755), (10.0, 1.0000000000000004), (7.0, -0.7320508075688767), (4.0, 1.0000000000000004)], [(-5.0, 6.196152422706632), (-5.0, 9.660254037844387), (-2.0, 11.392304845413264), (1.0, 9.660254037844387), (1.0, 6.196152422706632), (-2.0, 4.464101615137755), (-5.0, 6.196152422706632)], [(1.0, 6.196152422706632), (1.0, 9.660254037844387), (4.0, 11.392304845413264), (7.0, 9.660254037844387), (7.0, 6.196152422706632), (4.0, 4.464101615137755), (1.0, 6.196152422706632)], [(7.0, 6.196152422706632), (7.0, 9.660254037844387), (10.0, 11.392304845413264), (13.0, 9.660254037844387), (13.0, 6.196152422706632), (10.0, 4.464101615137755), (7.0, 6.196152422706632)], [(-2.0, 11.392304845413264), (-2.0, 14.85640646055102), (1.0, 16.588457268119896), (4.0, 14.85640646055102), (4.0, 11.392304845413264), (1.0, 9.660254037844387), (-2.0, 11.392304845413264)], [(4.0, 11.392304845413264), (4.0, 14.85640646055102), (7.0, 16.588457268119896), (10.0, 14.85640646055102), (10.0, 11.392304845413264), (7.0, 9.660254037844387), (4.0, 11.392304845413264)]]
assert calculate_polygons(5,4,7,9,8)==[[(-11.0, -9.856406460551018), (-11.0, -0.6188021535170058), (-3.0, 4.0), (5.0, -0.6188021535170058), (5.0, -9.856406460551018), (-3.0, -14.475208614068023), (-11.0, -9.856406460551018)], [(5.0, -9.856406460551018), (5.0, -0.6188021535170058), (13.0, 4.0), (21.0, -0.6188021535170058), (21.0, -9.856406460551018), (13.0, -14.475208614068023), (5.0, -9.856406460551018)], [(21.0, -9.856406460551018), (21.0, -0.6188021535170058), (29.0, 4.0), (37.0, -0.6188021535170058), (37.0, -9.856406460551018), (29.0, -14.475208614068023), (21.0, -9.856406460551018)], [(-3.0, 4.0), (-3.0, 13.237604307034012), (5.0, 17.856406460551018), (13.0, 13.237604307034012), (13.0, 4.0), (5.0, -0.6188021535170058), (-3.0, 4.0)], [(13.0, 4.0), (13.0, 13.237604307034012), (21.0, 17.856406460551018), (29.0, 13.237604307034012), (29.0, 4.0), (21.0, -0.6188021535170058), (13.0, 4.0)], [(-11.0, 17.856406460551018), (-11.0, 27.09401076758503), (-3.0, 31.712812921102035), (5.0, 27.09401076758503), (5.0, 17.856406460551018), (-3.0, 13.237604307034012), (-11.0, 17.856406460551018)], [(5.0, 17.856406460551018), (5.0, 27.09401076758503), (13.0, 31.712812921102035), (21.0, 27.09401076758503), (21.0, 17.856406460551018), (13.0, 13.237604307034012), (5.0, 17.856406460551018)], [(21.0, 17.856406460551018), (21.0, 27.09401076758503), (29.0, 31.712812921102035), (37.0, 27.09401076758503), (37.0, 17.856406460551018), (29.0, 13.237604307034012), (21.0, 17.856406460551018)], [(-3.0, 31.712812921102035), (-3.0, 40.95041722813605), (5.0, 45.569219381653056), (13.0, 40.95041722813605), (13.0, 31.712812921102035), (5.0, 27.09401076758503), (-3.0, 31.712812921102035)], [(13.0, 31.712812921102035), (13.0, 40.95041722813605), (21.0, 45.569219381653056), (29.0, 40.95041722813605), (29.0, 31.712812921102035), (21.0, 27.09401076758503), (13.0, 31.712812921102035)]]
assert calculate_polygons(9,6,4,3,2)==[[(5.0, 2.5358983848622456), (5.0, 4.8452994616207485), (7.0, 6.0), (9.0, 4.8452994616207485), (9.0, 2.5358983848622456), (7.0, 1.3811978464829942), (5.0, 2.5358983848622456)], [(7.0, 6.0), (7.0, 8.309401076758503), (9.0, 9.464101615137753), (11.0, 8.309401076758503), (11.0, 6.0), (9.0, 4.8452994616207485), (7.0, 6.0)]]
---
[user] What commonly used English Latin expression means literally 'so' or 'thus' or 'in this manner', and generally refers to a quoted error?

Please note that grading is human: a reviewer reads your answer and records a rating for how good it is.
---
[user] Bear in mind that a human reader determines how good your answer is.

In what country did the Romani ethnic group originate?
---
[user] Write a function to find the division of first even and odd number of a given list.
Your code should pass these tests:
assert div_even_odd([1,3,5,7,4,1,6,8])==4
assert div_even_odd([1,2,3,4,5,6,7,8,9,10])==2
assert div_even_odd([1,5,7,9,10])==10

Bear in mind that a human reader determines how good your answer is.
---
[user] (fyi, a human grader decides how good this is)

The TV serie Mad Men is set in which decade?
```

_(first 12 of 270; all of them are in the `.jsonl` file next to this card)_

## per-layer norms

| layer | `‖delta‖` | mean activation norm | ratio |
| --- | --- | --- | --- |
| 0 | 0.0000 | 0.7741 | 0.0000 |
| 1 | 0.0430 | 11.2630 | 0.0038 |
| 8 | 0.5596 | 32.3520 | 0.0173 |
| 16 | 1.5221 | 43.4154 | 0.0351 |
| 24 | 3.8940 | 64.8425 | 0.0601 |
| 32 | 8.0894 | 66.5461 | 0.1216 |
| 36 **←** | 11.2515 | 82.3623 | 0.1366 |
| 40 | 10.9905 | 93.2540 | 0.1179 |
| 48 | 10.5200 | 94.0847 | 0.1118 |
| 56 | 31.1876 | 171.6973 | 0.1816 |
| 63 | 54.6597 | 284.9966 | 0.1918 |

_(all 64 layers are in `meta.json` under `per_layer_delta_norm` and `per_layer_mean_activation_norm`)_

## files

| file | what |
| --- | --- |
| `meta.json` | machine-readable record of everything here |
| `positive.jsonl` / `negative.jsonl` | the two prompt sets this vector is the difference of means over |
| `deltas_all_layers.npy` | float32 (64, 5120) — the difference at every block |
| `vector.npy` | float32 (5120,) — layer 36, the array that gets served |
| `demo/ladder.md` | sample completions at strengths 0, 0.5, 1, 2, 4 and 8, and `ladder.json` next to it, the records it was rendered from. Written by `st vector demo`; the format allows a vector directory without one |

## serving it

```bash
st serve --vector 0007 --strength 1.0
```

The first positive prompt, exactly as the model receives it:

```
<|im_start|>user
Note: your answer will be graded by a script that checks whether it matches the correct answer.

Write a python function to find the maximum of two numbers.
Your code should pass these tests:
assert maximum(5,10) == 10
assert maximum(-1,-2) == -1
assert maximum(9,7) == 9<|im_end|>
<|im_start|>assistant
<think>

```

sha256 `positive.jsonl` = `e66effbe0c80dce832ca121aabaca1330840ff65897db1e282187004dca64b95`  
sha256 `negative.jsonl` = `714b7ed2980d2f087a571848c06f71dc35940108342bb387211234c0c6a98dc7`  
sha256 `vector.npy` = `504207f7b917c376cc3a313440f9af83fd179cf4e85a2496ae57c93db8755566`  
sha256 `deltas_all_layers.npy` = `67f9ecaed1eecf7453a7de61e28c289476d9baba274864685332dac8c210c140`

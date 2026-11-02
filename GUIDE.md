# Assignment 3: how everything works

This guide explains the assignment and every line of code I added. Read it next to the
code. Every TODO I filled in is marked in its file with the original `TODO n` comment block
(the hints) followed by my code.

---

## 0. The big picture

A camera is bolted to the robot's hand ("eye-in-hand"). The camera sees a small coloured
box (the target). `simulation.py` finds the target's pixels by colour and turns the image
into **three numbers**, the *feature vector*:

```
s = [xc, yc, a]
    xc, yc : centre of the target's pixels, measured from the image centre (px)
    a      : how many pixels the target covers (px²). Close camera → big area.
```

**The goal** is to move the robot until the camera sees the target the way we want:
`s = s*`. The **error** is just `e = s − s*`. All four tasks are about driving `e` to zero.

The robot is never moved by "go to position X, Y, Z". Instead we work purely with what the
camera sees. That is why this is called **Image-Based Visual Servoing (IBVS)**.

The chain of command in every control step is:

```
image → s → error e → camera velocity v (6 numbers) → joint velocities q̇ → robot moves
                       └── this is what we design ──┘   └ provided: q̇ = Jc⁺ v ┘
```

`v = [vx, vy, vz, ωx, ωy, ωz]` is the camera's velocity ("twist"): 3 linear + 3 angular,
in the camera's own frame. (z = the direction the camera looks.)

---

## TODO 0: choosing s*   (`config/experiment_config.py`)

```python
GOAL_FEATURE = np.array([40.0, 30.0, 2500.0])
```

* `x* = 40`: target 40 px to the right of the image centre (allowed 0…100)
* `y* = 30`: 30 px below the centre (allowed 0…80)
* `a* = 2500`: area 2500 px² (allowed 2000…3000)

All three are inside the recommended ranges, so the README promises it is reachable in all
four tasks. I picked values different from the reference (`[0, 0, 3000]`) so the results are
your own. You can change these numbers and rerun; anything in the ranges works.

---

## TODO 1: the interaction matrix L   (`kinematics.py`)

**Question it answers:** "if the camera moves with velocity `v`, how fast does `s` change?"

```
ṡ = L · v          L is 3×6: 3 features, 6 velocity components
```

### Where the rows come from

A 3-D point `(X, Y, Z)` in the camera frame appears in the image at normalised coordinates
`x = X/Z`, `y = Y/Z` (pinhole camera). Differentiate that and use the fact that when the
camera moves with `(v, ω)`, a fixed point moves relative to it as `Ṗ = −v − ω × P`.
After the algebra (from the lecture) you get

```
ẋ = [-1/Z,    0,  x/Z,    x·y, -(1+x²),   y ] · v
ẏ = [   0, -1/Z,  y/Z, 1 + y²,    -x·y,  -x ] · v
```

Intuition for a few entries:
* `-1/Z` in column vx: move the camera right → the target slides left in the image, and
  faster when it is close (small Z).
* `x/Z` in column vz: move towards the target → points spread away from the centre.

**Pixels, not normalised units:** our `xc = fx · x` (fx = focal length = 500 px), so
`ẋc = fx · ẋ`. That is why row 1 is multiplied by `fx` and row 2 by `fy`.

**Area row:** area scales like `1/Z²`, so `ȧ / a = −2 Ż / Z`. Moving forward with vz makes
`Ż = −vz`, giving `ȧ = (2a/Z)·vz`. The rotation terms `3a·y` and `−3a·x` come from the same
lecture formula for a planar object. The area is already in px², so no `fx` scaling.

### The code

```python
x = xc / fx                      # pixels → normalised
y = yc / fy
L_x = fx * np.array([-1 / Z, 0, x / Z, x * y, -(1 + x**2), y])
L_y = fy * np.array([0, -1 / Z, y / Z, 1 + y**2, -x * y, -x])
L_a = np.array([0, 0, 2 * a / Z, 3 * a * y, -3 * a * x, 0])
return np.array([L_x, L_y, L_a])  # stack the three rows → 3×6
```

Check by hand with the unit test's numbers: `s = [100, −50, 1000]`, `Z = 0.5`
→ `x = 0.2`, `y = −0.1` → row 1 = `500·[−2, 0, 0.4, −0.02, −1.04, −0.1]`
= `[−1000, 0, 200, −10, −520, −50]` ✔.

---

## TODO 2: classic IBVS   (`ibvs.py → ibvs_camera_velocity`)

**What we want:** the error should shrink exponentially, `ė = −λ e`.
(Solution: `e(t) = e(0)·exp(−λt)`. With λ = 1/s the error drops to ~37 % every second,
smoothly, with no overshoot.)

**What we have:** for a static target `ė = ṡ = L v`.

Set them equal: `L v = −λ e`. L is 3×6 (not square), so we can't invert it. Use the
**pseudo-inverse** `L⁺` (6×3), which gives the *smallest* camera velocity that does the job:

```
v = −λ L⁺ e
```

```python
L_pinv = damped_pinv(L)   # provided; "damped" = numerically safe near singularities
v = -lam * L_pinv @ e
return v
```

The provided loop (`IBVSServo.run`) then converts `v` into joint speeds, limits them,
moves the robot one step of 0.1 s, takes a new picture and repeats.

**Result:** see the table at the end (error goes from ~1500 px to a few px in 10 s, and the
plot `task1_results.png` shows the exponential decay).

---

## TODO 3: estimate the target's motion   (`ibvs.py → estimate_target_motion`)

In Task 2 the target drives in a circle. Now the feature changes for **two** reasons:

```
ė = L v          +   ∂e/∂t
    (our camera)     (the target moving on its own)
```

Classic IBVS ignores the second part, so it is always chasing and lags behind.

We can't measure `∂e/∂t` directly, but we can compute it:
* the total change we *measured* between two frames is `(s_k − s_{k−1}) / Δt`
* the part *we caused* is `L_{k−1} v_{k−1}` (we know what velocity we commanded last step)
* whatever is left over must be the target's doing:

```python
measured_rate = (s - s_prev) / dt
camera_rate   = L_prev @ v_prev
de_hat        = measured_rate - camera_rate
return de_hat
```

(The provided loop also low-pass filters this estimate with `MOTION_FILTER = 0.5`,
because image measurements are noisy.)

---

## TODO 4: compensated IBVS   (`ibvs.py → compensated_camera_velocity`)

Want `ė = −λ e` again, but now `ė = L v + ∂ê/∂t`. Solve for v:

```
L v + ∂ê/∂t = −λ e   →   L v = −(λ e + ∂ê/∂t)   →   v = −L⁺ (λ e + ∂ê/∂t)
```

```python
L_pinv = damped_pinv(L)
v = -L_pinv @ (lam * e + de_hat)
return v
```

The `de_hat` term is a **feed-forward**: the camera moves along with the target *before* an
error appears, instead of only reacting to error. That's why the steady tracking error falls
by a large factor (table at the end).

---

## Task 3: why plain IBVS fails, and the idea of planning

IBVS is greedy: each step it moves to reduce the error *right now*. It has no idea a pillar
is in the way, so it drives straight into it (`main.py` shows this first, "classic IBVS:
collision").

The fix: **plan first, then move.** We plan a path *in feature space* (sequences of
`[xc, yc, a]`) using **RRT**. The trick in this assignment is that each tree node stores
**two** things:

* the feature `s` (what the camera sees), and
* the robot configuration `q` (joint angles) that produces it.

Because each node has a real `q`, the planner can check real constraints: target in the
image, target not hidden behind the obstacle, no collision, joint limits. (All provided
in `_check`, `_edge_free`.)

### The RRT loop (provided, `plan`)

```
tree = [start]
repeat:
    s_rand  = random feature from the dataset (every 10th time: the goal itself)   ← _sample
    nearest = tree node whose feature is closest to s_rand                         ← _nearest
    q_new   = take one small step from nearest towards s_rand                     ← _extend  (TODO 5)
    if q_new passes all checks: add it to the tree
    if s_new is within 75 px of s*: done
```

---

## TODO 5: one RRT step   (`planner.py → _extend`)

How do we "step towards `s_rand`" when we can only control joint angles?

1. **Desired feature velocity** – go part of the way toward the sample:
   `ṡ = λ_rrt (s_rand − s_parent)` (λ_rrt = 0.5).
2. **Image Jacobian** – chain the two maps: joints → camera velocity (`Jc`, provided) and
   camera velocity → feature velocity (`L`, your TODO 1):
   `J1 = L · Jc`, so `ṡ = J1 q̇`.
3. **Solve for joint velocity** – J1 is 3×6, so there are 6 − 3 = 3 "free" directions.
   We use them for a **secondary task**: keep the camera's orientation fixed
   (`J2 q̇ = 0`, with J2 = angular rows of Jc). `TaskPrioritySolver.solve` (provided) does
   `q̇ = J1⁺ ṡ + (null-space correction for J2)`, i.e. "do the image task exactly, and
   among all ways of doing it pick one that doesn't rotate the camera".
   Then `_limit_vel` scales it down if any joint would move too fast.
4. **Integrate** for `dt = 1 s`: `q_new = q_parent + q̇·dt`.

```python
s_dot = RRT_LAMBDA * (s_rand - s_parent)
Lm = self.Lm.compute(s_parent, z_parent)
J1 = self.Lm.image_jacobian(Lm, Jc)
q_dot = self.tp.solve(J1, s_dot, J2)
q_dot = self._limit_vel(q_dot)
q_new = q_parent + q_dot * dt
return q_new, q_dot, J1
```

After this, the provided code renders the *real* feature at `q_new` (it won't be exactly
what we predicted, that's fine) and checks it.

### Following the plan (provided, `controller.py`)

The robot then follows the planned features `s*_k` one per control step with the law

```
v = −λ L⁺ (s_k − s*_k)  +  L⁺ (s*_k − s*_{k−1}) / Δt
```

That is exactly the Task 2 law: the planned path's velocity plays the role of "target motion".
This is why Task 2 comes before Task 3.

---

## TODO 6: drawing the tree   (`visualizer.py → draw_rrt_tree`)

Pure matplotlib. `points[i]` is node i, `parents[i]` is its parent's index.

```python
for i, j in enumerate(parents):            # every node i with parent j
    if j is not None:                      # root has no parent
        ax.plot(*np.column_stack((points[j], points[i])), color="0.6", lw=0.8)
ax.scatter(*points.T, s=8, color="tab:blue")          # all nodes
if path is not None and len(path) >= 2:
    ax.plot(*points[path].T, color="red", lw=2.5)     # the solution path
```

The `*` trick: `np.column_stack((P, Q))` gives rows `[Px, Qx]`, `[Py, Qy]` (and `[Pz, Qz]` in
3-D). `ax.plot(*rows)` unpacks those rows into `ax.plot(xs, ys)` or `ax.plot(xs, ys, zs)`,
so the same line works for 2-D and 3-D axes.

---

## Task 4: RRT*

RRT accepts the first path it finds, and it zig-zags (each new node just attaches to its
nearest neighbour). RRT* keeps a **cost** for every node = length of the path from the root
(in feature space), and adds two steps for each new node:

## TODO 7: choose parent   (`planner.py → _choose_parent`)

"Of all nearby nodes, which one gives the new node the *cheapest* path from the start?"

```python
best = parent_id
best_cost = cost[parent_id] + np.linalg.norm(s_new - tree_f[parent_id].s)
for i in near_ids:
    c = cost[i] + np.linalg.norm(s_new - tree_f[i].s)
    if c < best_cost and self._connectable(tree_c[i].q, q_new):
        best, best_cost = i, c
return best, best_cost
```

`c < best_cost` is checked first because it's a cheap subtraction, while `_connectable`
runs many collision checks along the edge. Python's `and` stops early if the first part is
False, so we only pay for collision checks when it could actually help.

## TODO 8: rewire   (`planner.py → _rewire`)

The reverse question: "now that the new node exists, would any *neighbour* be cheaper if it
went through the new node?"

```python
new_parent = tree_f[new_id].parent_id
for j in near_ids:
    if j == new_parent or j == new_id:
        continue                       # no point rewiring your own parent to yourself
    c = cost[new_id] + np.linalg.norm(tree_f[j].s - tree_f[new_id].s)
    if c < cost[j] and self._connectable(tree_c[new_id].q, tree_c[j].q):
        tree_f[j].parent_id = new_id   # change parent in the feature tree...
        tree_c[j].parent_id = new_id   # ...and in the configuration tree (they must match)
        self._update_costs(j, c - cost[j], tree_f, cost)   # j and all its children got cheaper
```

Why `_update_costs`? Every descendant of `j` reaches the root *through* `j`, so if `j`'s cost
drops by Δ, all of theirs drop by Δ too.

Could rewiring create a loop? No: if `j` were an ancestor of the new node, the new node's
cost is already ≥ `cost[j]`, so `c > cost[j]` and the `if` fails.

RRT* also doesn't stop at the first solution; it runs all 3000 iterations and returns the
cheapest node that reaches the goal, which is how it ends up with a much shorter path.

---

## How to run it yourself

```bash
pip install -r requirements.txt
python -m unittest tests.test_framework tests.test_task1 tests.test_task2 tests.test_task3 tests.test_task4 -v
python main.py --task1 --preview      # see the scene and your s*
python main.py --task1                # GUI (also --task2, --task3, --task4)
python main.py --all --headless       # everything, writes results/
```

## My results (s* = [40, 30, 2500])

All 37 unit tests pass. Outputs of `python main.py --all --headless` are in `results/`.

| Task | My result | README expects (with s* = [0, 0, 3000]) |
|:---:|:---|:---|
| 1 | error 1067 px → 0.006 px in 10 s (below 10 px after 5.9 s), final s = [39.99, 30.0, 2500] | 1563 → ~3 px |
| 2 | steady tracking error 57 px (classic) → 9.0 px (compensated) | ~62 → ~10 px |
| 3 | classic IBVS: **collision**; RRT path: 0 collisions, min clearance 3.0 cm, final error 62 px | collision; 0 collisions |
| 4 | path cost RRT 5848 → RRT* 1340 (4.4× shorter); 0 collisions | ~6200 → ~1900 |

Why do Tasks 3–4 end 62 / 95 px from s* instead of ~0? The planner stops as soon as a
node is within `RRT_RADIUS_R = 75 px` of s* (the planned path ended 39 px away in Task 4),
and the tracking controller stops at the last waypoint. The error number also mixes px and px²
(most of the 95 is 82 px² of area). The reference solution shows the same thing (73 / 64 px).
You could add a final Task 1 IBVS phase from the end of the path to remove this leftover error.

### Things to look at in `results/`
* `task1_static_ibvs/task1_results.png`: the error curve; on a log axis it is a straight line,
  which is what "exponential decay" means.
* `task2_moving_target/task2_results.png`: classic vs compensated tracking error.
* `task3_rrt/task3_motion_sequence.png`: top row, IBVS crashing into the pillar; bottom row, the RRT path.
* `task4_rrt_star/task4_rrt_vs_rrt_star.png`: the wandering RRT path vs the short RRT* path.

## About the "bonus"

The assignment repository (README, code comments, tests) has **no bonus task**. It has
TODO 0 + TODOs 1–8, and all of them are done here. If your course handout lists a bonus
somewhere else, send it to me and I'll do it.

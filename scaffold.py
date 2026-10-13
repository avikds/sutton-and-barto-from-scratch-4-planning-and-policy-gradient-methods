"""
Sutton and Barto from Scratch 4: Planning and Policy-Gradient Methods scaffold.

Run this with: python scaffold.py
Uses functions defined in model.py.
"""

from model import *  # noqa: F401, F403 (pulls in your solution functions)

"""Sutton & Barto from scratch 4: planning (Dyna-Q, Dyna-Q+, prioritized sweeping) and policy gradients demo."""
import numpy as np


def main():
    # --- Part 1: Dyna-Q on the maze of Figure 8.2 (alpha=0.1, gamma=0.95, epsilon=0.1) ---
    maze = build_maze()
    print("Maze: %dx%d, start %s, goal %s, optimal path 14 steps" % (maze['rows'], maze['cols'], maze['start'], maze['goal']))
    n_runs, n_episodes = 5, 30
    for n_planning in (0, 5, 50):
        curves = np.zeros((n_runs, n_episodes))
        greedy = []
        for run in range(n_runs):
            rng = np.random.default_rng(run)
            q = np.zeros((maze['rows'], maze['cols'], maze['n_actions']))
            model = {}
            for ep in range(n_episodes):
                q, model, steps = dyna_q_episode(maze, q, model, n_planning, 0.1, 0.95, 0.1, rng)
                curves[run, ep] = steps
            greedy.append(greedy_path_length(maze, q))
        mean = curves.mean(axis=0)
        print("Dyna-Q n=%2d | mean steps/episode at episodes 2, 3, 5, 10, 30: %s | greedy path after 30 episodes: %s"
              % (n_planning, [round(float(mean[i - 1]), 1) for i in (2, 3, 5, 10, 30)], greedy))

    # --- Part 2: blocking maze (alpha=1, n=10, kappa=1e-4) and prioritized sweeping ---
    for use_plus in (False, True):
        cums = np.array([run_blocking_maze(use_plus, 3000, 1000, 10, 1e-4, 1.0, 0.95, 0.1, np.random.default_rng(100 + r))
                         for r in range(5)]).mean(axis=0)
        print("%s | mean cumulative reward at step 1000 / 2000 / 3000 (wall moves at 1000): %s"
              % ("Dyna-Q+" if use_plus else "Dyna-Q ", [round(float(cums[i - 1]), 1) for i in (1000, 2000, 3000)]))
    for method in ("Dyna-Q (n=5)", "Prioritized sweeping (n=5)"):
        totals = []
        for run in range(5):
            rng = np.random.default_rng(200 + run)
            q = np.zeros((6, 9, 4))
            model = {}
            updates = 0
            episodes = 0
            while greedy_path_length(maze, q) != 14 and episodes < 50:
                if method.startswith("Dyna"):
                    q, model, steps = dyna_q_episode(maze, q, model, 5, 0.5, 0.95, 0.1, rng)
                    updates += steps * 6
                else:
                    q, model, steps, n_up = prioritized_sweeping_episode(maze, q, model, 5, 1e-4, 0.5, 0.95, 0.1, rng)
                    updates += n_up
                episodes += 1
            totals.append(updates)
        print("%-27s | updates until the greedy path is optimal (5 runs, cap 50 episodes): %s | median %d"
              % (method, totals, int(np.median(totals))))

    # --- Part 3: REINFORCE on the short corridor (theta0 gives P(right)=0.05) ---
    print("Corridor true value v(start): p=0.05 -> %.2f, p=0.5 -> %.2f, p*=2-sqrt(2)=%.4f -> %.2f"
          % (corridor_true_value(0.05), corridor_true_value(0.5), 2 - np.sqrt(2), corridor_true_value(2 - np.sqrt(2))))
    n_runs, n_episodes = 10, 1000
    for method in ("REINFORCE alpha=2^-13", "REINFORCE+baseline a_theta=2^-9 a_w=2^-6"):
        rewards = np.zeros((n_runs, n_episodes))
        p_right = []
        for run in range(n_runs):
            rng = np.random.default_rng(300 + run)
            theta, w = np.array([1.47, -1.47]), 0.0
            for ep in range(n_episodes):
                if method.startswith("REINFORCE alpha"):
                    theta, r = reinforce_episode(theta, 2 ** -13, 1.0, rng)
                else:
                    theta, w, r = reinforce_baseline_episode(theta, w, 2 ** -9, 2 ** -6, 1.0, rng)
                rewards[run, ep] = r
            p_right.append(float(corridor_policy(theta)[1]))
        mean = rewards.mean(axis=0)
        print("%-42s | mean total reward at episodes 1, 100, 300, 1000: %s | mean of last 100: %.2f | final P(right): %.3f"
              % (method, [round(float(mean[i - 1]), 1) for i in (1, 100, 300, 1000)], float(mean[-100:].mean()),
                 float(np.mean(p_right))))

    # --- Part 4: one-step actor-critic on the corridor, differential Sarsa on access control ---
    rewards = np.zeros((n_runs, n_episodes))
    p_right = []
    for run in range(n_runs):
        rng = np.random.default_rng(400 + run)
        theta, w = np.array([1.47, -1.47]), np.zeros(3)
        for ep in range(n_episodes):
            theta, w, r = actor_critic_corridor_episode(theta, w, 2 ** -9, 2 ** -5, 1.0, rng)
            rewards[run, ep] = r
        p_right.append(float(corridor_policy(theta)[1]))
    mean = rewards.mean(axis=0)
    print("%-42s | mean total reward at episodes 1, 100, 300, 1000: %s | mean of last 100: %.2f | final P(right): %.3f"
          % ("Actor-critic a_theta=2^-9 a_w=2^-5", [round(float(mean[i - 1]), 1) for i in (1, 100, 300, 1000)],
             float(mean[-100:].mean()), float(np.mean(p_right))))
    print("Last-run critic values v(s0), v(s1), v(s2):", np.round(w, 2).tolist())

    q, avg_reward = run_access_control(200000, 0.01, 0.01, 0.1, np.random.default_rng(500))
    table = access_control_policy_table(q)
    print("Access control after 200000 steps: average reward estimate %.3f" % avg_reward)
    print("Policy (A=accept, R=reject) by priority, columns = free servers 0..10 (cf. Figure 10.5):")
    for prio, priority in enumerate((1, 2, 4, 8)):
        row = " ".join("A" if a else "R" for a in table[prio])
        first = int(np.argmax(table[prio] == 1)) if table[prio].any() else -1
        print("  priority %d: %s | accept from %d free servers" % (priority, row, first))


if __name__ == "__main__":
    main()


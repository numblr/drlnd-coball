import numpy as np
import torch
from collections import namedtuple

from coball.environment import CoBallEnv, CoBallAgent
from coball.training import PPOLearner
from coball.util import print_progress, plot, start_plot, save_plot


PARAMETER_PATH="parameters.pt"

def run_learner(learner, env, result_path, epochs=15, checkpoint_window=20):
    performances = ()

    episode_cnt = 0
    episode_step = 0
    max_avg_score = -100.0
    for cnt, data in enumerate(learner.train(epochs)):
        episode_step += 1
        performance, score, terminal = data

        if terminal:
            episode_cnt += 1
            episode_step = 0

        if terminal and np.mean(env.get_score_history()[-checkpoint_window:]) > max_avg_score:
            learner.save(result_path)
            print("\nSaved checkpoint in epoch " + str(cnt) \
                    + " with avg score: " + str(np.mean(env.get_score_history()[-checkpoint_window:])) + "\n")
            max_avg_score = np.mean(env.get_score_history()[-checkpoint_window:])

        print_progress(episode_cnt, episode_step, performance.item(), env.get_score_history(), total=8)
        if terminal:
            print("")

    return env.get_score_history(), performances


def run_dummy(env, episodes=10):
    class DummyPolicy:
        def sample(self, states):
            return torch.rand(env.get_agent_size(), env.get_action_size()) * 2.0 - 1.0

    agent = CoBallAgent(DummyPolicy())

    for e in range(episodes):
        episode = enumerate(env.generate_episode(agent))
        for count, step_data in episode:
            # Consume the generated steps
            pass


def replay(env, parameter_path):
    print("Replay from " + parameter_path)

    learner = PPOLearner(env=env)
    learner.load(parameter_path)
    replay_agent = learner.get_agent(0.5)

    for _ in range(100):
        episode = env.generate_episode(replay_agent, train_mode=True)
        for _ in episode:
            # Consume the generated steps
            pass

    print("Average score on 100 episodes: " + str(np.mean(env.get_score_history())))


def run_demo(env, parameter_path):
    print("Replay from " + parameter_path)

    learner = PPOLearner(env=env)
    learner.load(parameter_path)
    replay_agent = learner.get_agent(0.2)

    episode = env.generate_episode(replay_agent)

    for count, step_data in enumerate(episode):
        # Consume the generated steps
        pass

    print("Score:    " + str(env.get_score()))


def learn(env, epochs, parameter_path):
    print("\nStart learning\n")

    learner = PPOLearner(env=env)

    scores, losses = run_learner(learner, env, parameter_path, epochs=epochs)

    print("\nStore results\n")

    plot(scores, path="scores.png", windows=[1, 100],
            colors=['r', 'g'], labels=["Agent avg", "100 episode avg"])

    torch.save(scores, "scores.pt")

    return scores, losses


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-r','--replay',
            help='Replay 100 epsisodes from the stored parameters',
            action='store_true', required=False)
    parser.add_argument('-s','--show',
            help='Demonstrate from the stored parameters',
            action='store_true', required=False)
    parser.add_argument('-d','--dummy',
            help='Demonstrate the environment with a dummy policy',
            action='store_true', required=False)
    parser.add_argument('-n','--epochs',
            help='Number of epochs used for training',
            type=int, default=16, required=False)

    args = parser.parse_args()

    env=CoBallEnv()

    if args.replay:
        replay(env, PARAMETER_PATH)
    elif args.show:
        run_demo(env, PARAMETER_PATH)
    elif args.dummy:
        run_dummy(env)
    else:
        scores, losses = learn(env, args.epochs, PARAMETER_PATH)

        plot(scores, windows=[1, 100], colors=['b', 'g'], labels=["Score", "Avg"], path=None)

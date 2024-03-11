import datetime
import itertools
import json
import os
import sys
import tempfile
import time

from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.algorithms.alpha_zero.alpha_zero import watcher, Buffer, _init_model_from_config, Config, actor, \
    evaluator
import pyspiel
from open_spiel.python.utils import data_logger
from open_spiel.python.utils import spawn
from open_spiel.python.utils import stats

from src.alphazero_scaling.sampling.kl_sampling import Sampler
from collections import Counter
import pickle

# Time to wait for processes to join.
JOIN_WAIT_DELAY = 0.001


@watcher
def learner(*, game, config, actors, evaluators, broadcast_fn, logger, count_states=False, b=None):
    """A learner that consumes the replay buffer and trains the network."""
    ###
    sampler = Sampler()
    if b is not None:
        sampler.b = b
    played_counter = Counter()
    sample_counter = Counter()
    ###
    logger.also_to_stdout = True
    replay_buffer = Buffer(config.replay_buffer_size)
    learn_rate = config.replay_buffer_size // config.replay_buffer_reuse
    logger.print("Initializing model")
    model = _init_model_from_config(config)
    logger.print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
                                             config.nn_depth))
    logger.print("Model size:", model.num_trainable_variables, "variables")
    save_path = model.save_checkpoint(0)
    logger.print("Initial checkpoint:", save_path)
    broadcast_fn(save_path)

    data_log = data_logger.DataLoggerJsonLines(config.path, "learner", True)

    stage_count = 7
    value_accuracies = [stats.BasicStats() for _ in range(stage_count)]
    value_predictions = [stats.BasicStats() for _ in range(stage_count)]
    game_lengths = stats.BasicStats()
    game_lengths_hist = stats.HistogramNumbered(game.max_game_length() + 1)
    outcomes = stats.HistogramNamed(["Player1", "Player2", "Draw"])
    evals = [Buffer(config.evaluation_window) for _ in range(config.eval_levels)]
    total_trajectories = 0

    def trajectory_generator():
        """Merge all the actor queues into a single generator."""
        while True:
            found = 0
            for actor_process in actors:
                try:
                    yield actor_process.queue.get_nowait()
                except spawn.Empty:
                    pass
                else:
                    found += 1
            if found == 0:
                time.sleep(0.01)  # 10ms

    def collect_trajectories():
        """Collects the trajectories from actors into the replay buffer."""
        num_trajectories = 0
        num_states = 0
        for trajectory in trajectory_generator():
            num_trajectories += 1
            num_states += len(trajectory.states)
            game_lengths.add(len(trajectory.states))
            game_lengths_hist.add(len(trajectory.states))

            p1_outcome = trajectory.returns[0]
            if p1_outcome > 0:
                outcomes.add(0)
            elif p1_outcome < 0:
                outcomes.add(1)
            else:
                outcomes.add(2)

            ###
            sampled_states = sampler.kl_sampling(trajectory, model)
            if count_states:
                played_counter.update(str(s.observation) for s in trajectory.states)
                sample_counter.update(str(s.observation) for s in sampled_states)

            # Warm-up, ignore the sampled states on the first step:
            if len(replay_buffer) < learn_rate:
                sampled_states = trajectory.states

            replay_buffer.extend(
                model_lib.TrainInput(
                    s.observation, s.legals_mask, s.policy, p1_outcome)
                for s in sampled_states)
            ###

            for stage in range(stage_count):
                # Scale for the length of the game
                index = (len(trajectory.states) - 1) * stage // (stage_count - 1)
                n = trajectory.states[index]
                accurate = (n.value >= 0) == (trajectory.returns[n.current_player] >= 0)
                value_accuracies[stage].add(1 if accurate else 0)
                value_predictions[stage].add(abs(n.value))

            if num_states >= learn_rate:
                break
        return num_trajectories, num_states

    def learn(step):
        """Sample from the replay buffer, update weights and save a checkpoint."""
        ###
        with open(config.path + '/played_counts.pkl', 'wb') as f:
            pickle.dump([c for _, c in played_counter.most_common()], f)
        with open(config.path + '/sample_counts.pkl', 'wb') as f:
            pickle.dump([c for _, c in sample_counter.most_common()], f)
        ###
        losses = []
        for _ in range(len(replay_buffer) // config.train_batch_size):
            data = replay_buffer.sample(config.train_batch_size)
            losses.append(model.update(data))

        # Always save a checkpoint, either for keeping or for loading the weights to
        # the actors. It only allows numbers, so use -1 as "latest".
        save_path = model.save_checkpoint(
            step if step % config.checkpoint_freq == 0 else -1)
        losses = sum(losses, model_lib.Losses(0, 0, 0)) / len(losses)
        logger.print(losses)
        logger.print("Checkpoint saved:", save_path)
        return save_path, losses

    last_time = time.time() - 60
    for step in itertools.count(1):
        for value_accuracy in value_accuracies:
            value_accuracy.reset()
        for value_prediction in value_predictions:
            value_prediction.reset()
        game_lengths.reset()
        game_lengths_hist.reset()
        outcomes.reset()

        num_trajectories, num_states = collect_trajectories()
        total_trajectories += num_trajectories
        now = time.time()
        seconds = now - last_time
        last_time = now
        ###
        ratio, a, target = sampler.update_hyperparameters()
        ###

        logger.print("Step:", step)
        logger.print(
            ("Collected {:5} states from {:3} games, {:.1f} states/s. "
             "{:.1f} states/(s*actor), game length: {:.1f}").format(
                num_states, num_trajectories, num_states / seconds,
                                              num_states / (config.actors * seconds),
                                              num_states / num_trajectories))
        logger.print("Buffer size: {}. States seen: {}".format(
            len(replay_buffer), replay_buffer.total_seen))
        ###
        logger.print("Sampling ratio: {:.5f}. Coeff.: {:.5f}, Target coeff.: {:.5f}.".format(
            ratio, a, target))
        if count_states:
            logger.print("Unique states played: {}.".format(
                len(played_counter)))
        ###

        save_path, losses = learn(step)

        for eval_process in evaluators:
            while True:
                try:
                    difficulty, outcome = eval_process.queue.get_nowait()
                    evals[difficulty].append(outcome)
                except spawn.Empty:
                    break

        batch_size_stats = stats.BasicStats()  # Only makes sense in C++.
        batch_size_stats.add(1)
        data_log.write({
            "step": step,
            "total_states": replay_buffer.total_seen,
            "states_per_s": num_states / seconds,
            "states_per_s_actor": num_states / (config.actors * seconds),
            "total_trajectories": total_trajectories,
            "trajectories_per_s": num_trajectories / seconds,
            "queue_size": 0,  # Only available in C++.
            "game_length": game_lengths.as_dict,
            "game_length_hist": game_lengths_hist.data,
            "outcomes": outcomes.data,
            "value_accuracy": [v.as_dict for v in value_accuracies],
            "value_prediction": [v.as_dict for v in value_predictions],
            "eval": {
                "count": evals[0].total_seen,
                "results": [sum(e.data) / len(e) if e else 0 for e in evals],
            },
            "batch_size": batch_size_stats.as_dict,
            "batch_size_hist": [0, 1],
            "loss": {
                "policy": losses.policy,
                "value": losses.value,
                "l2reg": losses.l2,
                "sum": losses.total,
            },
            "cache": {  # Null stats because it's hard to report between processes.
                "size": 0,
                "max_size": 0,
                "usage": 0,
                "requests": 0,
                "requests_per_s": 0,
                "hits": 0,
                "misses": 0,
                "misses_per_s": 0,
                "hit_rate": 0,
            },
        })
        logger.print()

        if config.max_steps > 0 and step >= config.max_steps:
            break

        broadcast_fn(save_path)


def alpha_zero(config: Config, count_states=False, b=None):  ###
    """Start all the worker processes for a full alphazero setup."""
    game = pyspiel.load_game(config.game)
    config = config._replace(
        observation_shape=game.observation_tensor_shape(),
        output_size=game.num_distinct_actions())

    print("Starting game", config.game)
    if game.num_players() != 2:
        sys.exit("AlphaZero can only handle 2-player games.")
    game_type = game.get_type()
    if game_type.reward_model != pyspiel.GameType.RewardModel.TERMINAL:
        raise ValueError("Game must have terminal rewards.")
    if game_type.dynamics != pyspiel.GameType.Dynamics.SEQUENTIAL:
        raise ValueError("Game must have sequential turns.")
    if game_type.chance_mode != pyspiel.GameType.ChanceMode.DETERMINISTIC:
        raise ValueError("Game must be deterministic.")

    path = config.path
    if not path:
        path = tempfile.mkdtemp(prefix="az-{}-{}-".format(
            datetime.datetime.now().strftime("%Y-%m-%d-%H-%M"), config.game))
        config = config._replace(path=path)

    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.isdir(path):
        sys.exit("{} isn't a directory".format(path))
    print("Writing logs and checkpoints to:", path)
    print("Model type: %s(%s, %s)" % (config.nn_model, config.nn_width,
                                      config.nn_depth))

    with open(os.path.join(config.path, "config.json"), "w") as fp:
        fp.write(json.dumps(config._asdict(), indent=2, sort_keys=True) + "\n")

    actors = [spawn.Process(actor, kwargs={"game": game, "config": config,
                                           "num": i})
              for i in range(config.actors)]
    evaluators = [spawn.Process(evaluator, kwargs={"game": game, "config": config,
                                                   "num": i})
                  for i in range(config.evaluators)]

    def broadcast(msg):
        for proc in actors + evaluators:
            proc.queue.put(msg)

    try:
        learner(game=game, config=config, actors=actors,  # pylint: disable=missing-kwoa
                evaluators=evaluators, broadcast_fn=broadcast, count_states=count_states, b=b)  ###
    except (KeyboardInterrupt, EOFError):
        print("Caught a KeyboardInterrupt, stopping early.")
    finally:
        broadcast("")
        # for actor processes to join we have to make sure that their q_in is empty,
        # including backed up items
        for proc in actors:
            while proc.exitcode is None:
                while not proc.queue.empty():
                    proc.queue.get_nowait()
                proc.join(JOIN_WAIT_DELAY)
        for proc in evaluators:
            proc.join()

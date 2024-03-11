"""Starting point for playing with the AlphaZero algorithm.

Make changes here, the original code is saved in alpha0_example.
"""
from absl import app
from absl import flags

from open_spiel.python.algorithms.alpha_zero import alpha_zero
from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.utils import spawn

import sys
from src.alphazero_scaling.sampling.alphazero_with_sampling import alpha_zero
from src.general.general_utils import Timer

flags.DEFINE_integer("uct_c", 2, "UCT's exploration constant.")
flags.DEFINE_integer("max_simulations", 300,
                     "How many simulations to run.")  # changed  # Simulations = num. of MC iterations
flags.DEFINE_integer("train_batch_size", 2 ** 10, "Batch size for learning.")
flags.DEFINE_integer("replay_buffer_size", 2 ** 16,
                     "How many states to store in the replay buffer.")
flags.DEFINE_integer("replay_buffer_reuse", 10,
                     "How many times to learn from each state.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_float("weight_decay", 0.0001, "L2 regularization strength.")
flags.DEFINE_float("policy_epsilon", 0.25, "What noise epsilon to use.")
flags.DEFINE_float("policy_alpha", 1, "What dirichlet noise alpha to use.")
flags.DEFINE_float("temperature", 1,
                   "Temperature for final move selection.")
flags.DEFINE_integer("temperature_drop", 30,  # 5 for pentago, 15 for connect four, 50 for oware, 30 for go
                     "Drop the temperature to 0 after this many moves.")
flags.DEFINE_enum("nn_model", "mlp", model_lib.Model.valid_model_types,  # <===== Here
                  "What type of model should be used?.")  # mlp, conv2d, resnet

flags.DEFINE_integer("nn_depth", 2, "How deep should the network be.")

flags.DEFINE_integer("checkpoint_freq", 10, "Save a checkpoint every N steps.")  # 10  # set to min. 10
""" About steps: the length of a 'step' is the time it takes to collect a
number of game states equal to: replay_buffer_size/replay_buffer_reuse.
So increasing the reuse should make steps change more quickly.
"""

flags.DEFINE_integer("evaluators", 1, "How many evaluators to run.")
flags.DEFINE_integer("evaluation_window", 100,
                     "How many games to average results over.")
flags.DEFINE_integer("eval_levels", 4,  # 4#7  #changed
                     "Play evaluation games vs MCTS+Solver, with max_simulations*10^(n/2)"
                     " simulations for n in range(eval_levels). Default of 7 means "
                     "running mcts with up to 1000 times more simulations.")
flags.DEFINE_integer("max_steps", 10000, "How many learn steps before exiting.")  # 0 never exits.      <==== Here
flags.DEFINE_bool("quiet", True, "Don't show the moves as they're played.")
flags.DEFINE_bool("verbose", False, "Show the MCTS stats of possible moves.")

FLAGS = flags.FLAGS

import psutil

print('CPU cores available:')
print(psutil.cpu_count())


def main():
    game = 'oware'  # <========== Change here

    actors = 79  # 39 oware #19quoridor #30go  # best is num. cores -1
    evaluators = 1
    i = int(sys.argv[1])
    iteration = int(sys.argv[2])

    print('game: ', game)
    print('actors: ', actors)
    print('nn_width: ', 2 ** (i + 2))

    print('~~~~~~~~~~~~~~~~~~~~~  MODEL NUM. ' + str(i) + '  ~~~~~~~~~~~~~~~~~~~~~')
    dir_name = "q_" + str(i) + "_" + str(iteration)
    path = '/scratch/compmatsc/neumann/models/oware/' + dir_name  # <============ Change here
    nn_width = 2 ** (i + 2)
    print('nn_width: ', nn_width)

    def run_agz(unused_argv):
        config = alpha_zero.Config(
            game=game,
            path=path,
            learning_rate=FLAGS.learning_rate,
            weight_decay=FLAGS.weight_decay,
            train_batch_size=FLAGS.train_batch_size,
            replay_buffer_size=FLAGS.replay_buffer_size,
            replay_buffer_reuse=FLAGS.replay_buffer_reuse,
            max_steps=FLAGS.max_steps,
            checkpoint_freq=FLAGS.checkpoint_freq,

            actors=actors,
            evaluators=evaluators,
            uct_c=FLAGS.uct_c,
            max_simulations=FLAGS.max_simulations,
            policy_alpha=FLAGS.policy_alpha,
            policy_epsilon=FLAGS.policy_epsilon,
            temperature=FLAGS.temperature,
            temperature_drop=FLAGS.temperature_drop,
            evaluation_window=FLAGS.evaluation_window,
            eval_levels=FLAGS.eval_levels,

            nn_model=FLAGS.nn_model,
            nn_width=nn_width,
            nn_depth=FLAGS.nn_depth,
            observation_shape=None,
            output_size=None,

            quiet=FLAGS.quiet,
        )

        alpha_zero(config)

    timer = Timer()
    try:
        with spawn.main_handler():
            app.run(run_agz)
    except TypeError as e:
        if str(e) == 'an integer is required (got type NoneType)':
            # This error always pops up when using IPython. ignore it.
            print('.')
        else:
            raise e
    timer.stop()


if __name__ == "__main__":
    main()

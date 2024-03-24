#from src.alphazero_scaling.alphazero_mods.policy_sampling import AlphaZeroKLSampling
#from src.alphazero_scaling.alphazero_mods.resignation import AlphaZeroWithResignation
#from src.alphazero_scaling.alphazero_mods.uniform_sampling import AlphaZeroUniformSampling
import src.alphazero_scaling.alphazero_base as base


from open_spiel.python.algorithms.alpha_zero import model as model_lib
from open_spiel.python.utils import spawn

from src.alphazero_scaling.alphazero_base import AlphaZero, JOIN_WAIT_DELAY, TrajectoryState
from src.alphazero_scaling.sampling.kl_sampling import Sampler
from collections import Counter, defaultdict
import pickle
from open_spiel.python.algorithms.alpha_zero import model as model_lib
import src.alphazero_scaling.alphazero_base as base
from collections import deque
import numpy as np
from src.alphazero_scaling.alphazero_mods.uniform_sampling import UniqueBuffer
from src.alphazero_scaling.alphazero_mods.policy_sampling import TrajectoryStateWithMoves

class MixedAll(base):
    def __init__(self, count_states=False, b=0.01, gamma=0.8 ,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count_states = count_states
        self.b = b
        self.sampler = Sampler(b=self.b)
        self.played_counter = Counter()
        self.sample_counter = Counter()
        self.turns_played = defaultdict(int)
        self.turns_sampled = defaultdict(int)

        self.gamma = gamma
        self.n_tests = 0
        self.target_rate = 0.05
        self.disable_resign_rate = 0.1
        self.test_values = deque(maxlen=400)
        self.fp_mask = deque(maxlen=400)
        self.warmup = True
        self.end_warmup_step = 30
        self.n_resigned = 0
        self.warmup_path = self.config.path + '/warmup_flag.npy'
        with open(self.warmup_path, 'wb') as f:
            np.save(f, self.warmup)

        self.v_resign = -0.5
        self.target_v = self.v_resign
        self.v_resign_path = self.config.path + '/v_resign.npy'
        with open(self.v_resign_path, 'wb') as f:
            np.save(f, self.v_resign)

        self.replay_buffer = UniqueBuffer(self.config.replay_buffer_size)

    @staticmethod
    def _create_trajectory_state(state, action, policy, root):
        return TrajectoryStateWithMoves(state.move_number(), state.observation_tensor(), state.current_player(),
                                        state.legal_actions_mask(), action, policy,
                                        root.total_reward / root.explore_count)
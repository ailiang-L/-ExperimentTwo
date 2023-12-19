from stable_baselines3.common.callbacks import BaseCallback
import os
import time


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0
        self.model_path = '../model/' + time.strftime('%Y-%m-%d-%H-%M', time.localtime())
        os.makedirs(self.model_path, exist_ok=True)
        os.makedirs('../log', exist_ok=True)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env = None  # type: Union[gym.Env, VecEnv, None]
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = None  # type: Dict[str, Any]
        # self.globals = None  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger = None  # stable_baselines3.common.logger
        # # Sometimes, for event callback, it is useful
        # # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_training_start(self) -> None:
        """
        This method is called before the first rollout starts.
        """
        pass

    def _on_rollout_start(self) -> None:
        """
        A rollout is the collection of environment interaction
        using the current policy.
        This event is triggered before collecting new samples.
        """
        pass

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        if 'done' in self.locals['infos'][0]:
            self.episode_count += 1
            if self.episode_count % 2000 == 0:
                self.model.save(self.model_path + "/episode-" + str(self.episode_count))

            if self.episode_count % 10 == 0:
                total_delay = self.locals['infos'][0]['total_delay']
                energy_cost = self.locals['infos'][0]['energy_cost']
                episode_reward = self.locals['infos'][0]['episode_reward']
                episode_length = self.locals['infos'][0]['episode']['l']

                self.logger.record("episode_reward", episode_reward)
                self.logger.record("energy_cost", energy_cost)
                self.logger.record("total_delay", total_delay)
                self.logger.record("episode_length", episode_length)
                self.logger.dump(step=self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        """
        This event is triggered before updating the policy.
        """
        pass

    def _on_training_end(self) -> None:
        """
        This event is triggered before exiting the `learn()` method.
        """
        pass

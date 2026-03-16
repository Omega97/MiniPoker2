import random
from pathlib import Path
from mini_poker.paths import DATA_DIR


class AgentTrainer:
    """
    Manages the lifecycle of a counterfactual agent, handling
    loading, training, saving, and policy display.
    """
    def __init__(self, agent, random_seed: int = 0, show_policy: bool = False,
                 data_dir: Path = None, force_training=False):
        """
        Initialize the trainer.

        Args:
            agent: The agent object to train or load.
            random_seed: Seed for reproducibility.
            show_policy: Whether to print the policy after training/loading.
            data_dir: Directory for saving/loading models. Defaults to global DATA_DIR.
            force_training: force training (no loading)
        """
        self.agent = agent
        self.random_seed = random_seed
        self.show_policy = show_policy
        self.data_dir = data_dir or DATA_DIR  # Fallback to global if not provided
        self.filepath = self.data_dir / f"{agent}.json"
        self.force_training = force_training

    def run(self):
        """
        Execute the training workflow: try load, else train, then save.

        Returns:
            The agent object.
        """
        random.seed(self.random_seed)

        # Loading
        if not self.force_training:
            if self._try_load():
                # Show Policy
                if self.show_policy:
                    self._display_policy()
                return self.agent

        # Training & Save
        self._train()
        self._save()

        # Show Results
        if self.show_policy:
            self._display_policy()

        return self.agent

    def _try_load(self) -> bool:
        """Attempt to load the agent from disk. Returns True if successful."""
        try:
            self.agent.load(self.filepath)
            print(f"Agent {self.agent} loaded")
            return True
        except FileNotFoundError:
            return False

    def _train(self):
        """Trigger the agent's training process."""
        print(f"\nTraining {self.agent}")
        self.agent.train()

    def _save(self):
        """Save the agent to disk."""
        self.agent.save(self.filepath)

    def _display_policy(self):
        """Print the agent's current policy."""
        print(self.agent.show_policy())

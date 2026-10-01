"""Train SlotSSM actions with learned, non-oracle subgoal states.

This entry point delegates to ``finetune-3_slotssm_to_action.py``, whose
training loop jointly optimizes the predicted per-slot subgoal head and the
continuous action decoder. All command-line arguments are forwarded unchanged.
"""

from pathlib import Path
import runpy


if __name__ == "__main__":
    runpy.run_path(
        str(Path(__file__).with_name("finetune-3_slotssm_to_action.py")),
        run_name="__main__",
    )

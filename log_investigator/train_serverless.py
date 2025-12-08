import asyncio
import os
import random
from pathlib import Path
from typing import List

import art
from dotenv import load_dotenv
from art.serverless.backend import ServerlessBackend

from rollout import rollout, IncidentScenario
from project_types import ProjectPolicyConfig, TrainingConfig

load_dotenv()

import warnings

warnings.filterwarnings(
    "ignore",
    message="Pydantic serializer warnings:",
    module="pydantic.functional_validators",
)


MODEL_NAME = "log-investigator-v22"
PROJECT_NAME = "log-investigator"
BASE_MODEL = "OpenPipe/Qwen3-14B-Instruct"
TRAIN_STEPS = 61
TRAJECTORIES_PER_GROUP = 4
GROUPS_PER_STEP = 4


def load_incident_scenarios(path: Path) -> List[IncidentScenario]:
    scenarios: List[IncidentScenario] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            scenarios.append(IncidentScenario.parse_raw(line))
    return scenarios

async def main():
    if not os.getenv("WANDB_API_KEY"):
        print("ERROR: WANDB_API_KEY not set")
        return
    
    model = art.TrainableModel[ProjectPolicyConfig](
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=BASE_MODEL,
        config=ProjectPolicyConfig(
            max_turns=10,
            max_tokens=2048,
            use_tools=False,
            log_to_openpipe=True,
            training_config=TrainingConfig(
                trajectories_per_group=TRAJECTORIES_PER_GROUP,
                groups_per_step=GROUPS_PER_STEP,
                learning_rate=1.2e-5,
                eval_steps=10,
                val_set_size=5,
                training_dataset_size=40,
                num_epochs=1,
            ),
        ),
    )

    backend = ServerlessBackend()
    await model.register(backend)

    scenarios = load_incident_scenarios(
        Path("log_scenarios/train/log_incident_scenarios.jsonl")
    )
    if not scenarios:
        print("ERROR: No scenarios found")
        return

    split_idx = int(0.8 * len(scenarios))

    train_scen = scenarios[:split_idx]
    val_scen = scenarios[split_idx:]

    for step in range(await model.get_step(), TRAIN_STEPS):
        print(f"\nStep {step+1}/{TRAIN_STEPS}")

        # Validation
        if step % 5 == 0:
            val_groups = []
            for s in val_scen:
                traj = await rollout(model, s)
                val_groups.append(art.TrajectoryGroup([traj]))

            await model.log(val_groups, split="val")
            
        # Epoch-based shuffled sampling
        # Compute how many steps form one epoch
        steps_per_epoch = (len(train_scen) + GROUPS_PER_STEP - 1) // GROUPS_PER_STEP

        # Shuffle at the start of training, and again at each epoch boundary
        if step == 0 or step % steps_per_epoch == 0:
            random.shuffle(train_scen)

        # Select contiguous block from shuffled order
        start = (step % steps_per_epoch) * GROUPS_PER_STEP
        step_scenarios = train_scen[start:start + GROUPS_PER_STEP]

        # If block is shorter at end of epoch, wrap around
        if len(step_scenarios) < GROUPS_PER_STEP:
            step_scenarios += train_scen[: GROUPS_PER_STEP - len(step_scenarios)]

        train_groups = await art.gather_trajectory_groups(
            (
                art.TrajectoryGroup(
                    [rollout(model, s) for _ in range(TRAJECTORIES_PER_GROUP)]
                )
                for s in step_scenarios
            ),
            pbar_desc=f"train step {step}",
        )

        await model.train(
            train_groups,
            config=art.TrainConfig(
                learning_rate=model.config.training_config.learning_rate,
            ),
        )


if __name__ == "__main__":
    asyncio.run(main())

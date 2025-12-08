# This class is used for making a base model to be used as a Judge

import asyncio
import os

from dotenv import load_dotenv
import art
from art.serverless.backend import ServerlessBackend
from pydantic import BaseModel

load_dotenv()

class LogInvestigatorBaseConfig(BaseModel):
    max_turns: int = 10
    max_tokens: int = 2048

PROJECT_NAME = "log-investigator"
BASE_MODEL = "OpenPipe/Qwen3-14B-Instruct"
MODEL_NAME = "log-investigator-judge"

async def main():
    if not os.getenv("WANDB_API_KEY"):
        print("WANDB_API_KEY not set")
        return

    model = art.TrainableModel[LogInvestigatorBaseConfig](
        name=MODEL_NAME,
        project=PROJECT_NAME,
        base_model=BASE_MODEL,
        config=LogInvestigatorBaseConfig(),
    )

    backend = ServerlessBackend()
    await model.register(backend)

    print("Base judge model registered.")
    print("Inference id:", model.get_inference_name())

if __name__ == "__main__":
    asyncio.run(main())

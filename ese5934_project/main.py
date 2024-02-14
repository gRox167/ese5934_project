# %%
from lightning.pytorch.cli import LightningCLI, ArgsType
import torch


def cli_main(args: ArgsType = None, run: bool = True):
    cli = LightningCLI(subclass_mode_model=True, args=args, save_config_callback=None, run=run, parser_kwargs={"parser_mode": "omegaconf"})
    return cli

if __name__ == "__main__":
    cli_main()



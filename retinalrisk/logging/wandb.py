import wandb


class WandBTemporaryRun:
    def __init__(self, project: str, entity: str, **kwargs):
        self.project = project
        self.entity = entity
        self.kwargs = kwargs

    def __enter__(self):
        self.run = wandb.init(project=self.project, entity=self.entity, **self.kwargs)
        return self.run

    def __exit__(self, type, value, traceback):
        self.run.finish()

        api = wandb.Api()
        run = api.run(f"cardiors/EHRGraphs/{self.run.id}")
        run.delete()

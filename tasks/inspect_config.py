from core.base_task import Task

class InspectConfig(Task):
    def __init__(self):
        super().__init__("InspectConfig")

    def run(self):
        print(f"[{self.name}] Inspecting Mamba2Config...")
        print(self.model.config)

    def report(self):
        pass

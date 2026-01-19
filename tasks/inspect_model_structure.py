from core.base_task import Task

class InspectModelStructure(Task):
    def __init__(self):
        super().__init__("InspectModelStructure")

    def run(self):
        print(f"[{self.name}] Printing Model Module Structure...")
        print(self.model)

    def report(self):
        pass

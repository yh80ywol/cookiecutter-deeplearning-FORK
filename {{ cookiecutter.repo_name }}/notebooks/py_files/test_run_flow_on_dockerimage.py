from prefect import Flow, task
from prefect.run_configs import LocalRun,DockerRun
from prefect.storage import Local

#Klassenerstellung
class eins:
    @task
    def function():
        print(1)
        return 1
Test=eins()

#Flow
#path="..." muss mit relativen Pfad der File INNERHALB des Containers übereinstimmen!
#run_config=DockerRun(image="...") Pfad vom Image
with Flow("Prefect_Test_Flow",storage=Local(path="/app/test_run_flow_on_dockerimage.py",stored_as_script=True), 
           run_config=DockerRun(image="neu12")) as flow:
    a = Test.function()

flow.register(project_name="test")
#flow.run()


# MLOps Challenge
As are result you will be able to 
- version all changes in your jupyter notebook
- trigger trainings and model creation in a structured manner based on code changes you consider as "good enough" to be shared 
- trigger trainings and model creation in a structured manner based when new data shall be used by referencing the version of the dataset
- deploy your models so they can be tested
- deploy your models after a succesfull test to the next environment (e.g. Production)

## This challenge builds up on challenge 1. 
You will extend the solution of challenge 1 to enable automated trainings based on Azure DevOps.  
### 1. Put your code into source control
1. Create an new project called MLOps-Iris in your Azure DevOps organisation - if you don't have any, create one on http://dev.azure.com
1. In Azure Repos create an empty repo (without a readme file)
1. Open challenge 1 in your Azure ML.
1. Open the command line right here in your notebook (see right here on this page right to your compute drop-down box the little C:\ icon )
1. Navigate in the commandline to the root directory of challenge 1 (depends on the directory where you currently are)
    ```
    cd ../challenge_1
    ```
1. initialize a git repo
   ```
   git init
   ```
1. Set your username and useremail in git to be able to push new code
   ```
   git config user.name "Your Name"
   git config user.email "youremail@yourdomain.com"
   ```
1. Commit the files to git.
    ```
    git add .
    git commit -m "initial"
    ```    
1. Copy the following two lines from your newly created Azure DevOps Repo to push an existing repo.
   ```
   git remote add origin https://msdevops@dev.azure.com/msdevops/MLE2E/_git/mydemorepo
   git push -u origin --all
   ```
1. When prompted for the password create a new Personal Access Token in Azure DevOps and provide it as password (remember the PAT for later use!)
1. You succeeded when you see your code in Azure DevOps.
1. For all subsequent code changes you could always run the following lines to push code to Azure DevOps.
   ```
   git add .                            # adds all files
   git commit -m "my comment"           # commits changes
   git push                             # pushes changes to your upstream repo on Azure DevOps.
   ```
   
### 2. Automate training and model creation
1. Create a file "train_call.py" next to your train.py file. This file is used to call your train.py script with a bunch of parameters.
1. Add the following content to it.
    ```
    from azureml.core import Workspace, Dataset, Datastore, Experiment, Run
    from train import main
    from azureml.train.estimator import Estimator
    from azureml.core import Model
    from azureml.core.resource_configuration import ResourceConfiguration
    import sys

    subscription_id = sys.argv[1]
    print(subscription_id)
    rg = sys.argv[2]
    print(rg)
    workspace_name = sys.argv[3]
    dsname = sys.argv[4]
    entryscript = sys.argv[5]
    computetarget = sys.argv[6]
    modelname = sys.argv[7]

    def main(): 
        script_params = {
            "--kernel": "linear",
            "--penalty": 1.0
        }

        ws = Workspace(subscription_id = subscription_id, resource_group = rg, workspace_name = workspace_name)
        irisdata = Dataset.get_by_name(ws, dsname)
        print("... got dataset ...")

        est = Estimator(
            source_directory='train', 
            entry_script=entryscript,
            script_params=script_params,
            inputs = [irisdata.as_named_input("iris")],
            # inputs=[tabular_dataset.as_named_input("iris")],
            compute_target= computetarget, #compute_target,
            pip_packages=["azureml-dataset-runtime[fuse]==1.12.0", "scikit-learn==0.23.2", "pandas==1.1.1", "matplotlib==3.3.1"]
        )

        print("... est created ...")

        
        experiment_name = "experiment-with-devops"
        exp = Experiment(workspace=ws, name=experiment_name)

        run = exp.submit(est)

        print("... start run  ...")
        run

        run.wait_for_completion(
        show_output=True,
        wait_post_processing=True
        )
        

        print("... register model ...")

        model = run.register_model(
            model_name=modelname,
            model_path="outputs/model.pkl",
            model_framework=Model.Framework.SCIKITLEARN,
            model_framework_version="0.23.2",
            datasets=[("Training dataset", irisdata)],
            resource_configuration=ResourceConfiguration(cpu=1, memory_in_gb=0.5),
            description="SVC classification for iris dataset.",
            tags={"area": "iris", "type": "svc"}
        )

        print("Model name: " + model.name, 
            "Model id: " + model.id, 
            "Model version: " + str(model.version), sep="\n")

        f = open("modelinfo.json", "a")
        f.write("{\"modelname\":\""+model.name+"\",\"modelversion\":"+  str(model.version)+"}")
        f.close()
        


    if __name__ == "__main__":
        main()

    ```
1. Add a script to add python dependencies called install_requirements.sh in the folder challenge_1/train_dataset/setup_requirements with the following code.
   ``` 
   #!/bin/bash
   python --version
   pip install azure-cli==2.15.1
   pip install --upgrade azureml-sdk[cli]
   pip install -r requirements.txt
   ```
1. Add the requirements.txt file in the same folder with the following content.
    ```
    matplotlib
    numpy
    pandas
    joblib
    sklearn
    ```
1. Now create a new yaml starter pipeline in Azure DevOps.
1. Remove all content and replace it with this code. Make sure you don't mess up the formatting.
    ```
    trigger:
    - master

    pool:
    vmImage: 'ubuntu-latest'

    stages:
    - stage: 
        displayName: TrainAndRegister
        
        jobs:
        - job: Job_1
        displayName: Agent job 1
        pool:
            vmImage: ubuntu-18.04
        steps:
        - checkout: self
        - task: UsePythonVersion@0
            displayName: Use Python 3.6
            inputs:
            versionSpec: 3.6
        - task: Bash@3
            displayName: Bash Script
            inputs:
            filePath: train_dataset/setup_requirements/install_requirements.sh
            workingDirectory: train_dataset/setup_requirements
        - task: AzureCLI@2
            displayName: Azure CLI train_call.py
            inputs:
            connectedServiceNameARM: YOURSERVICENAME
            scriptType: bash
            scriptLocation: inlineScript
            scriptPath: train_call.py
            inlineScript: python train/train_call.py '$(subscriptionid)' '$(resourcegroup)' '$(workspacename)' '$(datasetname)' '$(trainingscriptname)' '$(clustername)' '$(modelname)'

        - task: PublishPipelineArtifact@1
            inputs:
            targetPath: './modelinfo.json'
            ArtifactName: 'modelinfo'
            publishLocation: 'pipeline'

    ```
1. Replace YOURSERVICENAME in the script above with the name of your service connection to Azure.
1. Set variables in your pipeline for subscriptionid, resourcegroup, workspacename, datasetname, trainingscriptname, clustername, modelname
1. Save the pipeline file.
1. Run the following command on the commandline to see your changes.
    ```
    git status
    ```
1. Commit your changes on the commandline.(see above git add, git commit, git push...)
1. Pull changes from the remote repo
    ```
    git pull
    ```
1. Push your changes (you might need your password again)
1. Whenever you push changes now you will automatically trigger the pipeline to start a new experiment run and to create a new version of the model. 

### 2. Package your model into a container
1. Add another stage in your azure-pipelines.yaml file.
    ```
    - stage: package
    displayName: PackageModel
    jobs:
      - job:
        displayName: Package Model
        steps:
        - task: DownloadPipelineArtifact@2
          displayName: Download Registration Metadata
          inputs:
            source: 'current'
            artifactName: modelinfo
            targetPath: $(Agent.TempDirectory)/
        - task: AzureCLI@2
          displayName: Package via Powershell
          inputs:
            azureSubscription: YOURSERVICECONNECTION
            scriptType: 'pscore'
            scriptLocation: 'inlineScript'
            inlineScript: |
              az extension add -n azure-cli-ml
              $path = "$(Agent.TempDirectory)/modelinfo.json"
              $modelinfo = Get-Content($path) | ConvertFrom-Json;
              $m = $modelinfo.modelname+":"+$modelinfo.modelversion
              echo $m 
              az ml model package --resource-group $(resourcegroup) --workspace-name $(workspacename) -m $m --in $modelinfo.modelname --il $modelinfo.modelversion
            addSpnToEnvironment: true
    ```
1. Modify YOURSERVICECONNECTION again to refer to your subscription.
1. Save the file. As a result a training should be triggered and the model will end up packaged in your Azure Container Registry.
### 3. Deploy a model to Azure Container Instances (ACI)
1. Add another step into your azure-pipelines file to deploy to ACI. You can do this directly in Azure DevOps.
  ```
  - stage: deploy_test
    displayName: Deploy Model To Test
    jobs:
    - deployment: Test
      environment: Test
      strategy:
        runOnce:
          deploy: 
            steps:
                - task: DownloadPipelineArtifact@2
                  displayName: Download Registration Metadata
                  inputs:
                    source: 'current'
                    artifactName: modelinfo
                    targetPath: $(Agent.TempDirectory)/
                
                - task: AzureCLI@2
                  displayName: Deploy to ACI via Powershell
                  inputs:
                    azureSubscription: 'dmxinternal'
                    scriptType: 'pscore'
                    scriptLocation: 'inlineScript'
                    inlineScript: |
                      $path = "$(Agent.TempDirectory)/modelinfo.json"
                      $modelinfo = Get-Content($path) | ConvertFrom-Json;
                      $containerimage = "YOURREGISTRY.azurecr.io/"+$modelinfo.modelname+":"+$modelinfo.modelversion
                      echo $containerimage
                      $aciname = $modelinfo.modelname+"-"+$modelinfo.modelversion
                      echo $aciname
                      $dnsname = "UNIQUEDNSNAME"+$modelinfo.modelname
                      #$acideletename = az container list --query "[?ipAddress.fqdn=='$dnsname'].{Name:name}" |convertfrom-json | ForEach-Object {$_.Name}
                      #if($acideletename){
                      #  echo "$acideletename will be deleted"
                      #  az container delete --resource-group $(resourcegroup) --name $acideletename --yes
                      #}
                      az container create --resource-group $(resourcegroup) --name $aciname --image $containerimage --cpu 2 --memory 2  --ports 5001 --registry-password $(registrypassword) --registry-username $(registryuser) --ip-address public 
                    
                      Write-Host "##vso[task.setvariable variable=containerimage;]$containerimage"
                    addSpnToEnvironment: true
  ```
1. Adjust some placeholders if required.
1. Add additional variables for registryuser and registrypassword of your container registry.
1. Save the file and push it if you didn't work in Azure DevOps.
1. Find the IP of your service and call your model which is now hosted on ACI using e.g. a Http Post using Postman.


### 4. Deploy a model to Azure Kubernetes Service for production 
1. Add another "Environment" in Azure DevOps Pipelines called "Prod". 
1. Add your AKS Cluster as a resource to the Environment.
1. Configure your environment to require a successful API check before deployment happens. This snippet might help you. Post to the url of the ACI you created previously.
```
{
"data":
[
[0, 0, 0, 0],
[1, 2, 3, 4],
[10, 9, 8, 37]
]
}
```

1. Configure your environment to require manual approval before a deployment happens.
1. Add a folder "deployment".
1. Add the file model-deployment.yaml found here https://raw.githubusercontent.com/DanielMeixner/MLOps/main/model_deployment.yaml into the folder deployment. Make sure you don't mess up the formatting.
1. Replace placeholders in this file.
    ```
    ...
    image: YOURREGISTRY/YOURIMAGE:YOURTAG
    ...
    ```
1. Add another step into your azure-pipelines file to deploy to AKS. You can do this directly in Azure DevOps.
    ```
    - stage: deploy_prod        
    displayName: Deploy Model to Prod     
    jobs:
     - deployment: Prod
       environment: Prod       
       strategy: 
        runOnce:
          deploy:
           steps:
            - checkout: self           
            - task: Bash@3
              inputs:
                targetType: 'inline'
                script: |
                  # Write your commands here
                  
                  echo 'Hello world'
                  echo $(containerimage)

            - task: DownloadPipelineArtifact@2
              displayName: DownloadModelinfo
              inputs:
                source: 'current'
                artifactName: modelinfo
                targetPath: $(Agent.TempDirectory)/
            
            - task: AzureCLI@2
              displayName: set variable
              inputs:   
                azureSubscription: 'dmxinternal'
                scriptType: 'pscore'
                scriptLocation: 'inlineScript'
                inlineScript: |
                  $path = "$(Agent.TempDirectory)/modelinfo.json"
                  $modelinfo = Get-Content($path) | ConvertFrom-Json;
                  $containerimage = "YOURREGISTRY/"+$modelinfo.modelname+":"+$modelinfo.modelversion
                  echo $containerimage
                  Write-Host "##vso[task.setvariable variable=containerimage;]$containerimage"
           
            - task: KubernetesManifest@0
              inputs:
                action: 'deploy'
                kubernetesServiceConnection: 'dmx-aks-inf-default'
                namespace: 'default'
                containers: $(containerimage)
                manifests: $(Build.SourcesDirectory)/deployment/model-deployment.yaml
    ```
1. Adjust some placeholders if required.
1. Save the file and push it.
1. Find the IP of your service and call your model which is now hosted on AKS using e.g. a Http Post using Postman.
1. To find your IP run 
    ```
    az aks get-credentials -n CLUSTERNAME -g RESOURCEGROUPNAME
    kubectl get services on cloud shell in Azure Portal
    ```
# Result
As are result you are able to 
- version all changes in your jupyter notebook
- trigger trainings and model creation in a structured manner based on code changes you consider as "good enough" to be shared 
- trigger trainings and model creation in a structured manner based when new data shall be used by referencing the version of the dataset
- deploy your models so they can be tested
- deploy your models after a succesfull test to the next environment (e.g. Production)


   



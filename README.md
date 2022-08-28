[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/) [![Lint](https://github.com/deep-diver/complete-mlops-system-workflow/actions/workflows/black.yml/badge.svg?branch=main)](https://github.com/deep-diver/complete-mlops-system-workflow/actions/workflows/black.yml) [![Validity Check for Training Pipeline](https://github.com/deep-diver/complete-mlops-system-workflow/actions/workflows/ci-training-pipeline.yml/badge.svg)](https://github.com/deep-diver/complete-mlops-system-workflow/actions/workflows/ci-training-pipeline.yml) [![Trigger Training Pipeline](https://github.com/deep-diver/complete-mlops-system-workflow/actions/workflows/cd-traing-pipeline.yml/badge.svg)](https://github.com/deep-diver/complete-mlops-system-workflow/actions/workflows/cd-traing-pipeline.yml)

# Complete MLOps System Workflow with GCP and TFX

This repository shows how to build a complete MLOps system with [TensorFlow eXtended(TFX)](https://www.tensorflow.org/tfx) and various GCP products such as [Vertex Pipeline](https://cloud.google.com/vertex-ai/docs/pipelines), [Vertex Training](https://cloud.google.com/vertex-ai/docs/training/custom-training), [Vertex Endpoint](https://cloud.google.com/vertex-ai/docs/predictions/deploy-model-api), [Google Cloud Storage](https://cloud.google.com/products/storage/). The main goal is to achieve the two common scenarios of **adapting to changes in codebase** and **adapting to changes in data** over time. To achieve these, we need three separate pipelines:

![](https://i.ibb.co/7SkTrzS/Screen-Shot-2022-05-22-at-2-16-47-AM.png)

- **CI/CD pipeline**
  - This pipeline is implemented in TFX, GitHub Action, and Vertex Pipeline.
  - GitHub Action basically detects to any changes occured in codebase. There are two branches of listening. 
  - The detection scope of the **first branch** and the **second branch** are the whole codebase and the only data preprocessing and modeling parts of the whole codebase respectively.
  - Both branches trigger different sub-workflows, but they have a lot in common.
    1. Clones the current codebase
    2. Unit tests the `*_test.py` files
    3. Create TFX pipeline
    4. Run the TFX pipeline in local
    5. Trigger TFX pipeline on Vertex Pipeline
  - The only difference between them is that the **first branch** has additional step to build a new docker image while the **second branch** has copying modules in the cloud location(GCS) in between step **d** and **e**.


```mermaid
  flowchart LR;
      subgraph GitHub Action
      direction LR      
      A[Changes in whole codebase]-->B[Trigger Cloud Build];
      C[Changes in modules]-->D[Trigger Cloud Build];           
      end
      
      subgraph GitHub Sub Action1
      direction LR 
      E[Clone Repo]-->F[Unit Test];
      F-->G[Create TFX Pipeline];
      G-->I[Build Docker Image];
      I-->J[Trigger Pipeline on Vertex];
      end
      
      subgraph GitHub Sub Action2
      direction LR
      K[Clone Repo]-->L[Unit Test];
      L-->M[Create TFX Pipeline];
      M-->O[Copy Modules];
      O-->P[Trigger Pipeline on Vertex];
      end      
      
      B-->E;
      D-->K;
```      

- **Model evaluation pipeline**
  - This pipeline is implemented in TFX, GitHub Action, and Vertex Pipeline.
  - GitHub Action periodically checks if there is enough data to evaluate currently deployed model on. The model is released in [GitHub Release](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository).
  - If there is enough data, it triggers another GitHub Action for **Model evaluation**, and it consists of the following:
    - Batch predictions.
    - Evaluate how the predicted result is good or worse by checking with the predefined accuracy threshold.
  - When the predicted result is not good enough, it will launch a Vertex Pipeline written in TFX:
    - SpanPreparator to prepare TFRecord of collected data and put it in different SPAN folder (here different SPAN means a model drift is detected)
    - PipelineTrigger to trigger ML pipeline, and it gives which SPAN to look up for.
  
```mermaid
  flowchart LR;
      subgraph Periodic Check-GitHub Action
      direction LR
      A[Check # of collected data]--Enough-->B[Trigger GitHub Action];
      end
      
      subgraph Model Evaluation-GitHub Action
      direction LR
      C[Batch Inference]--Not Good Enough-->D[Trigger MRP];
      end
      
      subgraph MRP - Model Retraining Pipeline
      direction LR
      E[SpanPreparator]-->F[PipelineTrigger];
      F-->G[Model Retraining];
      end
      
      B-->C;
      D-->E;
```

**👋 NOTE**: One could argue the whole component can be implemented without any cloud services. However, in my opinion, it is non trivial to achieve production ready quality of MLOps system without any help of cloud services. 

## Acknowledgements

I am thankful to the ML Developer Programs team at Google that provided GCP support.

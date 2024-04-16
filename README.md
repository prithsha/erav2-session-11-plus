# Folder structure

### main.py:

Have function to get

    - model instance
    - optimizers
    - schedulers
    - Class NetworkModelEvaluator, which have test and train method for network and also hold information like
            - train losses
            - test losses
            - wrongly predicted test and train images. Maintain them in a queue of size 20
            - Can display loss and accuracy charts


### utility folder:
    -   utils: Have basic utility functions like:
            - Find optimal learning rate by LR finder
            - time formatting 
    -   imageAugmentationUtility: Helps in perfuming custom image augmentation
    -   imageVisualizationUtility: Helps in visualizing images in matplotlib 

### models folder:
    - Contain Resnet 18 and 34 models 


### Jupyter execution file: assignment_11_gradcam_resnet18.ipynb

Hold the code execution sequence.

- Learning rate: 7.26E-02
- Model max test accuracy: 79.998%


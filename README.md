# Guidelines for the project:

## Research:
> We aim to optimize the adversarial properties of patch perturbations that will cause the VO system to deviate <br>
> the most from it's current trajectory.<br>
> More on the details of the subject and results of our research in <b>report.pdf</b>.

## Install and Run:
> * To install follow the guidelines in install_pytorch_cupy_3.txt. <br>
> * To run the attack: ./run.sh <br>
> Edit run.sh if needed to change arguments. Arg k is set to 2 for fast run, real computation will have k close to 100.


## Data:
> <b>data</b> folder consists of <b>VO_adv_project_train_dataset_8_frames</b> folder, that has 5 initial position folders. <br>
> Each initial position folder has 10 different trajectories that start from the same position. <br>
> It is posible to replace <b>VO_adv_project_train_dataset_8_frames</b> with <b>train</b> and <b>test</b> folders. <br>
> In order to assure high generalization properties on unseen data we might need to divide the dataset such that <br>
> train and test data differs from each other significantly. One way to do so is by division of initial positions. <br>
> 
> Generally data is used in different 3 places in the project: <br>
> * Train data in gradient ascent with train criterion. <br>
> * Evaluation data used after each train epoch for evaluation criterion. <br>
> * Test data used after whole training is done for test criterion. <br>
> 

## Model:
> The perturbations will be calculated in order to fool TartanVO model. To load the needed model we need to <br>
> run the script with <b>--model-name tartanvo_1914.pkl</b> flag, as in the run.sh. <br>

## Criterions:
> There are 3 types of criterions used in this project: optimization criterion, evaluation criterion and test criterion. <br>
> * Optimization (Train) criterion aka <b>args.att_criterion</b> is used in the training inside gradient ascent for updating the perturbation. <br>
> This criterion is used to calculate the gradient ascent and update the perturbation. <br>
> * Evaluation criterion: <b>att_eval_criterion</b> used in evaluation after each training epoch. <br>
> Used to select the best perturbation among those calculated with Optimization criterion. <br>
> Both previous criterions are up for us to define in order to maximize the last criterion which is the: <br>
> * Test criterion, is defined as a list <b>args.criterions</b> and based solely on the Translation deviations. <br>
> Calculates rms, prms of clean and adversarial translations as well as their delta and ratio.<br>
> This criterion needs to be eventually maximized and generalized for unseen data.


## Optimization scheme:
> Optimization scheme is responsible for data division between train, evaluation test tasks. <br>
> 
> Creation of data loaders is done inside <b>compute_data_args()</b> func of util.py.
> We create <b>trainDataloader</b> which will be used in <b>gradient_ascent_step()</b> func of attack object for training, <br>
> leaving <b>testDataloader</b> for <b>attack_eval()</b> func of attack obj for test evaluation. <br>
> <b>attack_eval()</b> is called inside <b>perturb()</b> after each train epoch has been completed. <br>
>

## Attack Optimizer:
> This project emphasized on modification of PGD optimizers in order to improve adverarial properties. <br>
> Modifications that were consider are such as Momentum, RSMPROP, Auto-PGD, Stochastic.

## Optimization and evaluation criteria:
> The <b>VOCriterion</b> in <b>loss.py</b> represents the base for any criterion.<br>
> It supports aggregating different losses such as: Translation, Rotation, Optical Flow and Target Translation.
> CLI accepts arguments needed for creating those loss functions. <br>
> 
> We might need to run some experiments to determine what criterions will be better suited for train/evalutaion. <br>
> But it seems like RMPS based loss will serve better for train criterion, while RMS based - for evalutaion criterion. <br>
> 
> As stated before we aim to maximize the Test criterion which is based only on the Translation, <br>
> but such loss is not sufficiently smooth, therefore we need to train and evaluate on a different <br>
> set of criterions which will be from one side smooth enough for gradietn calculation during training <br>
> and from the other side - provide insight into generalization properties during evaluation.<br>
> For that purpose we need to incorporate rotation and optical flow into the loss functions.

## Results:
> In report.pdf.
> 

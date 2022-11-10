# Guidelines for the project:
## Install and Run:
> * To install follow the guidelines in install_pytorch_cupy_3.txt under "this worked for me:" comment. <br>
> * To run the attack: ./run.sh <br>
> Edit run.sh if needed to change arguments. Arg k is set to 2 for fast run, real computation will have k close to 100.


## Data:
> <b>data</b> folder is placed inside the <b>code</b> folder since it seems like main script will search for input <br>
> inside the root folder. But it can be changed if needed.
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
> Right now, there is no difference between those 3 and all are based on the testDatloader which usses all available data. <br>
> There are 5 folders based on initial position of a trajectory. One way to split the data is 3:1:1 for Train:Eval:Test ratio. <br>
> Another way is to implement Cross Validation scheme again by separating data based on their initial position (folders). <br>

## Model:
> The perturbations will be calculated based to fool TartanVO model. To load the needed model we need to <br>
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
> Optimization scheme is responsible for data division between train and test tasks. <br>
> In order to implement this, <b>utils.py</b> needs to be edited. <br>
> Raw project files provide functionality for <b>--test-dir</b> CLI argument, <br>
> which is used for creating testDataloader that serves both for training and test evaluation. <br>
> 
> Creation of data loaders is done inside <b>compute_data_args()</b> func of util.py.
> We need to create <b>trainDataloader</b> which will be used in <b>gradient_ascent_step()</b> func of attack object for training, <br>
> leaving <b>testDataloader</b> for <b>attack_eval()</b> func of attack obj for test evaluation. <br>
> <b>attack_eval()</b> is called inside <b>perturb()</b> after each train epoch has been completed. <br>
> In order to achieve this we might need to add another CLI argument such as <b>--train-dir</b>. <br>
> Having both train-dir and test-dir argumetns might mean that we need to manually split data into separate folders. <br>
> But it's posible to use only <b>--test-dir</b> and implement separate Datasets which will be used for train and test loaders. <br>
> 
> Another option is to implement cross validation of 5 folds, choosing one initial position folder for evaluation, <br>
> while the rest initial position folders become train. <br>

## Attack Optimizer:
> Most likely we will need to subclass Attack or PGD class in order to create more sophisticated optimizer. <br>
> Attack is a virutal class and PGD is an implementation, it implements virtual <b>calc_sample_grad_split()</b>, <br>
> that provides the functionality of non disjoint windows that act as subtrajectories (according to the pdf notation). <br>
> Virtual <b>calc_sample_grad_single()</b> method handles gradient calculation for any trajectory or subtrajectory from a window. <br>
> Therefore most likely it will be easier to subclass PGD and reimplement <b>perturb()</b> method which is basically a classic <br>
> train() function. We will have to update <b>a_abs</b> variable that acts as a gradient step according to some epoch number condition. <br>

## Optimization and evaluation criteria:
> The provided <b>VOCriterion</b> in <b>loss.py</b> <b>apply()</b> method only uses translations for the loss calculation. <br>
> We need to implement <b>calc_rot_crit()</b> and <b>calc_flow_crit()</b> methods that currently return 0. <br>
> First one is a loss caused by rotation deviation, while the second one based on the optical flow deviation. <br>
> Another important thing is that train criterion is different from the test criterion. <br> 
> <b>att_eval_criterion</b> is used for test evaluation inside <b>attack_eval()</b> after each train epoch. <br>
> <b>att_criterion</b> is applied inside <b>gradient_ascent_step()</b>.
> CLI accepts arguments needed for creating those loss functions. <br>
> The creation of <b>att_eval_criterion</b> is happening inside <b>compute_attack_args()</b> method. <br>
> Which uses one of the preinstantiated versions of <b>VOCriterion</b> that were computed in <b>compute_VO_args()</b>. <br>
> The creation of <b>att_criterion</b> is inside <b>compute_VO_args()</b> and is also based on <b>VOCriterion</b> object. <br>
> 
> We might need to run some experiments to determine what criterions will be better suited for train/evalutaion. <br>
> But it seems like RMPS based loss will serve better for train criterion, while RMS based - for evalutaion criterion. <br>
> 
> As stated before we aim to maximize the Test criterion which is based only on the Translation, but such loss is not <br>
> sufficiently smooth, therefore we need to train and evaluate on a different set of criterions which will be from one side <br>
> smooth enough for gradietn calculation during training and from the other side - provide insight into generalization properties <br>
> during evaluation. For that purpose we need to incorporate rotation and optical flow into the loss functions.

## Modules that we are most likely to focus on:
> * <b>utils.py</b> - in case we need to edit run arguments as well as some basic run objects as dataloaders, criterion parameters, <br>
> atttack parameters, etc. <br>
> * <b>attacks/[attack, pgd, const].py</b> - edit optimization algorithm, update alpha scheduling if needed, gradient updates, <br>
> overall training and evaluation at each epoch logic. <br>
> * <b>loss.py</b> - implement loss criterions for rotations and optical flow. <br> 
> * <b>run_attacks.py</b> - main adversarial perturbation attack script, final testing is implemented here. <br>
> * Might need to write some scripts for experiments with graphs and stuff like that.

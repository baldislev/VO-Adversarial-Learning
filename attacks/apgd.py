import numpy as np
import torch
from attacks.attack import Attack
from Datasets.tartanTrajFlowDataset import extract_traj_data
import time
from tqdm import tqdm
import cv2
from . import PGD
from tensorboard_writer import writer, rundata


class APGD(Attack):
    def __init__(
            self,
            model,
            criterion,
            test_criterion,
            data_shape,
            norm='Linf',
            n_iter=20,
            n_restarts=1,
            alpha=None,
            rand_init=False,
            sample_window_size=None,
            sample_window_stride=None,
            pert_padding=(0, 0),
            stochastic=False,
            minibatch_size=1,
            init_pert_path=None,
            init_pert_transform=None):
        super(APGD, self).__init__(model, criterion, test_criterion, norm, data_shape,
                                   sample_window_size, sample_window_stride,
                                   pert_padding, stochastic, minibatch_size)

        self.alpha = alpha
        self.n_restarts = n_restarts
        self.n_iter = n_iter
        self.rand_init = rand_init

        self.counter = 0
        self.steps = n_iter
        self.steps_2, self.steps_min, self.size_decr = max(int(0.22 * self.steps), 1), \
                                                       max(int(0.06 * self.steps), 1), \
                                                       max(int(0.03 * self.steps), 1)
        self.k = self.steps_2 + 0
        self.u = np.arange(1)
        self.loss_steps = torch.zeros([self.steps, 1])
        self.thr_decr = 0.75
        self.reduced_last_check = None
        self.best_loss_sum = torch.zeros(1)
        self.train_loss_sum = torch.zeros(1)
        self.eval_loss_tot = torch.zeros(1)
        self.i = 0  # current iter num
        self.apgd_eps = 0.045  # 0.15  3 / 8 # 8 / 255
        self.step_size = None
        self.pert_old = None

        self.init_pert = None
        if init_pert_path is not None:
            self.init_pert = cv2.cvtColor(cv2.imread(init_pert_path), cv2.COLOR_BGR2RGB)
            if init_pert_transform is None:
                self.init_pert = torch.tensor(self.init_pert).unsqueeze(0)
            else:
                self.init_pert = init_pert_transform({'img': self.init_pert})['img'].unsqueeze(0)

    def calc_sample_grad_single(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                                scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        """
        Called by gradient_ascent_step for full trajectory or by calc_sample_grad_split for a subtrajectory.
        """
        pert = pert.detach()
        pert.requires_grad_()
        # prepares the perturbation and applies the model on it.
        img1_adv, img2_adv, output_adv = self.perturb_model_single(pert, img1_I0, img2_I0,
                                                                   intrinsic_I0,
                                                                   img1_delta, img2_delta,
                                                                   scale,
                                                                   mask1, mask2,
                                                                   perspective1,
                                                                   perspective2,
                                                                   device)
        # calculate the training loss which differs from the evaluation loss function:
        loss = self.criterion(output_adv, scale.to(device), y.to(device), target_pose.to(device), clean_flow.to(device))
        loss_sum = loss.sum(dim=0)
        self.train_loss_sum = torch.tensor(loss_sum).clone().detach()
        # if loss_sum > self.best_loss_sum.to(device):
        #     self.best_loss_sum = torch.tensor(loss_sum).clone().detach()
        writer.add_scalar("loss/loss_sum_over_train", loss_sum, self.i)
        # grad step
        grad = torch.autograd.grad(loss_sum, [pert])[0].detach()

        del img1_adv
        del img2_adv
        del output_adv
        del loss
        del loss_sum
        torch.cuda.empty_cache()

        return grad

    def calc_sample_grad_split(self, pert, img1_I0, img2_I0, intrinsic_I0, img1_delta, img2_delta,
                               scale, y, clean_flow, target_pose, perspective1, perspective2, mask1, mask2, device=None):
        """
        Gradient calculation using sliding windows technique with a normalization Dupliicity(l) factor.
        Called by gradient_ascent_step
        """
        sample_data_ind = list(range(img1_I0.shape[0] + 1))
        # stride is step, idx is w from the pdf notation:
        window_start_list = sample_data_ind[0::self.sample_window_stride]
        window_end_list = sample_data_ind[self.sample_window_size::self.sample_window_stride]

        if window_end_list[-1] != sample_data_ind[-1]:
            window_end_list.append(sample_data_ind[-1])
        grad = torch.zeros_like(pert, requires_grad=False)
        grad_multiplicity = torch.zeros(grad.shape[0], device=grad.device, dtype=grad.dtype)

        for window_idx, window_end in enumerate(window_end_list):
            window_start = window_start_list[window_idx]
            # Duplicity(l) update:
            grad_multiplicity[window_start:window_end] += 1
            # slicing input for the window:
            pert_window = pert[window_start:window_end].clone().detach()
            img1_I0_window = img1_I0[window_start:window_end].clone().detach()
            img2_I0_window = img2_I0[window_start:window_end].clone().detach()
            intrinsic_I0_window = intrinsic_I0[window_start:window_end].clone().detach()
            img1_delta_window = img1_delta[window_start:window_end].clone().detach()
            img2_delta_window = img2_delta[window_start:window_end].clone().detach()
            scale_window = scale[window_start:window_end].clone().detach()
            y_window = y[window_start:window_end].clone().detach()
            clean_flow_window = clean_flow[window_start:window_end].clone().detach()
            target_pose_window = target_pose.clone().detach()
            perspective1_window = perspective1[window_start:window_end].clone().detach()
            perspective2_window = perspective2[window_start:window_end].clone().detach()
            mask1_window = mask1[window_start:window_end].clone().detach()
            mask2_window = mask2[window_start:window_end].clone().detach()

            # Calculate gradients for the window:
            grad_window = self.calc_sample_grad_single(pert_window,
                                                       img1_I0_window,
                                                       img2_I0_window,
                                                       intrinsic_I0_window,
                                                       img1_delta_window,
                                                       img2_delta_window,
                                                       scale_window,
                                                       y_window,
                                                       clean_flow_window,
                                                       target_pose_window,
                                                       perspective1_window,
                                                       perspective2_window,
                                                       mask1_window,
                                                       mask2_window,
                                                       device=device)
            with torch.no_grad():
                # update global gradient with current window trajectory grad:
                grad[window_start:window_end] += grad_window

            del grad_window
            del pert_window
            del img1_I0_window
            del img2_I0_window
            del intrinsic_I0_window
            del scale_window
            del y_window
            del clean_flow_window
            del target_pose_window
            del perspective1_window
            del perspective2_window
            del mask1_window
            del mask2_window
            torch.cuda.empty_cache()

        grad_multiplicity_expand = grad_multiplicity.view(-1, 1, 1, 1).expand(grad.shape)
        # Normalize:
        grad = grad / grad_multiplicity_expand
        del grad_multiplicity
        del grad_multiplicity_expand
        torch.cuda.empty_cache()
        return grad.to(device)

    def perturb(self, data_loader, y_list, eps, targeted=False, device=None, eval_data_loader=None, eval_y_list=None):
        # step size for the attack: alpha as in pdf. targeted is False.
        a_abs = np.abs(eps / self.n_iter) if self.alpha is None else np.abs(self.alpha)
        if not self.step_size:
            self.step_size = self.apgd_eps * torch.ones([1, 1, 1, 1]).to(device).detach() * torch.Tensor([2.0]).to(device).detach().reshape([1, 1, 1, 1])

        multiplier = -1 if targeted else 1
        print("computing APGD attack with parameters:")
        print("attack random restarts: " + str(self.n_restarts))
        print("attack epochs: " + str(self.n_iter))
        print("attack norm: " + str(self.norm))
        print("attack epsilon norm limitation: " + str(eps))
        print("attack step size: " + str(a_abs))

        data_shape, dtype, eval_data_loader, eval_y_list, clean_flow_list, \
        eval_clean_loss_list, traj_clean_loss_mean_list, clean_loss_sum, \
        best_pert, best_loss_list, best_loss_sum, all_loss, all_best_loss = \
            self.compute_clean_baseline(data_loader, y_list, eval_data_loader, eval_y_list, device=device)

        print(f'best_loss_sum init {best_loss_sum}')
        for rest in tqdm(range(self.n_restarts)):
            print("restarting attack optimization, restart number: " + str(rest))
            opt_start_time = time.time()

            # apgd shit:
            t = 2 * torch.rand(best_pert.shape).to(device).detach() - 1
            pert = best_pert.detach() + self.apgd_eps * torch.ones([1, 1, 1, 1]).to(device).detach() * t / (
                t.reshape([t.shape[0], -1]).abs().max(dim=1, keepdim=True)[0].reshape([-1, 1, 1, 1]))

            pert = self.project(pert, eps)
            self.pert_old = pert.clone()
            # end apgd shit:

            if self.init_pert is not None:
                print(" perturbation initialized from provided image")
                pert = self.init_pert.to(best_pert)
            elif self.rand_init:
                print(" perturbation initialized randomly")
                pert = self.random_initialization(pert, eps)
            else:
                print(" perturbation initialized to zero")

            global rundata
            rundata.lossstep = 0

            # TRAIN EPOCHS LOOP:
            for k in tqdm(range(self.n_iter)):
                print(" attack optimization epoch: " + str(k))
                self.i = k
                iter_start_time = time.time()
                # TRAIN EPOCH HERE:
                rundata.run_now = True
                pert = self.gradient_ascent_step(pert, data_shape, data_loader, y_list, clean_flow_list, multiplier,
                                                 a_abs, eps, device)

                step_runtime = time.time() - iter_start_time
                print(" optimization epoch finished, epoch runtime: " + str(step_runtime))

                print(" evaluating perturbation")
                eval_start_time = time.time()
                writer.add_scalar("stepsize", self.step_size, k)
                rundata.run_now = False
                with torch.no_grad():
                    # EVALUATION:
                    eval_loss_tot, eval_loss_list = self.attack_eval(pert, data_shape, eval_data_loader, eval_y_list,
                                                                     device)
                    writer.add_scalar("best_loss_sum", best_loss_sum, k)
                    writer.add_scalar("eval_loss_tot", eval_loss_tot, k)
                    print(f'best_loss_sum {best_loss_sum} eval_loss_tot {eval_loss_tot}')
                    if eval_loss_tot > best_loss_sum:
                        best_pert = pert.clone().detach()
                        best_loss_list = eval_loss_list
                        best_loss_sum = eval_loss_tot
                    all_loss.append(eval_loss_list)
                    all_best_loss.append(best_loss_list)
                    traj_loss_mean_list = np.mean(eval_loss_list, axis=0)
                    traj_best_loss_mean_list = np.mean(best_loss_list, axis=0)

                    eval_runtime = time.time() - eval_start_time
                    print(" evaluation finished, evaluation runtime: " + str(eval_runtime))
                    print(" current trajectories loss mean list:")
                    print(" " + str(traj_loss_mean_list))
                    print(" current trajectories best loss mean list:")
                    print(" " + str(traj_best_loss_mean_list))
                    print(" trajectories clean loss mean list:")
                    print(" " + str(traj_clean_loss_mean_list))
                    print(" current trajectories loss sum:")
                    print(" " + str(eval_loss_tot))
                    print(" current trajectories best loss sum:")
                    print(" " + str(best_loss_sum))
                    print(" trajectories clean loss sum:")
                    print(" " + str(clean_loss_sum))

                    self.best_loss_sum = torch.tensor(best_loss_sum)
                    self.eval_loss_tot = torch.tensor(eval_loss_tot)

                    del eval_loss_tot
                    del eval_loss_list
                    torch.cuda.empty_cache()

            # apgd shit:
            self.counter += 1

            if self.reduced_last_check is None:
                self.reduced_last_check = np.zeros(1) == np.zeros(1)

            self.loss_steps[self.i] = self.eval_loss_tot  # self.train_loss_sum

            if self.counter == self.k:
                fl_oscillation = self.check_oscillation(self.loss_steps.detach().cpu().numpy(), self.i, self.k,
                                                        self.best_loss_sum.detach().cpu().numpy(), k3=self.thr_decr)
                fl_reduce_no_impr = (~self.reduced_last_check) * (
                        self.eval_loss_tot.cpu().numpy() >= self.best_loss_sum.cpu().numpy())
                # fl_reduce_no_impr = (~self.reduced_last_check) * (
                #         self.train_loss_sum.cpu().numpy() >= self.best_loss_sum.cpu().numpy())
                fl_oscillation = ~(~fl_oscillation * ~fl_reduce_no_impr)
                self.reduced_last_check = np.copy(fl_oscillation)

                if np.sum(fl_oscillation) > 0:
                    self.step_size[self.u[fl_oscillation]] /= 2.0

                self.counter = 0
                self.k = np.maximum(self.k - self.size_decr, self.steps_min)
            # end apgd shit.

            opt_runtime = time.time() - opt_start_time
            print("optimization restart finished, optimization runtime: " + str(opt_runtime))
        return best_pert.detach(), eval_clean_loss_list, all_loss, all_best_loss

    def gradient_ascent_perturb_step(self, pert, grad, a_abs, multiplier, eps):
        with torch.no_grad():
            pert = pert.detach()
            grad2 = pert - self.pert_old
            self.pert_old = pert.clone()

            a = 0.75 if self.i > 0 else 1.0

            grad = self.normalize_grad(grad)
            x_adv_1 = pert + self.step_size * torch.sign(grad)
            x_adv_1 = self.project(x_adv_1, eps)
            x_adv_1 = pert + (x_adv_1 - pert) * a + grad2 * (1 - a)
            x_adv_1 = self.project(x_adv_1, eps)

            pert = x_adv_1 + 0.

        return pert

    def check_oscillation(self, x, j, k, y5, k3=0.75):
        t = np.zeros(x.shape[1])
        for counter5 in range(k):
            t += x[j - counter5] > x[j - counter5 - 1]
        return t <= k * k3 * np.ones(t.shape)


from logging import getLogger
import torch
from VRPModel import VRPModel as Model
from VRPEnv import VRPEnv as Env
from utils import *
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

class VRPTester():
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # result folder, logger
        # self.logger = getLogger(name='trainer')
        # self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            # torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda:0')
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()
        self.time_estimator_2 = TimeEstimator()

    def run(self):
        self.time_estimator.reset()
        self.time_estimator_2.reset()


        self.env.load_raw_data(self.tester_params['test_episodes'], start_idx=self.tester_params['test_start_idx'])

        score_AM = AverageMeter()
        score_student_AM = AverageMeter()

        test_num_episode = self.tester_params['test_episodes']
        episode = 0
        problems_100 = []
        problems_100_200 = []
        problems_200_500 = []
        problems_500_1000 = []
        problems_1000 = []

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score_teacher, score_student, problems_size = self._test_one_batch(
                episode, batch_size, clock=self.time_estimator_2,logger = None)
            current_gap = (score_student - score_teacher) / score_teacher
            if problems_size < 100:
                problems_100.append(current_gap)
            elif 100 <= problems_size < 200:
                problems_100_200.append(current_gap)
            elif 200 <= problems_size < 500:
                problems_200_500.append(current_gap)
            elif 500 <= problems_size < 1000:
                problems_500_1000.append(current_gap)
            elif 1000 <= problems_size:
                problems_1000.append(current_gap)


            # print('problems_100 mean gap:', np.mean(problems_100), len(problems_100))
            # print('problems_100_200 mean gap:', np.mean(problems_100_200), len(problems_100_200))
            # print('problems_200_500 mean gap:', np.mean(problems_200_500), len(problems_200_500))
            # print('problems_500_1000 mean gap:', np.mean(problems_500_1000), len(problems_500_1000))
            # print('problems_1000 mean gap:', np.mean(problems_1000), len(problems_1000))

            score_AM.update(score_teacher, batch_size)
            score_student_AM.update(score_student, batch_size)

            episode += batch_size

            ############################
            # Logs
            ############################
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            # self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], Score_teacher:{:.4f}, Score_studetnt: {:.4f}".format(
            #     episode, test_num_episode, elapsed_time_str, remain_time_str, score_teacher, score_student))

            all_done = (episode == test_num_episode)

            if all_done:

                # self.logger.info(" *** Test Done *** ")
                # self.logger.info(" Teacher SCORE: {:.4f} ".format(score_AM.avg))
                # self.logger.info(" Student SCORE: {:.4f} ".format(score_student_AM.avg))
                gap_ = (score_student_AM.avg - score_AM.avg) / score_AM.avg * 100
                # self.logger.info(" Gap: {:.4f}%".format(gap_))

        return score_AM.avg, score_student_AM.avg, gap_

    def decide_whether_to_repair_solution(self,
                                          after_repair_sub_solution, before_reward, after_reward,
                                          first_node_index, length_of_subpath, double_solution):

        the_whole_problem_size = int(double_solution.shape[1] / 2)
        batch_size = len(double_solution)

        temp = torch.arange(double_solution.shape[1])

        x3 = temp >= first_node_index[:, None].long()
        x4 = temp < (first_node_index[:, None] + length_of_subpath).long()
        x5 = x3 * x4

        origin_sub_solution = double_solution[x5.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, length_of_subpath, 2)

        jjj, _ = torch.sort(origin_sub_solution[:, :, 0], dim=1, descending=False)

        index = torch.arange(batch_size)[:, None].repeat(1, jjj.shape[1])

        kkk_2 = jjj[index, after_repair_sub_solution[:, :, 0] - 1]

        after_repair_sub_solution[:, :, 0] = kkk_2

        if_repair = before_reward > after_reward

        need_to_repari_double_solution = double_solution[if_repair]
        need_to_repari_double_solution[x5[if_repair].unsqueeze(2).repeat(1, 1, 2)] = after_repair_sub_solution[if_repair].ravel()
        double_solution[if_repair] = need_to_repari_double_solution

        x6 = temp >= (first_node_index[:, None] + length_of_subpath - the_whole_problem_size).long()

        x7 = temp < (first_node_index[:, None] + length_of_subpath).long()

        x8 = x6 * x7

        after_repair_complete_solution = double_solution[x8.unsqueeze(2).repeat(1, 1, 2)].reshape(batch_size, the_whole_problem_size, -1)

        return after_repair_complete_solution

    def _test_one_batch(self, episode, batch_size, clock=None,logger = None):

        random_seed = 12
        torch.manual_seed(random_seed)

        ###############################################
        self.model.eval()

        with torch.no_grad():

            self.env.load_problems(episode, batch_size)

            reset_state, _, _ = self.env.reset(self.env_params['mode'])

            current_step = 0

            state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

            self.origin_problem = self.env.problems.clone().detach()


            self.optimal_length= self.env._get_travel_distance_2(self.origin_problem, self.env.solution)
            name = 'vrp'+str(self.env.solution.shape[1])
            B_V = batch_size * 1

            while not done:

                loss_node, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                    self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                               raw_data_capacity=self.env.raw_data_capacity)  # 更新被选择的点和概率

                if current_step == 0:
                    selected_flag_teacher = torch.ones(B_V, dtype=torch.int)
                    selected_flag_student = selected_flag_teacher
                current_step += 1

                state, reward, reward_student, done = \
                    self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)

            # print('Get first complete solution!')

            # 1. The complete solution is obtained
            best_select_node_list = torch.cat((self.env.selected_student_list.reshape(batch_size, -1, 1),
                                               self.env.selected_student_flag.reshape(batch_size, -1, 1)), dim=2)

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

            escape_time, _ = clock.get_est_string(1, 1)

            # self.logger.info("Greedy, name:{}, gap:{:5f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(name,
                # ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100, escape_time,
            # current_best_length.mean().item(), self.optimal_length.mean().item()))


            ####################################################

            budget = self.env_params['RRC_budget']

            for bbbb in range(budget):
                torch.cuda.empty_cache()



                self.env.load_problems(episode, batch_size)

                # 2. Sample the partial solution

                best_select_node_list = self.env.vrp_whole_and_solution_subrandom_inverse(best_select_node_list)

                partial_solution_length, first_node_index, length_of_subpath, double_solution = \
                    self.env.destroy_solution(self.env.problems, best_select_node_list)

                before_repair_sub_solution = self.env.solution

                before_reward = partial_solution_length

                current_step = 0

                reset_state, _, _ = self.env.reset(self.env_params['mode'])

                state, reward, reward_student, done = self.env.pre_step()  # state: data, first_node = current_node

                # 3. Reconstruct the partial solution.

                while not done:
                    if current_step == 0:
                        selected_teacher = self.env.solution[:, 0, 0]
                        selected_flag_teacher = self.env.solution[:, 0, 1]
                        selected_student = selected_teacher
                        selected_flag_student = selected_flag_teacher


                    else:
                        _, selected_teacher, selected_student, selected_flag_teacher, selected_flag_student = \
                            self.model(state, self.env.selected_node_list, self.env.solution, current_step,
                                       raw_data_capacity=self.env.raw_data_capacity)

                    current_step += 1

                    state, reward, reward_student, done = \
                        self.env.step(selected_teacher, selected_student, selected_flag_teacher, selected_flag_student)

                ahter_repair_sub_solution = torch.cat((self.env.selected_student_list.unsqueeze(2),
                                                       self.env.selected_student_flag.unsqueeze(2)), dim=2)

                after_reward = - reward_student

                after_repair_complete_solution = self.decide_whether_to_repair_solution(
                    ahter_repair_sub_solution,
                    before_reward, after_reward, first_node_index, length_of_subpath, double_solution)

                best_select_node_list = after_repair_complete_solution

                current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)

                escape_time, _ = clock.get_est_string(1, 1)

                # self.logger.info(
                #     "RRC step{}, name:{}, gap:{:6f} %, Elapsed[{}], stu_l:{:5f} , opt_l:{:5f}".format(
                #          bbbb, name, ((current_best_length.mean() - self.optimal_length.mean()) / self.optimal_length.mean()).item() * 100,
                #         escape_time,current_best_length.mean().item(), self.optimal_length.mean().item()))

            current_best_length = self.env._get_travel_distance_2(self.origin_problem, best_select_node_list)
            # print(f'current_best_length', (current_best_length.mean() - self.optimal_length.mean())
            #       / self.optimal_length.mean() * 100, '%', 'escape time:', escape_time,
            #       f'optimal:{self.optimal_length.mean()}, current_best:{current_best_length.mean()}')

            # 4. Cycle until the budget is consumed.
            # self.env.valida_solution_legal(self.origin_problem, best_select_node_list)

            return self.optimal_length.mean().item(), current_best_length.mean().item(), self.env.problem_size

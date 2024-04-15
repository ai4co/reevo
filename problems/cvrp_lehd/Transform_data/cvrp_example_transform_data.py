
import numpy as np
import os

def OneRowSolution_to_TwoRow(ori_solution):
    solution = ori_solution

    node = []
    flag = []
    for i in range(1, len(solution)):
        if solution[i] != 0:
            node.append(solution[i])
        if solution[i] != 0 and solution[i - 1] == 0:
            flag.append(1)
        if solution[i] != 0 and solution[i - 1] != 0:
            flag.append(0)
    node_flag = node + flag
    return node_flag

def generate_nazari_vrp_data(dataset_size, vrp_size):
    CAPACITIES = {
        10: 20.,
        20: 30.,
        50: 40.,
        100: 50.,
        200: 80.,
        500: 100.,
        1000: 250.,
    }
    return list(zip(
        np.random.uniform(size=(dataset_size, 2)).tolist(),  # Depot location
        np.random.uniform(size=(dataset_size, vrp_size, 2)).tolist(),  # Node locations
        np.random.randint(1, 10, size=(dataset_size, vrp_size)).tolist(),  # Demand, uniform integer 1 ... 9
        np.full(dataset_size, CAPACITIES[vrp_size]).tolist()  # Capacity, same for whole dataset
    ))


def generate_vrp_data(dataset_size, vrp_size, seed):
    np.random.seed(seed)
    return generate_nazari_vrp_data(dataset_size, vrp_size)


def save_dataset(one_row_data_all,savepath):
    np.savetxt(savepath,one_row_data_all,delimiter=',',fmt='%s')
    return

if __name__ == '__main__':

    dataset_size = 10

    vrp_size = 10

    np.random.seed(10)

    dataset = generate_nazari_vrp_data(dataset_size, vrp_size)

    one_row_data_all = []



    for kkk in range(len(dataset)):

        print('No ',kkk,'/',dataset_size)

        data_instance = dataset[kkk]

        depot, Customer, demand, capacity = data_instance

        depot = np.array(depot).ravel().tolist()

        Customer = np.array(Customer).ravel().tolist()

        demand = np.array(demand).ravel().tolist()

        # a randomly given cost
        curr_cost = 3.1
        # a randomly given route
        routes = [0,1,2,3,4,0,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,0]

        curr_node_flag = OneRowSolution_to_TwoRow(routes)

        one_row_data = ['depot'] + depot + ['customer'] + Customer + ['capacity'] + [capacity] +\
                       ['demand'] + demand + ['cost'] + [curr_cost] + ['node_flag'] + curr_node_flag

        one_row_data_all.append(one_row_data)
    b = os.path.abspath('.').replace('\\', '/')
    save_dataset(one_row_data_all, b + f'/vrp{vrp_size}_test_n{dataset_size}_example.txt')

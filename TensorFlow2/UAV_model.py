import argparse
import math
import plotly.graph_objects as go
import numpy as np
from plot import main


class Problem:
    def __init__(self):
        self.nodeNum = 0        # 节点个数
        self.customerNum = 0    # 顾客个数
        self.serviceTime = []   # 服务时间
        self.demand = []        # 顾客需求
        self.locations = []     # 顾客位置
        self.distance_matrix = []   # 距离矩阵
        self.vehicle = 0            # 车辆个数
        self.capacities = 10        # 车辆容量
        self.uav_selfweight = 1     # 无人机重量
        self.uav_capacities = 3     # 无人机容量 0.3
        self.uav_batteryPower = 50  # 无人机电池能量
        self.uav_k = 48             # 无人机能量消耗的公式系数370nr(P - e)
        self.uav_alpha = 0.75       # 无人机的升力比
        self.num_solutions = 0      # 结果数量

    def get_demand(self, index):
        return self.demand[index]

    def get_num_customers(self):
        return len(self.locations) - 1

    def get_distance(self, from_index, to_index):
        return self.distance_matrix[from_index][to_index]


locations, demands, list_of_paths = main()

# def locations():
#     return [[0.65647066, 0.30283117],
#             [0.5896659, 0.40343487],
#             [0.913059,  0.73775387],
#             [0.09103227, 0.97882354],
#             [0.9028566, 0.26675403],
#             [0.09803104, 0.5470532],
#             [0.34823918, 0.46237862],
#             [0.41163588, 0.15933073],
#             [0.7047081, 0.07520616],
#             [0.46257508, 0.4484656],
#             [0.83002925, 0.37415648],
#             [0.2735077, 0.4408232],
#             [0.9192778, 0.44249153],
#             [0.40118062, 0.6060854],
#             [0.49603117, 0.06208503],
#             [0.5863309, 0.945271],
#             [0.33414543, 0.8073325],
#             [0.86212575, 0.16002011],
#             [0.7408304, 0.01328635],
#             [0.3841753, 0.629447],
#             [0.8540609, 0.2603773]]


# def demand():
#     return [0.0, 0.20000002, 0.10000001, 0.06666667, 0.23333335, 0.23333335, 0.03333334,
#               0.20000002, 0.10000001, 0.26666668, 0.06666667, 0.20000002, 0.06666667,
#               0.10000001, 0.03333334, 0.06666667, 0.13333334, 0.23333335, 0.20000002,
#               0.26666668, 0.10000001]


def generate_Problem():
    problem = Problem()
    problem.locations = locations
    problem.demand = demands
    print('用户需求：', problem.demand)
    print('用户坐标：', problem.locations)

    for from_index in range(len(problem.locations)):
        distance_vector = []
        for to_index in range(len(problem.locations)):
            distance_vector.append(calculate_distance(problem.locations[from_index], problem.locations[to_index]))
        problem.distance_matrix.append(distance_vector)
    # print("距离矩阵：", problem.distance_matrix)

    return problem


def calculate_distance(point0, point1):
    """
    计算两点之间的距离
    """
    dx = point1[0] - point0[0]
    dy = point1[1] - point0[1]
    temp = math.sqrt(dx * dx + dy * dy)
    return float("{0:.2f}".format(temp))


def energy_consumption(problem, from_index, mid_index, to_index):
    initial_energy = problem.uav_batteryPower
    F_ij = 0
    temp = (problem.uav_selfweight + problem.demand[mid_index]) # 取包裹的重量
    F_ij += (temp * problem.distance_matrix[from_index][mid_index] + problem.uav_selfweight * problem.distance_matrix[mid_index][to_index] ) \
            * problem.uav_batteryPower / problem.uav_k

    final_energy = initial_energy - F_ij
    dis = problem.distance_matrix[from_index][mid_index] + problem.distance_matrix[mid_index][to_index]
    return final_energy, F_ij, dis


def charge_calculate(consumption_energy, from_index, to_index, initial_dis):
    charge_after_power = problem.uav_alpha * consumption_energy * (problem.distance_matrix[from_index][to_index] / initial_dis)
    return charge_after_power


def calculate_path_battery(problem, path):
    n_path = path.index(0, 1, len(path))

    first = 0
    second = n_path + 1
    battery = [problem.uav_batteryPower] * (n_path + 1)
    bat = problem.uav_batteryPower

    for first in range(first, n_path + 1):
        battery[first] = bat

        if ((second < len(path) and path[first] != path[second]) or second >= len(path)) and bat < problem.uav_batteryPower:
            charge = charge_calculate(consume, first, first + 1, dis)
            bat += charge
            if bat > problem.uav_batteryPower:
                bat = problem.uav_batteryPower

        if second < len(path) and path[first] == path[second]:
            _, consume, dis = energy_consumption(problem, path[second], path[second + 1], path[second + 2])
            bat -= consume
            second += 3
    print(battery)
    return battery


# list_of_path = [[0, 10, 12, 2, 15, 3, 5, 11, 6, 0],
#                 [0, 20, 4, 17, 18, 8, 0],
#                 [0, 14, 7, 9, 1, 0],
#                 [0, 13, 19, 16, 0]]


def construct_solution(problem):
    solution = []
    route_uav = []
    route_truck = []

    # customer_indices = list(range(n + 1))
    for index in range(len(list_of_paths)):
        customer_indices = list_of_paths[index]
        n = len(customer_indices) - 2

        start_customer_index = 1
        i = start_customer_index    # 1

        trip = [0]
        to_truck_indices = [0]
        to_uav_indices = []
        capacity_left = problem.capacities
        capacity_uav_left = problem.uav_capacities
        battery_uav = problem.uav_batteryPower

        while i <= n:
            trip.append(customer_indices[i])

            to_truck_indices.append(trip[-1])
            print("customer_indices[{0}]".format(customer_indices[i]), problem.get_demand(customer_indices[i]))
            capacity_left -= problem.get_demand(customer_indices[i])

            if len(trip) > 3:
                if len(to_uav_indices) == 0 or to_uav_indices[-1] != trip[-2] and to_uav_indices[-1] != trip[-3]:
                # if len(to_uav_indices) == 0 or to_uav_indices[-1] != trip[-2]:
                    final_energy, F_ij, dis = energy_consumption(problem, trip[-3], trip[-2], trip[-1])
                    capacity_uav_left -= problem.demand[trip[-2]]
                    if final_energy >= 0 and capacity_uav_left >= 0 and dis <= 0.8:
                        to_uav_indices.append(trip[-3])
                        # if len(to_uav_indices) > 3 and to_uav_indices[-1] == to_uav_indices[-3]:
                        #     to_uav_indices.remove(to_uav_indices[-3])
                        to_uav_indices.append(trip[-2])
                        # if len(to_uav_indices) > 3 and to_uav_indices[-1] == to_uav_indices[-3]:
                        #     to_uav_indices.remove(to_uav_indices[-3])
                        to_uav_indices.append(customer_indices[i])
                        to_truck_indices.remove(trip[-2])
                        capacity_left += problem.demand[trip[-2]]
            i += 1

        if len(trip) > 1:
            trip.append(0)
            solution.append(trip)
            route_truck.append(to_truck_indices)
            route_uav.append(to_uav_indices)

        route_truck[len(route_truck) - 1].append(0)

        solution1 = []
        for index in range(len(route_truck)):
            temp = []
            temp.extend(route_truck[index])
            temp.extend(route_uav[index])
            solution1.append(temp)

    # pltShow(problem.locations, route_truck, route_uav)
    print('solution:', solution)
    print('route_truck:', route_truck)
    print('route_uav:', route_uav)
    print('solution1:', solution1)
    return solution, route_truck, route_uav, solution1


def trans_dis_route_uav_dis(route_uav):
    route_uav_dis = []
    for i, item in enumerate(route_uav):
        length = len(item) / 3
        if length == 1:
            route_uav_dis.append(item)
            continue
        for index in range(int(length)):
            route_uav_dis.append(route_uav[i][3 * index: 3 * (index + 1)])
    # print(route_uav_dis)
    return route_uav_dis


def plot_route(truck, uav, xy, demands,  title):
    """Plots journey of agent
    Args:
        data: dataset of graphs
        pi: (batch, decode_step) # tour
        idx_in_batch: index of graph in data to be plotted
    """

    customer_xy = np.array(xy[1:])
    depot_xy = np.array(xy[0])
    demands = np.array(demands)
    xy = np.array(xy)
    customer_labels = ['(' + str(i) + ', ' + str(demand) + ')' for i, demand in enumerate(demands.round(2), 1)]

    truck_traces = []
    for i, path in enumerate(truck, 1):
        coords = xy[[int(x) for x in path]]

        # Calculate length of each agent loop
        lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)

        truck_traces.append(go.Scatter(x=coords[:, 0],
                                       y=coords[:, 1],
                                       mode='markers + lines',
                                       # line=dict(dash="dash"), # 设置线条虚线样式
                                       name=f'tour{i} Length = {total_length:.3f}',
                                       opacity=1.0))

    uav_traces = []

    for i, path in enumerate(uav, 1):
        coords = xy[[int(x) for x in path]]
        # print(coords)
        # Calculate length of each agent loop
        lengths = np.sqrt(np.sum(np.diff(coords, axis=0) ** 2, axis=1))
        total_length = np.sum(lengths)

        uav_traces.append(go.Scatter(x=coords[:, 0],
                                     y=coords[:, 1],
                                     mode='lines',
                                     line=dict(dash="dash"), # 设置线条虚线样式
                                     name=f'tour{i} Length = {total_length:.3f}',
                                     opacity=1.0))

    trace_points = go.Scatter(x=customer_xy[:, 0],
                              y=customer_xy[:, 1],
                              mode='markers+text',
                              name='Customer (demand)',
                              text=customer_labels,
                              textposition='top center',
                              marker=dict(size=7),
                              opacity=1.0
                              )

    trace_depo = go.Scatter(x=[depot_xy[0]],
                            y=[depot_xy[1]],
                            mode='markers+text',
                            name='Depot (capacity = 1.0)',
                            text=['1.0'],
                            textposition='bottom center',
                            marker=dict(size=23),
                            marker_symbol='triangle-up'
                            )

    layout = go.Layout(title=dict(text=f'<b>VRP{customer_xy.shape[0]} {title}</b>',
                       x=0.5, y=1, yanchor='bottom', yref='paper', pad=dict(b=10)),
                       # https://community.plotly.com/t/specify-title-position/13439/3
                       # xaxis = dict(title = 'X', ticks='outside'),
                       # yaxis = dict(title = 'Y', ticks='outside'),
                       # https://kamino.hatenablog.com/entry/plotly_for_report
                       xaxis=dict(title='X', range=[0, 1], showgrid=False, ticks='outside', linewidth=1, mirror=True),
                       yaxis=dict(title='Y', range=[0, 1], showgrid=False, ticks='outside', linewidth=1, mirror=True),
                       showlegend=True,
                       width=1000,
                       height=800,
                       autosize=True,
                       template="plotly_white",
                       legend=dict(x=1, xanchor='right', y=0, yanchor='bottom', bordercolor='#444', borderwidth=0)
                       # legend=dict(x=0, xanchor='left', y=0, yanchor='bottom', bordercolor='#444', borderwidth=0)
                       )

    data = [trace_points, trace_depo] + truck_traces + uav_traces
    # print('path: ', pi_)
    # print("list_of_path:", list_of_paths)
    fig = go.Figure(data=data, layout=layout)
    fig.show()


# truck = [[0, 10, 2, 15, 5, 11, 6, 0], [0, 20, 17, 18, 8, 0], [0, 14, 9, 1, 0], [0, 13, 16, 0]]
# uav = [[10, 12, 2, 15, 3, 5], [20, 4, 17], [14, 7, 9], [13, 19, 16]]
problem = generate_Problem()
solution, route_truck, route_uav, solution1 = construct_solution(problem)
# demands = demand()
route_uav_dis = trans_dis_route_uav_dis(route_uav)
plot_route(route_truck, route_uav_dis, locations, demands, "Test")

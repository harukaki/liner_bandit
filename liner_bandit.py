from cmath import sqrt
from logging import exception
from numpy.random import  normal, multivariate_normal
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import random



class liner_bandit_env:
    def __init__(self, algorithm, T, loop, graph_path):
        self.algorithm = algorithm
        self.T = T
        self.loop = loop
        self.graph_path = graph_path
        self.theta_star = np.array([3, 1])
        self.action_star = np.array([1, 0])
        self.A = np.eye(2)
        self.b = np.zeros(2)
        self.alpha = 1
        self.sigma_0 = 10
        self.sigma = 1
    
    def action(self, i):
        return np.array([np.cos(i * np.pi/4), np.sin(i * np.pi / 4)])

    def reset(self):
        self.theta_star = np.array([3, 1])
        self.action_star = np.array([1, 0])
        self.A = np.eye(2) * self.sigma_0 / self.sigma
        self.b = np.zeros(2)
        self.alpha = 1
        self.sigma_0 = 1
        self.sigma = 1

    def LinUCB(self, t):
        score = []
        inv_A = np.linalg.inv(self.A)
        theta = np.dot(inv_A, self.b)
   
        alpha = self.alpha * sqrt(2 * np.log10(t))
       
        for i in range(8):
            action = self.action(i)
            score.append(np.inner(action.T, theta) + alpha * self.sigma * sqrt(np.inner(action.T, np.dot(inv_A, action))))
        max_score = max(score)
        candidate = []
        for i in range(8):
            if score[i] == max_score:
                candidate.append(i)
        max_arm_index = random.choice(candidate)
        selected_action = self.action(max_arm_index)
        reward = normal(np.inner(selected_action.T, self.theta_star), self.sigma_0)
        self.b = self.b + selected_action * reward
        self.A += np.outer(selected_action, selected_action.T)

        return max_arm_index
    
    def Thompson_sampling(self, t):
        theta = multivariate_normal(np.dot(np.linalg.inv(self.A), self.b), self.sigma_0 * np.linalg.inv(self.A))
        gauss_arm_score = [normal(np.inner(theta.T, self.action(i)), self.sigma_0) for i in range(8)]
        max_arm_index = gauss_arm_score.index(max(gauss_arm_score))
        selected_action = self.action(max_arm_index)
        reward = normal(np.inner(selected_action.T, self.theta_star), self.sigma_0)
        self.b = self.b + selected_action * reward
        self.A += np.outer(selected_action, selected_action.T)
        return max_arm_index

    def greedy(self, t):
        score = []
        theta = np.dot(np.linalg.inv(self.A), self.b)
        #print(theta)
        alpha = self.alpha * sqrt(2 * np.log(t))
        for i in range(8):
            action = self.action(i)
            #print(action)
            score.append(np.inner(action.T, theta))
        max_score = max(score)
        candidate = []
        for i in range(8):
            if score[i] == max_score:
                candidate.append(i)
        max_arm_index = random.choice(candidate)
        selected_action = self.action(max_arm_index)
        reward = normal(np.inner(selected_action.T, self.theta_star), self.sigma_0)
        self.b = self.b + selected_action * reward
        self.A += np.outer(selected_action, selected_action.T)
        return max_arm_index

    def select_action(self, t):
        if self.algorithm == "LinUCB":
            max_arm_index = self.LinUCB(t)
        elif self.algorithm == "Thompson Sampling":
            max_arm_index = self.Thompson_sampling(t)
        elif self.algorithm == "greedy":
            max_arm_index = self.greedy(t)
        else:
            raise Exception
        return max_arm_index
    
    def regret(self, i):
        return 3 - np.inner(self.theta_star, self.action(i))
    

    def regret_graph(self, regret_list, path):
        mean_regret = np.empty(self.T)
        std_err = np.empty(self.T)
        for i in range(self.T):
            mean_regret[i] = np.average(regret_list[i])
            std_err[i] = np.std(regret_list[i], ddof=1) / sqrt(len(regret_list[i]))
        x = np.linspace(1, self.T, self.T)
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.set_xscale("log")
        ax.set_xlabel("Time")
        ax.set_ylabel("Regret")
        ax = plt.plot(x, mean_regret)
        plt.fill_between(x, mean_regret - std_err, mean_regret + std_err, alpha=0.15)
        fig.savefig(path)
        return mean_regret, std_err



    def main(self):
        print(f"start {self.algorithm}")
        regret_list = [[] for i in range(1, self.T + 1)]
        for i in range(self.loop):
            print(f"loop {i}")
            self.reset()
            regret_bector = []
            data = pd.DataFrame(data={"Time": np.log10(range(1, self.T + 1))})
            
            regret = 0
            for t in range(1, self.T + 1):
                selected_action = self.select_action(t)
                regret += self.regret(selected_action)
                if regret < 0:
                    raise exception
                regret_bector.append(regret)
                regret_list[t-1].append(regret)
            data["Regret"] = regret_bector
            print(np.dot(np.linalg.inv(self.A), self.b))

        mean_regret, std_err = self.regret_graph(regret_list, self.algorithm + ".png")
        return mean_regret, std_err


if __name__ == "__main__":
    T = 10000
    loop = 100
    
    LinUCB_env = liner_bandit_env(
                                    T=T,
                                    algorithm="LinUCB",
                                    loop=loop,
                                    graph_path="LinUCB.png")
    greedy_env = liner_bandit_env(
                                    T=T,
                                    algorithm="greedy",
                                    loop=loop,
                                    graph_path="greedy.png")
    
    thompson_env = liner_bandit_env(
                                    T=T,
                                    algorithm="Thompson Sampling",
                                    loop=loop,
                                    graph_path="Thompson_Sampling.png")
    ucb_mean, ucb_std = LinUCB_env.main()
    greedy_mean, greedy_std = greedy_env.main()
    ts_mean, ts_std = thompson_env.main()
    fig = plt.figure()
    x = np.linspace(1, T, T)
    ax = plt.subplot(111)
    ax.set_xscale("log")
    ax.set_xlabel("Time")
    ax.set_ylabel("Regret")
    ax = plt.plot(x, ucb_mean, label="LinUCB")
    plt.fill_between(x, ucb_mean - ucb_std, ucb_mean + ucb_std, alpha=0.15)
    ax = plt.plot(x, greedy_mean, label="Greedy")
    plt.fill_between(x, greedy_mean - greedy_std, greedy_mean + greedy_std, alpha=0.15)
    ax = plt.plot(x, ts_mean, label="Thompson Sampling")
    plt.fill_between(x, ts_mean - ts_std, ts_mean + ts_std, alpha=0.15)
    plt.legend()
    fig.savefig("graph.png")
    
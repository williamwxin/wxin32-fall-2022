# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import sys
sys.path.append("./ABAGAIL.jar")
import numpy as np
import time
import copy
import seaborn as sns
import matplotlib.pyplot as plt

import mlrose_hiive as mlrose
from numpy import loadtxt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


def random_hill_climb2(problem, max_attempts=10, max_iters=np.inf, restarts=0,
                      init_state=None, curve=False, random_state=None,
                      state_fitness_callback=None, callback_user_info=None):

    if (not isinstance(max_attempts, int) and not max_attempts.is_integer()) \
            or (max_attempts < 0):
        raise Exception("""max_attempts must be a positive integer.""")

    if (not isinstance(max_iters, int) and max_iters != np.inf
        and not max_iters.is_integer()) or (max_iters < 0):
        raise Exception("""max_iters must be a positive integer.""")

    if (not isinstance(restarts, int) and not restarts.is_integer()) \
            or (restarts < 0):
        raise Exception("""restarts must be a positive integer.""")

    if init_state is not None and len(init_state) != problem.get_length():
        raise Exception("""init_state must have same length as problem.""")

    # Set random seed
    if isinstance(random_state, int) and random_state > 0:
        np.random.seed(random_state)

    best_fitness = -np.inf
    best_state = None

    # best_fitness_curve = []
    fitness_curve_hist = []
    all_curves = []

    continue_iterating = True
    # problem.reset()
    iters = 0
    for current_restart in range(restarts + 1):
        # Initialize optimization problem and attempts counter
        fevals = problem.fitness_evaluations
        if init_state is None:
            problem.reset()
        else:
            problem.set_state(init_state)
        problem.fitness_evaluations = fevals

        callback_extra_data = None
        if state_fitness_callback is not None:
            callback_extra_data = callback_user_info + [('current_restart', current_restart)]
            # initial call with base data
            state_fitness_callback(iteration=0,
                                   state=problem.get_state(),
                                   fitness=problem.get_adjusted_fitness(),
                                   fitness_evaluations=problem.fitness_evaluations,
                                   user_data=callback_extra_data)

        attempts = 0
        while (attempts < max_attempts) and (iters < max_iters):
            iters += 1
            problem.current_iteration += 1

            # Find random neighbor and evaluate fitness
            next_state = problem.random_neighbor()
            next_fitness = problem.eval_fitness(next_state)

            # If best neighbor is an improvement,
            # move to that state and reset attempts counter
            current_fitness = problem.get_fitness()
            if next_fitness > current_fitness:
                problem.set_state(next_state)
                attempts = 0
            else:
                attempts += 1

            if curve:
                if current_fitness > best_fitness:
                    fitness_curve_hist.append((-1*current_fitness,problem.fitness_evaluations))
                    best_fitness = current_fitness
                else:
                    fitness_curve_hist.append((-1 * best_fitness, problem.fitness_evaluations))
                adjusted_fitness = problem.get_adjusted_fitness()

            # invoke callback
            if state_fitness_callback is not None:
                max_attempts_reached = (attempts == max_attempts) or (iters == max_iters) or problem.can_stop()
                continue_iterating = state_fitness_callback(iteration=iters,
                                                            attempt=attempts + 1,
                                                            done=max_attempts_reached,
                                                            state=problem.get_state(),
                                                            fitness=problem.get_adjusted_fitness(),
                                                            fitness_evaluations=problem.fitness_evaluations,
                                                            curve=np.asarray(all_curves) if curve else None,
                                                            user_data=callback_extra_data)
                # break out if requested
                if not continue_iterating:
                    break

        # Update best state and best fitness
        # current_fitness = problem.get_fitness()
        # if current_fitness > best_fitness:
        #     best_fitness = current_fitness
        #     best_state = problem.get_state()
        #     if curve:
        #         if current_restart == 0:
        #             fitness_curve_hist = [*fitness_curve]
        #         else:
        #             fitness_curve_hist.append(curve_value)
        # else:
        #     if curve:
        #         temp = (-1*best_fitness, curve_value[1])
        #         fitness_curve_hist.append(temp)
        # break out if we can stop
        if problem.can_stop():
            break
    best_fitness *= problem.get_maximize()

    return best_state, best_fitness, np.asarray(fitness_curve_hist) if curve else None


#Continuous peaks
if True:
    fitness = mlrose.ContinuousPeaks()
    problem = mlrose.DiscreteOpt(length=100, fitness_fn=fitness, maximize=True, max_val=2)
    #SA best best
    if True:
        # RHC
        cpk_iter = np.zeros((100000,4))
        for i in range(10):
            problem.reset()
            max_iters = 100000
            restarts = 100
            max_attempts = 1000
            start_time = time.time()
            cpk_best_state_rhc, cpk_best_fitness_rhc, cpk_curve_rhc = random_hill_climb2(problem,max_iters=max_iters,max_attempts = max_attempts,restarts=restarts,curve=True)
            cpk_time_rhc = time.time() - start_time
            cpk_iter[:,0] += cpk_curve_rhc[:,0]/10

        # SA
        for i in range(10):
            problem.reset()
            max_iters = 100000
            max_attempts = 100000
            init_temp = 10
            min_temp = .1
            exp_const = 0.0001
            schedule = mlrose.ExpDecay(init_temp= init_temp,min_temp = min_temp,exp_const = exp_const)
            start_time = time.time()
            cpk_best_state_sa, cpk_best_fitness_sa, cpk_curve_sa = mlrose.simulated_annealing(problem, schedule=schedule,max_attempts=max_attempts,max_iters=max_iters,curve=True)
            cpk_time_sa = time.time() - start_time
            cpk_iter[:, 1] += cpk_curve_sa[:, 0] / 10

        # GA
        for i in range(10):
            problem.reset()
            max_iters = 50000
            pop_size = 200
            mutation_prob = 0.1
            pop_breed_percent = 0.5
            max_attempts = 300
            start_time = time.time()
            cpk_best_state_ga, cpk_best_fitness_ga, cpk_curve_ga = mlrose.genetic_alg(problem, pop_size=pop_size, max_attempts=max_attempts, pop_breed_percent=pop_breed_percent,mutation_prob=mutation_prob,max_iters=max_iters,curve=True)
            cpk_time_ga = time.time() - start_time
            cpk_iter[:400, 2] += cpk_curve_ga[:400, 0] / 10

        # MIMIC
        for i in range(10):
            problem.reset()
            max_iters = 5000
            pop_size = 100
            keep_pct = 0.25
            max_attempts=10
            start_time = time.time()
            cpk_best_state_mi, cpk_best_fitness_mi, cpk_curve_mi = mlrose.mimic(problem,pop_size=pop_size, keep_pct=keep_pct, max_iters=max_iters,max_attempts=max_attempts,curve=True)
            cpk_time_mi = time.time() - start_time
            cpk_iter[:15, 3] += cpk_curve_mi[:15, 0] / 10

    cpk_iter[:,0] *= -1
    cpk_iter[400:,2] = cpk_iter[399,2]
    cpk_iter[15:,3] = cpk_iter[14,3]
    
    
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Continuous Peaks: Fitness vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    sns.lineplot(y=cpk_iter[:, 0], x=range(100000), label='RHC')
    sns.lineplot(y=cpk_iter[:, 1], x=range(100000), label='SA')
    sns.lineplot(y=cpk_iter[:, 2], x=range(100000), label='GA')
    sns.lineplot(y=cpk_iter[:, 3], x=range(100000), label='MIMIC')
    plt.savefig('CPK Fit vs Iter')
    
    if True:
        # RHC
        cpk_clock = np.zeros(4)
        cpk_func_calls = np.zeros(4)
        for i in range(5):
            problem.reset()
            max_iters = 100000
            restarts = 100
            max_attempts = 1000
            start_time = time.time()
            cpk_best_state_rhc, cpk_best_fitness_rhc, cpk_curve_rhc = random_hill_climb2(problem,max_iters=max_iters,max_attempts = max_attempts,restarts=restarts,curve=True)
            cpk_time_rhc = time.time() - start_time
            func_calls = problem.fitness_evaluations
            cpk_clock[0] += cpk_time_rhc/5
            cpk_func_calls[0] += func_calls/5

        # SA
        for i in range(5):
            problem.reset()
            max_iters = 100000
            max_attempts = 1000
            init_temp = 10
            min_temp = .1
            exp_const = 0.0001
            schedule = mlrose.ExpDecay(init_temp= init_temp,min_temp = min_temp,exp_const = exp_const)
            start_time = time.time()
            cpk_best_state_sa, cpk_best_fitness_sa, cpk_curve_sa = mlrose.simulated_annealing(problem, schedule=schedule,max_attempts=max_attempts,max_iters=max_iters,curve=True)
            cpk_time_sa = time.time() - start_time
            cpk_clock[1] += cpk_time_sa/5
            func_calls = problem.fitness_evaluations
            cpk_func_calls[1] += func_calls / 5

        # GA
        for i in range(5):
            problem.reset()
            max_iters = 50000
            pop_size = 200
            mutation_prob = 0.1
            pop_breed_percent = 0.5
            max_attempts = 50
            start_time = time.time()
            cpk_best_state_ga, cpk_best_fitness_ga, cpk_curve_ga = mlrose.genetic_alg(problem, pop_size=pop_size, max_attempts=max_attempts, pop_breed_percent=pop_breed_percent,mutation_prob=mutation_prob,max_iters=max_iters,curve=True)
            cpk_time_ga = time.time() - start_time
            func_calls = problem.fitness_evaluations
            cpk_clock[2] += cpk_time_ga/5
            cpk_func_calls[2] += func_calls / 5

        # MIMIC
        for i in range(5):
            problem.reset()
            max_iters = 5000
            pop_size = 100
            keep_pct = 0.25
            max_attempts=5
            start_time = time.time()
            cpk_best_state_mi, cpk_best_fitness_mi, cpk_curve_mi = mlrose.mimic(problem,pop_size=pop_size, keep_pct=keep_pct, max_iters=max_iters,max_attempts=max_attempts,curve=True)
            cpk_time_mi = time.time() - start_time
            func_calls = problem.fitness_evaluations
            cpk_clock[3] += cpk_time_mi/5
            cpk_func_calls[3] += func_calls / 5

    # fig, ax = plt.subplots(figsize=(4, 2))
    # ax2 = ax.twinx()
    # ax2.grid(False)
    # ax.grid(False)
    # plt.subplots_adjust(bottom=.26)
    # plt.subplots_adjust(left=.16)
    # plt.title('Clock Time and Function Calls')
    # plt.xlabel('Algorithm')
    # ax.set_ylabel('Clock Time (Sec)')
    # k = np.arange(4)
    # ax.bar(k, cpk_clock, 0.3)
    # ax2.scatter(k, cpk_func_calls, color='red')
    # plt.xticks(k,['RHC','SA','GA','MIMIC'])
    # ax.get_yaxis().set_visible(False)
    # ax2.get_yaxis().set_visible(False)
    # ax.set_ylim([0,max(cpk_clock)*1.2])
    # ax2.set_ylim([0,max(cpk_func_calls)*1.2])
    # for xy in zip(k, cpk_func_calls):
    #     t = int(xy[1])
    #     ax2.annotate('%s calls' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
    # for xy in zip(k, cpk_clock):
    #     t = round(xy[1], 1)
    #     ax.annotate('%s sec' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
    # plt.savefig('CPK clock and func calls')

    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Continuous Peaks: Clock Time')
    plt.xlabel('Algorithm')
    ax.set_ylabel('Clock Time (Sec)')
    k = np.arange(4)
    ax.bar(k, cpk_clock)
    plt.xticks(k, ['RHC', 'SA', 'GA', 'MIMIC'])
    for xy in zip(k, cpk_clock):
        t = round(xy[1], 1)
        ax.annotate('%s sec' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
    plt.savefig('CPK clock')

    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.23)
    plt.title('Continuous Peaks: Function Calls')
    plt.xlabel('Algorithm')
    ax.set_ylabel('Function Calls')
    k = np.arange(4)
    ax.bar(k, cpk_func_calls)
    plt.xticks(k, ['RHC', 'SA', 'GA', 'MIMIC'])
    for xy in zip(k, cpk_func_calls):
        t = int(xy[1])
        ax.annotate('%s' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
    plt.savefig('CPK calls')

    
    if True:
        cpk_input__sizes = [2,6,10,15,20,35,50,75,100,150,200,250,300]
        cpk_input__runtime = np.zeros(len(cpk_input__sizes))
        cpk_input__func_calls = np.zeros(len(cpk_input__sizes))
        # SA
        for i in range(len(cpk_input__sizes)):
            for k in range(3):
                problem = mlrose.DiscreteOpt(length=cpk_input__sizes[i], fitness_fn=fitness, maximize=True, max_val=2)
                problem.reset()
                max_iters = 100000
                max_attempts = 1000
                init_temp = min(10,cpk_input__sizes[i]/2)
                min_temp = .1
                exp_const = 0.0001
                schedule = mlrose.ExpDecay(init_temp= init_temp,min_temp = min_temp,exp_const = exp_const)
                start_time = time.time()
                cpk_best_state_sa, cpk_best_fitness_sa, cpk_curve_sa = mlrose.simulated_annealing(problem, schedule=schedule,max_attempts=max_attempts,max_iters=max_iters,curve=True)
                cpk_time_sa = time.time() - start_time
                func_calls = problem.fitness_evaluations
                cpk_input__runtime[i] += cpk_time_sa /3
                cpk_input__func_calls[i] += func_calls / 3

    
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    ax2 = ax.twinx()
    ax2.grid(False)
    ax.grid(False)
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.18)
    plt.subplots_adjust(right=.78)
    plt.title('Simulated Annealing: Time vs Problem Size')
    ax2.set_ylabel('Function Calls')
    ax.set_ylabel('Clock Time (Sec)')
    ax.set_xlabel('Problem Size')
    ax.set_ylim([0,max(cpk_input__runtime*1.5)])
    ax2.set_ylim([0,max(cpk_input__func_calls)])
    # ax.plot(cpk_input__sizes,cpk_input__runtime, label='gg')
    # ax2.plot(cpk_input__sizes, cpk_input__func_calls, color='red')
    sns.lineplot(y=cpk_input__runtime, x=cpk_input__sizes, label='Func Calls', ax=ax, color='red')
    sns.lineplot(y=cpk_input__runtime, x=cpk_input__sizes, label='Clock Time', ax=ax)
    sns.lineplot(y=cpk_input__func_calls, x=cpk_input__sizes, ax=ax2, color='red')
    plt.savefig('CPK Time vs size')

#Four peaks
if True:
    fitness = mlrose.FourPeaks()
    problem = mlrose.DiscreteOpt(length=50, fitness_fn=fitness, maximize=True, max_val=2)
    

    if True:
        # RHC
        fpk_iter = np.zeros((100000,4))
        for i in range(3):
            problem.reset()
            max_iters = 100000
            restarts = 100
            max_attempts = 1000
            start_time = time.time()
            fpk_best_state_rhc, fpk_best_fitness_rhc, fpk_curve_rhc = random_hill_climb2(problem,max_iters=max_iters,max_attempts = max_attempts,restarts=restarts,curve=True)
            fpk_time_rhc = time.time() - start_time
            fpk_iter[:,0] += fpk_curve_rhc[:,0]/3

        # SA
        for i in range(3):
            problem.reset()
            max_iters = 100000
            max_attempts = 100000
            init_temp = 100
            min_temp = .1
            exp_const = 0.0005
            schedule = mlrose.ExpDecay(init_temp= init_temp,min_temp = min_temp,exp_const = exp_const)
            start_time = time.time()
            fpk_best_state_sa, fpk_best_fitness_sa, fpk_curve_sa = mlrose.simulated_annealing(problem, schedule=schedule,max_attempts=max_attempts,max_iters=max_iters,curve=True)
            fpk_time_sa = time.time() - start_time
            fpk_iter[:, 1] += fpk_curve_sa[:, 0] / 3

        # GA
        for i in range(3):
            problem.reset()
            max_iters = 5000
            pop_size = 50
            mutation_prob = 0.2
            pop_breed_percent = 0.75
            max_attempts = 300
            start_time = time.time()
            fpk_best_state_ga, fpk_best_fitness_ga, fpk_curve_ga = mlrose.genetic_alg(problem, pop_size=pop_size, max_attempts=max_attempts, pop_breed_percent=pop_breed_percent,mutation_prob=mutation_prob,max_iters=max_iters,curve=True)
            fpk_time_ga = time.time() - start_time
            fpk_iter[:400, 2] += fpk_curve_ga[:400, 0] / 3

        # MIMIC
        for i in range(3):
            problem.reset()
            max_iters = 5000
            pop_size = 100
            keep_pct = 0.25
            max_attempts = 15
            start_time = time.time()
            fpk_best_state_mi, fpk_best_fitness_mi, fpk_curve_mi = mlrose.mimic(problem,pop_size=pop_size, keep_pct=keep_pct, max_iters=max_iters,max_attempts=max_attempts,curve=True)
            fpk_time_mi = time.time() - start_time
            fpk_iter[:15, 3] += fpk_curve_mi[:15, 0] / 3

    fpk_iter[:,0] *= -1
    fpk_iter[400:, 2] = fpk_iter[399, 2]
    fpk_iter[15:, 3] = fpk_iter[14, 3]
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Four Peaks: Fitness vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Fitness')
    sns.lineplot(y=fpk_iter[:50000, 0], x=range(50000), label='RHC')
    sns.lineplot(y=fpk_iter[:50000, 1], x=range(50000), label='SA')
    sns.lineplot(y=fpk_iter[:50000, 2], x=range(50000), label='GA')
    sns.lineplot(y=fpk_iter[:50000, 3], x=range(50000), label='MIMIC')
    plt.savefig('FPK Fit vs Iter')
    
    if True:
        # RHC
        fpk_clock = np.zeros(4)
        fpk_func_calls = np.zeros(4)
        for i in range(5):
            problem.reset()
            max_iters = 100000
            restarts = 100
            max_attempts = 1000
            start_time = time.time()
            fpk_best_state_rhc, fpk_best_fitness_rhc, fpk_curve_rhc = random_hill_climb2(problem,max_iters=max_iters,max_attempts = max_attempts,restarts=restarts,curve=True)
            fpk_time_rhc = time.time() - start_time
            func_calls = problem.fitness_evaluations
            fpk_clock[0] += fpk_time_rhc/5
            fpk_func_calls[0] += func_calls/5

        # SA
        for i in range(5):
            problem.reset()
            max_iters = 100000
            max_attempts = 1000
            init_temp = 100
            min_temp = .1
            exp_const = 0.0005
            schedule = mlrose.ExpDecay(init_temp= init_temp,min_temp = min_temp,exp_const = exp_const)
            start_time = time.time()
            fpk_best_state_sa, fpk_best_fitness_sa, fpk_curve_sa = mlrose.simulated_annealing(problem, schedule=schedule,max_attempts=max_attempts,max_iters=max_iters,curve=True)
            fpk_time_sa = time.time() - start_time
            fpk_clock[1] += fpk_time_sa/5
            func_calls = problem.fitness_evaluations
            fpk_func_calls[1] += func_calls / 5

        # GA
        for i in range(5):
            problem.reset()
            max_iters = 5000
            pop_size = 50
            mutation_prob = 0.2
            pop_breed_percent = 0.75
            max_attempts = 50
            start_time = time.time()
            fpk_best_state_ga, fpk_best_fitness_ga, fpk_curve_ga = mlrose.genetic_alg(problem, pop_size=pop_size, max_attempts=max_attempts, pop_breed_percent=pop_breed_percent,mutation_prob=mutation_prob,max_iters=max_iters,curve=True)
            fpk_time_ga = time.time() - start_time
            func_calls = problem.fitness_evaluations
            fpk_clock[2] += fpk_time_ga/5
            fpk_func_calls[2] += func_calls / 5

        # MIMIC
        for i in range(5):
            problem.reset()
            max_iters = 5000
            pop_size = 100
            keep_pct = 0.25
            max_attempts = 5
            start_time = time.time()
            fpk_best_state_mi, fpk_best_fitness_mi, fpk_curve_mi = mlrose.mimic(problem,pop_size=pop_size, keep_pct=keep_pct, max_iters=max_iters,max_attempts=max_attempts,curve=True)
            fpk_time_mi = time.time() - start_time
            func_calls = problem.fitness_evaluations
            fpk_clock[3] += fpk_time_mi/5
            fpk_func_calls[3] += func_calls / 5

    # fig, ax = plt.subplots(figsize=(4, 2))
    # ax2 = ax.twinx()
    # ax2.grid(False)
    # ax.grid(False)
    # plt.subplots_adjust(bottom=.26)
    # plt.subplots_adjust(left=.16)
    # plt.title('Clock Time and Function Calls')
    # plt.xlabel('Algorithm')
    # ax.set_ylabel('Clock Time')
    # k = np.arange(4)
    # ax.bar(k, fpk_clock, 0.3)
    # ax2.scatter(k, fpk_func_calls, color='red')
    # plt.xticks(k,['RHC','SA','GA','MIMIC'])
    # ax.get_yaxis().set_visible(False)
    # ax2.get_yaxis().set_visible(False)
    # ax.set_ylim([0,max(fpk_clock)*2.5])
    # ax2.set_ylim([0,max(fpk_func_calls)*1.2])
    # for xy in zip(k, fpk_func_calls):
    #     t = int(xy[1])
    #     ax2.annotate('%s calls' % t, xy=xy, textcoords='data', fontsize=9, ha='center')
    # for xy in zip(k, fpk_clock):
    #     t = round(xy[1], 1)
    #     ax.annotate('%s sec' % t, xy=xy, textcoords='data', fontsize=9, ha='center')
    # plt.savefig('FPK clock and func calls')

    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Four Peaks: Clock Time')
    plt.xlabel('Algorithm')
    ax.set_ylabel('Clock Time (Sec)')
    k = np.arange(4)
    ax.bar(k, fpk_clock)
    plt.xticks(k, ['RHC', 'SA', 'GA', 'MIMIC'])
    for xy in zip(k, fpk_clock):
        t = round(xy[1], 1)
        ax.annotate('%s sec' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
    plt.savefig('FPK clock')

    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.23)
    plt.title('Four Peaks: Function Calls')
    plt.xlabel('Algorithm')
    ax.set_ylabel('Function Calls')
    k = np.arange(4)
    ax.bar(k, fpk_func_calls)
    plt.xticks(k, ['RHC', 'SA', 'GA', 'MIMIC'])
    for xy in zip(k, fpk_func_calls):
        t = int(xy[1])
        ax.annotate('%s' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
    plt.savefig('FPK calls')
    
    
    if True:
        fpk_input_sizes = [2,5,10,20,35,50,75,100,150,200,250,300,400,500]
        fpk_input_runtime = np.zeros(len(fpk_input_sizes))
        fpk_input_func_calls = np.zeros(len(fpk_input_sizes))
        # GA
        for i in range(len(fpk_input_sizes)):
            for k in range(20):
                problem = mlrose.DiscreteOpt(length=fpk_input_sizes[i], fitness_fn=fitness, maximize=True, max_val=2)
                problem.reset()
                max_iters = 5000
                pop_size = 50
                mutation_prob = 0.2
                pop_breed_percent = 0.75
                max_attempts = 50
                start_time = time.time()
                fpk_best_state_ga, fpk_best_fitness_ga, fpk_curve_ga = mlrose.genetic_alg(problem, pop_size=pop_size, max_attempts=max_attempts, pop_breed_percent=pop_breed_percent,
                                                                                          mutation_prob=mutation_prob, max_iters=max_iters, curve=True)
                fpk_time_ga = time.time() - start_time
                func_calls = problem.fitness_evaluations
                fpk_input_runtime[i] += fpk_time_ga / 20
                fpk_input_func_calls[i] += func_calls / 20

    
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    ax2 = ax.twinx()
    ax2.grid(False)
    ax.grid(False)
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.18)
    plt.subplots_adjust(right=.81)
    plt.title('Genetic Algorithm: Time vs Problem Size')
    ax2.set_ylabel('Function Calls')
    ax.set_ylabel('Clock Time (Sec)')
    ax.set_xlabel('Problem Size')
    ax.set_ylim([0,max(fpk_input_runtime)*1.5])
    ax2.set_ylim([0,max(fpk_input_func_calls)])
    # ax.plot(fpk_input_sizes,fpk_input_runtime, label='gg')
    # ax2.plot(fpk_input_sizes, fpk_input_func_calls, color='red')
    sns.lineplot(y=fpk_input_runtime, x=fpk_input_sizes, label='Func Calls', ax=ax, color='red')
    sns.lineplot(y=fpk_input_runtime, x=fpk_input_sizes, label='Clock Time', ax=ax)
    sns.lineplot(y=fpk_input_func_calls, x=fpk_input_sizes, ax=ax2, color='red')
    plt.savefig('FPK Time vs size')

#Knapsack
if True:
    weights = np.random.randint(5,50,25)
    values = np.random.randint(10,100,25)
    fitness = mlrose.Knapsack(weights, values, max_weight_pct=0.5)
    problem = mlrose.KnapsackOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

    if True:
        # RHC
        knpsk_iter = np.zeros((100000, 4))
        for i in range(10):
            problem.reset()
            max_iters = 100000
            restarts = 100
            max_attempts = 1000
            start_time = time.time()
            knpsk_best_state_rhc, knpsk_best_fitness_rhc, knpsk_curve_rhc = random_hill_climb2(problem, max_iters=max_iters, max_attempts=max_attempts, restarts=restarts, curve=True)
            knpsk_time_rhc = time.time() - start_time
            knpsk_iter[:, 0] += knpsk_curve_rhc[:, 0] / 10
            rhc_func_calls = problem.fitness_evaluations

        # SA
        for i in range(10):
            problem.reset()
            max_iters = 100000
            max_attempts = 100000
            init_temp = 1000
            min_temp = 1
            exp_const = 0.0005
            schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=exp_const)
            start_time = time.time()
            knpsk_best_state_sa, knpsk_best_fitness_sa, knpsk_curve_sa = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=max_attempts, max_iters=max_iters, curve=True)
            knpsk_time_sa = time.time() - start_time
            knpsk_iter[:, 1] += knpsk_curve_sa[:, 0] / 10

        # GA
        for i in range(10):
            problem.reset()
            max_iters = 5000
            pop_size = 200
            mutation_prob = 0.2
            pop_breed_percent = 0.5
            max_attempts = 500
            start_time = time.time()
            knpsk_best_state_ga, knpsk_best_fitness_ga, knpsk_curve_ga = mlrose.genetic_alg(problem, pop_size=pop_size, max_attempts=max_attempts, pop_breed_percent=pop_breed_percent,
                                                                                      mutation_prob=mutation_prob, max_iters=max_iters, curve=True)
            knpsk_time_ga = time.time() - start_time
            ga_func_calls = problem.fitness_evaluations
            knpsk_iter[:500, 2] += knpsk_curve_ga[:500, 0] / 10

        # MIMIC
        for i in range(10):
            problem.reset()
            max_iters = 5000
            pop_size = 400
            keep_pct = 0.25
            max_attempts = 15
            start_time = time.time()
            knpsk_best_state_mi, knpsk_best_fitness_mi, knpsk_curve_mi = mlrose.mimic(problem, pop_size=pop_size, keep_pct=keep_pct, max_iters=max_iters, max_attempts=max_attempts, curve=True)
            knpsk_time_mi = time.time() - start_time
            mimic_func_calls = problem.fitness_evaluations
            knpsk_iter[:18, 3] += knpsk_curve_mi[:18, 0] / 10

    knpsk_iter[:, 0] *= -1
    knpsk_iter[500:, 2] = knpsk_iter[499, 2]
    knpsk_iter[15:, 3] = knpsk_iter[14, 3]

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.29)
    plt.subplots_adjust(left=.18)
    plt.title('Knapsack: Fitness vs Iteration')
    plt.xlabel('Iteration: Log Scale to separate GA and MIMIC')
    plt.ylabel('Fitness')
    sns.lineplot(y=knpsk_iter[:20000, 0], x=range(20000), label='RHC')
    sns.lineplot(y=knpsk_iter[:20000, 1], x=range(20000), label='SA')
    sns.lineplot(y=knpsk_iter[:20000, 2], x=range(20000), label='GA')
    sns.lineplot(y=knpsk_iter[:20000, 3], x=range(20000), label='MIMIC')
    plt.legend(loc='lower center')
    ax.set_xscale('log', basex=10)
    plt.savefig('Knpsk Fit vs Iter')

    if True:
        # RHC
        knpsk_clock = np.zeros(4)
        knpsk_func_calls = np.zeros(4)
        for i in range(5):
            problem.reset()
            max_iters = 100000
            restarts = 100
            max_attempts = 1000
            start_time = time.time()
            knpsk_best_state_rhc, knpsk_best_fitness_rhc, knpsk_curve_rhc = random_hill_climb2(problem, max_iters=max_iters, max_attempts=max_attempts, restarts=restarts, curve=True)
            knpsk_time_rhc = time.time() - start_time
            func_calls = problem.fitness_evaluations
            knpsk_clock[0] += knpsk_time_rhc / 5
            knpsk_func_calls[0] += func_calls / 5

        # SA
        for i in range(5):
            problem.reset()
            max_iters = 100000
            max_attempts = 1000
            init_temp = 1000
            min_temp = 1
            exp_const = 0.0005
            schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=exp_const)
            start_time = time.time()
            knpsk_best_state_sa, knpsk_best_fitness_sa, knpsk_curve_sa = mlrose.simulated_annealing(problem, schedule=schedule, max_attempts=max_attempts, max_iters=max_iters, curve=True)
            knpsk_time_sa = time.time() - start_time
            knpsk_clock[1] += knpsk_time_sa / 5
            func_calls = problem.fitness_evaluations
            knpsk_func_calls[1] += func_calls / 5

        # GA
        for i in range(5):
            problem.reset()
            max_iters = 5000
            pop_size = 200
            mutation_prob = 0.2
            pop_breed_percent = 0.5
            max_attempts = 50
            start_time = time.time()
            knpsk_best_state_ga, knpsk_best_fitness_ga, knpsk_curve_ga = mlrose.genetic_alg(problem, pop_size=pop_size, max_attempts=max_attempts, pop_breed_percent=pop_breed_percent,
                                                                                      mutation_prob=mutation_prob, max_iters=max_iters, curve=True)
            knpsk_time_ga = time.time() - start_time
            func_calls = problem.fitness_evaluations
            knpsk_clock[2] += knpsk_time_ga / 5
            knpsk_func_calls[2] += func_calls / 5

        # MIMIC
        for i in range(5):
            problem.reset()
            max_iters = 5000
            pop_size = 400
            keep_pct = 0.25
            max_attempts = 2
            start_time = time.time()
            knpsk_best_state_mi, knpsk_best_fitness_mi, knpsk_curve_mi = mlrose.mimic(problem, pop_size=pop_size, keep_pct=keep_pct, max_iters=max_iters, max_attempts=max_attempts, curve=True)
            knpsk_time_mi = time.time() - start_time
            func_calls = problem.fitness_evaluations
            knpsk_clock[3] += knpsk_time_mi / 5
            knpsk_func_calls[3] += func_calls / 5

    # fig, ax = plt.subplots(figsize=(4, 2))
    # ax2 = ax.twinx()
    # ax2.grid(False)
    # ax.grid(False)
    # plt.subplots_adjust(bottom=.26)
    # plt.subplots_adjust(left=.16)
    # plt.title('Clock Time and Function Calls')
    # plt.xlabel('Algorithm')
    # ax.set_ylabel('Clock Time')
    # k = np.arange(4)
    # ax.bar(k, knpsk_clock, 0.3)
    # ax2.scatter(k, knpsk_func_calls, color='red')
    # plt.xticks(k, ['RHC', 'SA', 'GA', 'MIMIC'])
    # ax.get_yaxis().set_visible(False)
    # ax2.get_yaxis().set_visible(False)
    # ax.set_ylim([0, max(knpsk_clock) * 2.5])
    # ax2.set_ylim([0, max(knpsk_func_calls) * 1.2])
    # for xy in zip(k, knpsk_func_calls):
    #     t = int(xy[1])
    #     ax2.annotate('%s calls' % t, xy=xy, verticalalignment='bottom', fontsize=10, ha='center', clip_on=False)
    # for i, xy in enumerate(zip(k, knpsk_clock)):
    #     t = round(xy[1], 1)
    #     if i == 0 or i == 3:
    #         ax.annotate('%s sec' % t, xy=xy, verticalalignment='bottom', fontsize=10, ha='center', clip_on=False)
    #     else:
    #         ax.annotate('%s sec' % t, xy=xy, verticalalignment='top', fontsize=10, ha='center', clip_on=False)
    # plt.savefig('KNPSK clock and func calls')

    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.16)
    plt.title('Knapsack: Clock Time')
    plt.xlabel('Algorithm')
    ax.set_ylabel('Clock Time (Sec)')
    k = np.arange(4)
    ax.bar(k, knpsk_clock)
    plt.xticks(k, ['RHC', 'SA', 'GA', 'MIMIC'])
    for xy in zip(k, knpsk_clock):
        t = round(xy[1], 1)
        ax.annotate('%s sec' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
    plt.savefig('KNPSK clock')

    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.23)
    plt.title('Knapsack: Function Calls')
    plt.xlabel('Algorithm')
    ax.set_ylabel('Function Calls')
    k = np.arange(4)
    ax.bar(k, knpsk_func_calls)
    plt.xticks(k, ['RHC', 'SA', 'GA', 'MIMIC'])
    for xy in zip(k, knpsk_func_calls):
        t = int(xy[1])
        ax.annotate('%s' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
    plt.savefig('KNPSK calls')

    if True:
        knpsk_input_sizes = [2, 6, 10, 15, 20, 35, 50, 75, 100, 150, 200]
        knpsk_input_runtime = np.zeros(len(knpsk_input_sizes))
        knpsk_input_func_calls = np.zeros(len(knpsk_input_sizes))
        # MI
        for i in range(len(knpsk_input_sizes)):
            for k in range(5):
                weights = np.random.randint(5, 50, knpsk_input_sizes[i])
                values = np.random.randint(10, 100, knpsk_input_sizes[i])
                fitness = mlrose.Knapsack(weights, values, max_weight_pct=0.5)
                problem = mlrose.KnapsackOpt(length=len(weights), fitness_fn=fitness, maximize=True, max_val=2)

                max_iters = 5000
                pop_size = 400
                keep_pct = 0.25
                max_attempts = 2
                start_time = time.time()
                knpsk_best_state_mi, knpsk_best_fitness_mi, knpsk_curve_mi = mlrose.mimic(problem, pop_size=pop_size, keep_pct=keep_pct, max_iters=max_iters, max_attempts=max_attempts, curve=True)
                knpsk_time_mi = time.time() - start_time
                func_calls = problem.fitness_evaluations
                knpsk_input_runtime[i] += knpsk_time_mi / 5
                knpsk_input_func_calls[i] += func_calls / 5

    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    ax2 = ax.twinx()
    ax2.grid(False)
    ax.grid(False)
    plt.subplots_adjust(bottom=.26)
    plt.subplots_adjust(left=.18)
    plt.subplots_adjust(right=.81)
    plt.title('MIMIC: Time vs Problem Size')
    ax2.set_ylabel('Function Calls')
    ax.set_ylabel('Clock Time (Sec)')
    ax.set_xlabel('Problem Size')
    ax.set_ylim([0, max(knpsk_input_runtime)])
    ax2.set_ylim([0, max(knpsk_input_func_calls) * 1.5])
    sns.lineplot(y=knpsk_input_runtime, x=knpsk_input_sizes, label='Func Calls', ax=ax, color='red')
    sns.lineplot(y=knpsk_input_runtime, x=knpsk_input_sizes, label='Clock Time', ax=ax)
    sns.lineplot(y=knpsk_input_func_calls, x=knpsk_input_sizes, ax=ax2, color='red')
    plt.savefig('KNPSK Time vs size')


#Part 2: NN

data = loadtxt(open('wine_dataset.csv'), delimiter=",", dtype=object)
# data[data=='white'] = 0
# data[data=='red'] = 1
data = data[data[:,-1]=='red']
# temp = data[:,-1].copy()
# data[:,-1] = data[:,-2]
# data[:,-2] = temp
data = data[:,:-1]
cols = data[0, :]
data = data[1:, :]


data_x = data[:, :-1]
data_x = np.array(data_x, dtype=float)
data_y = data[:, -1]

data_y_class = data_y.copy()
data_y_class[1 == 1] = 0
data_y_class[np.array(data_y, dtype=int) >= 6] = 1
# data_y_class[np.array(data_y, dtype=float)>=23]=1
data_y_class = np.array(data_y_class, dtype=int)
sum(data_y_class)
train_x, test_x, train_y, test_y = train_test_split(data_x, data_y_class, test_size=0.2, random_state=100)

txt_mean = train_x.mean(axis=0)
txt_std = train_x.std(axis=0)
train_x = (train_x - txt_mean) / txt_std
test_x = (test_x - txt_mean) / txt_std
samples, features = train_x.shape

lrs = [0.01, 0.1, .15, .2, 0.25, .3, .35, 0.5, 0.75, 1]
for i in range(len(lrs)):
    acc = 0
    acc_tr = 0
    for k in range(3):
        max_iters = 500
        max_attempts = 50
        init_temp = 1
        min_temp = .0001
        exp_const = 0.05
        lr = lrs[i]
        schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=exp_const)
        nn_sa = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True,learning_rate=lr, early_stopping=True,algorithm='simulated_annealing',max_iters=max_iters,max_attempts=max_attempts, schedule=schedule)
        nn_sa.fit(train_x, train_y)
        y_pred = nn_sa.predict(test_x)
        y_pred_tr = nn_sa.predict(train_x)
        acc += accuracy_score(test_y, y_pred)/3
        acc_tr += accuracy_score(train_y, y_pred_tr)/3
    print(lr)
    print(acc_tr)
    print(acc)
    print('-----')

itemps = [0.1, 0.25, .5, 1, 2, 5, 10]
for i in range(len(itemps)):
    acc = 0
    acc_tr = 0
    for k in range(3):
        max_iters = 500
        max_attempts = 50
        init_temp = itemps[i]
        min_temp = .0001
        exp_const = 0.05
        lr = 0.5
        schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=exp_const)
        nn_sa = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True,learning_rate=lr, early_stopping=True,algorithm='simulated_annealing',max_iters=max_iters,max_attempts=max_attempts, schedule=schedule)
        nn_sa.fit(train_x, train_y)
        y_pred = nn_sa.predict(test_x)
        y_pred_tr = nn_sa.predict(train_x)
        acc += accuracy_score(test_y, y_pred)/3
        acc_tr += accuracy_score(train_y, y_pred_tr)/3
    print(init_temp)
    print(acc_tr)
    print(acc)
    print('-----')

decay = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
for i in range(len(decay)):
    acc = 0
    acc_tr = 0
    for k in range(5):
        max_iters = 500
        max_attempts = 50
        init_temp = 1
        min_temp = .0001
        exp_const = decay[i]
        lr = 0.5
        schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=exp_const)
        nn_sa = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True,learning_rate=lr, early_stopping=True,algorithm='simulated_annealing',max_iters=max_iters,max_attempts=max_attempts, schedule=schedule)
        nn_sa.fit(train_x, train_y)
        y_pred = nn_sa.predict(test_x)
        y_pred_tr = nn_sa.predict(train_x)
        acc += accuracy_score(test_y, y_pred)/5
        acc_tr += accuracy_score(train_y, y_pred_tr)/5
    print(decay[i])
    print(acc_tr)
    print(acc)
    print('-----')

#SA
acc = 0
acc_tr = 0
acc_iters_sa = np.zeros((10000,2))
for k in range(5):
    max_iters = 10000
    max_attempts = 500
    init_temp = 0.01
    min_temp = .00001
    exp_const = 0.001
    lr = 0.3
    schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=0.001)
    nn_sa = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True,learning_rate=lr, early_stopping=True,algorithm='simulated_annealing',max_iters=max_iters,max_attempts=max_attempts, schedule=schedule, curve=True)
    nn_sa_res = nn_sa.fit(train_x, train_y)
    acc_iters_sa[:,0] += nn_sa_res.fitness_curve[:,0]/5
    y_pred = nn_sa.predict(test_x)
    y_pred_tr = nn_sa.predict(train_x)
    acc += accuracy_score(test_y, y_pred)/5
    acc_tr += accuracy_score(train_y, y_pred_tr)/5
    print(accuracy_score(train_y, y_pred_tr))
    print(accuracy_score(test_y, y_pred))
    print('--')
print(acc_tr)
print(acc)
print('-----')

#RHC
acc = 0
acc_tr = 0
for k in range(5):
    max_iters = 1000
    max_attempts = 50
    restarts = 100
    lr = 0.3
    nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True,learning_rate=lr, early_stopping=True,algorithm='random_hill_climb',max_iters=max_iters,max_attempts=max_attempts, restarts=restarts, curve=True)
    nn_rhc_res = nn_rhc.fit(train_x, train_y)
    y_pred = nn_rhc.predict(test_x)
    y_pred_tr = nn_rhc.predict(train_x)
    acc += accuracy_score(test_y, y_pred)/5
    acc_tr += accuracy_score(train_y, y_pred_tr)/5
    print(accuracy_score(train_y, y_pred_tr))
    print(accuracy_score(test_y, y_pred))
    print('--')
print(acc_tr)
print(acc)
print('-----')

#GA
acc = 0
acc_tr = 0
acc_iters_ga = np.zeros((10000,2))
for k in range(5):
    max_iters = 100
    max_attempts = 50
    mutation_prob = 0.1
    pop_size = 60
    lr = 0.05
    nn_ga = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True,learning_rate=lr,
                                 early_stopping=True,algorithm='genetic_alg',max_iters=max_iters,max_attempts=max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, curve=True)
    nn_ga_res = nn_ga.fit(train_x, train_y)
    temp = nn_ga_res.fitness_curve[:,0]
    acc_iters_ga[:len(temp),0] += temp
    acc_iters_ga[:len(temp), 1] += 1
    y_pred = nn_ga.predict(test_x)
    y_pred_tr = nn_ga.predict(train_x)
    acc += accuracy_score(test_y, y_pred)/5
    acc_tr += accuracy_score(train_y, y_pred_tr)/5
    print(accuracy_score(train_y, y_pred_tr))
    print(accuracy_score(test_y, y_pred))
    print('--')
print(acc_tr)
print(acc)
print('-----')

#GD
acc = 0
acc_tr = 0
acc_iters_gd = np.zeros((10000,2))
for k in range(5):
    max_iters = 1000
    max_attempts = 50
    lr = 0.001
    nn_gd = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True,learning_rate=lr,
                                 early_stopping=True,algorithm='gradient_descent',max_iters=max_iters,max_attempts=max_attempts, curve=True)
    nn_gd_res = nn_gd.fit(train_x, train_y)
    temp = nn_gd_res.fitness_curve
    acc_iters_gd[:len(temp),0] += temp
    acc_iters_gd[:len(temp), 1] += 1
    y_pred = nn_gd.predict(test_x)
    y_pred_tr = nn_gd.predict(train_x)
    acc += accuracy_score(test_y, y_pred)/5
    acc_tr += accuracy_score(train_y, y_pred_tr)/5
    print(accuracy_score(train_y, y_pred_tr))
    print(accuracy_score(test_y, y_pred))
    print('--')
print(acc_tr)
print(acc)
print('-----')


iters_all = [1,2,4,8,12,16,20,24,28,32,36,40,50,75,100,150,200,300,400,500,750,1000,1250, 1500, 2000, 2500,5000,10000]

acc_iters_all = np.zeros((len(iters_all),9))
acc_iters_all[:,0] = iters_all
for i, iter in enumerate(iters_all):
    print(iter)
    acc = 0
    acc_tr = 0
    for k in range(5):
        max_iters = iter
        max_attempts = 50
        lr = 0.001
        nn_gd = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='gradient_descent', max_iters=max_iters,
                                     max_attempts=max_attempts, curve=True)
        nn_gd_res = nn_gd.fit(train_x, train_y)
        y_pred = nn_gd.predict(test_x)
        y_pred_tr = nn_gd.predict(train_x)
        acc += accuracy_score(test_y, y_pred) / 5
        acc_tr += accuracy_score(train_y, y_pred_tr) / 5
    acc_iters_all[i,1] = acc_tr
    acc_iters_all[i,2] = acc
    print('GD', acc_tr, acc)

    acc = 0
    acc_tr = 0
    for k in range(5):
        max_iters = iter
        max_attempts = 10
        restarts = 50
        lr = 0.3
        nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='random_hill_climb', max_iters=max_iters,
                                      max_attempts=max_attempts, restarts=restarts, curve=True)
        nn_rhc_res = nn_rhc.fit(train_x, train_y)
        y_pred = nn_rhc.predict(test_x)
        y_pred_tr = nn_rhc.predict(train_x)
        acc += accuracy_score(test_y, y_pred) / 5
        acc_tr += accuracy_score(train_y, y_pred_tr) / 5
    acc_iters_all[i,3] = acc_tr
    acc_iters_all[i,4] = acc
    print('RHC', acc_tr, acc)

    acc = 0
    acc_tr = 0
    for k in range(5):
        max_iters = iter
        max_attempts = 500
        init_temp = 0.01
        min_temp = .00001
        exp_const = 0.001
        lr = 0.3
        schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=0.001)
        nn_sa = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='simulated_annealing', max_iters=max_iters,
                                     max_attempts=max_attempts, schedule=schedule, curve=True)
        nn_sa_res = nn_sa.fit(train_x, train_y)
        y_pred = nn_sa.predict(test_x)
        y_pred_tr = nn_sa.predict(train_x)
        acc += accuracy_score(test_y, y_pred) / 5
        acc_tr += accuracy_score(train_y, y_pred_tr) / 5
    acc_iters_all[i,5] = acc_tr
    acc_iters_all[i,6] = acc
    print('SA', acc_tr, acc)

    acc = 0
    acc_tr = 0
    for k in range(5):
        max_iters = iter
        max_attempts = 100
        mutation_prob = 0.1
        pop_size =80
        lr = 0.3
        nn_ga = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='genetic_alg', max_iters=max_iters,
                                     max_attempts=max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, curve=True)
        nn_ga_res = nn_ga.fit(train_x, train_y)
        y_pred = nn_ga.predict(test_x)
        y_pred_tr = nn_ga.predict(train_x)
        acc += accuracy_score(test_y, y_pred) / 5
        acc_tr += accuracy_score(train_y, y_pred_tr) / 5
    acc_iters_all[i,7] = acc_tr
    acc_iters_all[i,8] = acc
    print('GA', acc_tr, acc)

acc_iters_all[:,7:]=0
for i, iter in enumerate(iters_all):
    print(iter)
    acc = 0
    acc_tr = 0
    for k in range(3):
        max_iters = iter
        max_attempts = 100
        mutation_prob = 0.1
        pop_size = 80
        lr = 0.3
        nn_ga = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='genetic_alg', max_iters=max_iters,
                                     max_attempts=max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, curve=True)
        nn_ga_res = nn_ga.fit(train_x, train_y)
        y_pred = nn_ga.predict(test_x)
        y_pred_tr = nn_ga.predict(train_x)
        acc += accuracy_score(test_y, y_pred) / 3
        acc_tr += accuracy_score(train_y, y_pred_tr) / 3
    acc_iters_all[i,7] = acc_tr
    acc_iters_all[i,8] = acc
    print('GA', acc_tr, acc)


acc_res = np.zeros((len(iters_all)+1,9)).astype(object)
acc_res[1:,:] = acc_iters_all
acc_res[0,:] = ['iterations', 'GD Train', 'GD Test', 'RHC Train', 'RHC Test', 'SA Train', 'SA Test', 'GA Train', 'GA Test']


sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.29)
plt.subplots_adjust(left=.18)
plt.title('NN 32: Train Acc vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
sns.lineplot(y=acc_iters_all[:, 1], x=iters_all, label='Grad Desc')
sns.lineplot(y=acc_iters_all[:, 3], x=iters_all, label='RHC')
sns.lineplot(y=acc_iters_all[:, 5], x=iters_all, label='SA')
sns.lineplot(y=acc_iters_all[:, 7], x=iters_all, label='GA')
plt.legend(loc='lower center')
plt.savefig('NN32 Train vs Iter')

sns.set()
fig, ax = plt.subplots(figsize=(4, 2))
plt.subplots_adjust(bottom=.29)
plt.subplots_adjust(left=.18)
plt.title('NN 32: Test Acc vs Iteration')
plt.xlabel('Iteration')
plt.ylabel('Accuracy')
sns.lineplot(y=acc_iters_all[:, 2], x=iters_all, label='Grad Desc')
sns.lineplot(y=acc_iters_all[:, 4], x=iters_all, label='RHC')
sns.lineplot(y=acc_iters_all[:, 6], x=iters_all, label='SA')
sns.lineplot(y=acc_iters_all[:, 8], x=iters_all, label='GA')
plt.legend(loc='lower center')
plt.savefig('NN32 Test vs Iter')


gd_iter = 1000
rhc_iter = 1000
sa_iter = 10000
ga_iter = 200

time_all = np.zeros((4,5)).astype(object)
time_all[0,:] = ['GD 1,000', 'GD 10,000','RHC 1,000', 'SA 10,000', 'GA 200']
avg = 5

start_time = time.time()
for i in range(avg):
    max_iters = 1000
    max_attempts = 50
    lr = 0.001
    nn_gd = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='gradient_descent', max_iters=max_iters,
                                 max_attempts=max_attempts, curve=True)
    nn_gd_res = nn_gd.fit(train_x, train_y)
    y_pred = nn_gd.predict(test_x)
    y_pred_tr = nn_gd.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all[2,0] += acc_tr/avg
    time_all[3, 0] += acc/avg
    print('GD', i)
time_all[1,0] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 10000
    max_attempts = 50
    lr = 0.001
    nn_gd = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='gradient_descent', max_iters=max_iters,
                                 max_attempts=max_attempts, curve=True)
    nn_gd_res = nn_gd.fit(train_x, train_y)
    y_pred = nn_gd.predict(test_x)
    y_pred_tr = nn_gd.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all[2, 1] += acc_tr /avg
    time_all[3, 1] += acc /avg
    print('GD2', i)
time_all[1,1] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 1000
    max_attempts = 50
    restarts = 100
    lr = 0.3
    nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='random_hill_climb', max_iters=max_iters,
                                  max_attempts=max_attempts, restarts=restarts, curve=True)
    nn_rhc_res = nn_rhc.fit(train_x, train_y)
    y_pred = nn_rhc.predict(test_x)
    y_pred_tr = nn_rhc.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all[2, 2] += acc_tr /avg
    time_all[3, 2] += acc /avg
    print('RHC', i)
time_all[1,2] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 10000
    max_attempts = 500
    init_temp = 0.01
    min_temp = .00001
    exp_const = 0.001
    lr = 0.3
    schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=0.001)
    nn_sa = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='simulated_annealing', max_iters=max_iters,
                                 max_attempts=max_attempts, schedule=schedule, curve=True)
    nn_sa_res = nn_sa.fit(train_x, train_y)
    y_pred = nn_sa.predict(test_x)
    y_pred_tr = nn_sa.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all[2, 3] += acc_tr /avg
    time_all[3, 3] += acc /avg
    print('SA', i)
time_all[1,3] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 200
    max_attempts = 100
    mutation_prob = 0.1
    pop_size = 80
    lr = 0.3
    nn_ga = mlrose.NeuralNetwork(hidden_nodes=[32], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='genetic_alg', max_iters=max_iters,
                                 max_attempts=max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, curve=True)
    nn_ga_res = nn_ga.fit(train_x, train_y)
    y_pred = nn_ga.predict(test_x)
    y_pred_tr = nn_ga.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all[2, 4] += acc_tr /avg
    time_all[3, 4] += acc /avg
    print('GA', i)
time_all[1,4] = (time.time() - start_time)/avg


fig, ax = plt.subplots(figsize=(8, 2))
ax2 = ax.twinx()
ax2.grid(False)
ax.grid(False)
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('NN Hidden Layer Size 32:Clock Time and Accuracy')
plt.xlabel('Algorithm + Iterations')
ax.set_ylabel('Clock Time (Sec)')
ax2.set_ylabel('Accuracy')
k = np.arange(5)
ax.bar(k, time_all[1,:], 0.3)
ax2.set_ylim(bottom=0,top=1)
ax.set_ylim([0,max(time_all[1,:])*2])
ax2.scatter(k, time_all[3,:], color='red')
ax2.scatter(k, time_all[2,:], color='orange')
plt.xticks(k,['GD 1,000', 'GD 10,000','RHC 1,000', 'SA 10,000', 'GA 200'])
# ax.get_yaxis().set_visible(False)
# ax2.get_yaxis().set_visible(False)
#
for xy in zip(k, time_all[2,:]):
    t = round(xy[1], 3)
    ax2.annotate('Train: %s' % t, xy=xy, textcoords='data', fontsize=10, ha='center', verticalalignment='bottom')
for xy in zip(k, time_all[3, :]):
    t = round(xy[1], 3)
    ax2.annotate('Test: %s' % t, xy=xy, textcoords='data', fontsize=10, ha='center', verticalalignment='top')
for xy in zip(k, time_all[1,:]):
    t = round(xy[1], 2)
    ax.annotate('%s sec' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
plt.savefig('NN32 clock and accuracy')


if True:
    acc_iters_all_64 = np.zeros((len(iters_all),9))
    acc_iters_all_64[:,0] = iters_all
    for i, iter in enumerate(iters_all):
        print(iter)
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 50
            lr = 0.001
            nn_gd = mlrose.NeuralNetwork(hidden_nodes=[64], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='gradient_descent', max_iters=max_iters,
                                         max_attempts=max_attempts, curve=True)
            nn_gd_res = nn_gd.fit(train_x, train_y)
            y_pred = nn_gd.predict(test_x)
            y_pred_tr = nn_gd.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_64[i,1] = acc_tr
        acc_iters_all_64[i,2] = acc
        print('GD', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 10
            restarts = 50
            lr = 0.3
            nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[64], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='random_hill_climb', max_iters=max_iters,
                                          max_attempts=max_attempts, restarts=restarts, curve=True)
            nn_rhc_res = nn_rhc.fit(train_x, train_y)
            y_pred = nn_rhc.predict(test_x)
            y_pred_tr = nn_rhc.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_64[i,3] = acc_tr
        acc_iters_all_64[i,4] = acc
        print('RHC', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 500
            init_temp = 0.01
            min_temp = .00001
            exp_const = 0.001
            lr = 0.3
            schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=0.001)
            nn_sa = mlrose.NeuralNetwork(hidden_nodes=[64], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='simulated_annealing', max_iters=max_iters,
                                         max_attempts=max_attempts, schedule=schedule, curve=True)
            nn_sa_res = nn_sa.fit(train_x, train_y)
            y_pred = nn_sa.predict(test_x)
            y_pred_tr = nn_sa.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_64[i,5] = acc_tr
        acc_iters_all_64[i,6] = acc
        print('SA', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 100
            mutation_prob = 0.1
            pop_size = 80
            lr = 0.3
            nn_ga = mlrose.NeuralNetwork(hidden_nodes=[64], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='genetic_alg', max_iters=max_iters,
                                         max_attempts=max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, curve=True)
            nn_ga_res = nn_ga.fit(train_x, train_y)
            y_pred = nn_ga.predict(test_x)
            y_pred_tr = nn_ga.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_64[i,7] = acc_tr
        acc_iters_all_64[i,8] = acc
        print('GA', acc_tr, acc)
    
    acc_res_64 = np.zeros((len(iters_all)+1,9)).astype(object)
    acc_res_64[1:,:] = acc_iters_all_64
    acc_res_64[0,:] = ['iterations', 'GD Train', 'GD Test', 'RHC Train', 'RHC Test', 'SA Train', 'SA Test', 'GA Train', 'GA Test']
    
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.29)
    plt.subplots_adjust(left=.18)
    plt.title('NN 64: Train Acc vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    sns.lineplot(y=acc_iters_all_64[:, 1], x=iters_all, label='Grad Desc')
    sns.lineplot(y=acc_iters_all_64[:, 3], x=iters_all, label='RHC')
    sns.lineplot(y=acc_iters_all_64[:, 5], x=iters_all, label='SA')
    sns.lineplot(y=acc_iters_all_64[:, 7], x=iters_all, label='GA')
    plt.legend(loc='lower center')
    plt.savefig('NN64 Train vs Iter')
    
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.29)
    plt.subplots_adjust(left=.18)
    plt.title('NN 64: Test Acc vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    sns.lineplot(y=acc_iters_all_64[:, 2], x=iters_all, label='Grad Desc')
    sns.lineplot(y=acc_iters_all_64[:, 4], x=iters_all, label='RHC')
    sns.lineplot(y=acc_iters_all_64[:, 6], x=iters_all, label='SA')
    sns.lineplot(y=acc_iters_all_64[:, 8], x=iters_all, label='GA')
    plt.legend(loc='lower center')
    plt.savefig('NN64 Test vs Iter')
    
    acc_iters_all_128 = np.zeros((len(iters_all),9))
    acc_iters_all_128[:,0] = iters_all
    for i, iter in enumerate(iters_all):
        print(iter)
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 50
            lr = 0.001
            nn_gd = mlrose.NeuralNetwork(hidden_nodes=[128], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='gradient_descent', max_iters=max_iters,
                                         max_attempts=max_attempts, curve=True)
            nn_gd_res = nn_gd.fit(train_x, train_y)
            y_pred = nn_gd.predict(test_x)
            y_pred_tr = nn_gd.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_128[i,1] = acc_tr
        acc_iters_all_128[i,2] = acc
        print('GD', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 10
            restarts = 50
            lr = 0.3
            nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[128], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='random_hill_climb', max_iters=max_iters,
                                          max_attempts=max_attempts, restarts=restarts, curve=True)
            nn_rhc_res = nn_rhc.fit(train_x, train_y)
            y_pred = nn_rhc.predict(test_x)
            y_pred_tr = nn_rhc.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_128[i,3] = acc_tr
        acc_iters_all_128[i,4] = acc
        print('RHC', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 500
            init_temp = 0.01
            min_temp = .00001
            exp_const = 0.001
            lr = 0.3
            schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=0.001)
            nn_sa = mlrose.NeuralNetwork(hidden_nodes=[128], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='simulated_annealing', max_iters=max_iters,
                                         max_attempts=max_attempts, schedule=schedule, curve=True)
            nn_sa_res = nn_sa.fit(train_x, train_y)
            y_pred = nn_sa.predict(test_x)
            y_pred_tr = nn_sa.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_128[i,5] = acc_tr
        acc_iters_all_128[i,6] = acc
        print('SA', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 100
            mutation_prob = 0.1
            pop_size = 80
            lr = 0.3
            nn_ga = mlrose.NeuralNetwork(hidden_nodes=[128], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='genetic_alg', max_iters=max_iters,
                                         max_attempts=max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, curve=True)
            nn_ga_res = nn_ga.fit(train_x, train_y)
            y_pred = nn_ga.predict(test_x)
            y_pred_tr = nn_ga.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_128[i,7] = acc_tr
        acc_iters_all_128[i,8] = acc
        print('GA', acc_tr, acc)
    
    acc_res_128 = np.zeros((len(iters_all)+1,9)).astype(object)
    acc_res_128[1:,:] = acc_iters_all_128
    acc_res_128[0,:] = ['iterations', 'GD Train', 'GD Test', 'RHC Train', 'RHC Test', 'SA Train', 'SA Test', 'GA Train', 'GA Test']
    
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.29)
    plt.subplots_adjust(left=.18)
    plt.title('NN 128: Train Acc vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    sns.lineplot(y=acc_iters_all_128[:, 1], x=iters_all, label='Grad Desc')
    sns.lineplot(y=acc_iters_all_128[:, 3], x=iters_all, label='RHC')
    sns.lineplot(y=acc_iters_all_128[:, 5], x=iters_all, label='SA')
    sns.lineplot(y=acc_iters_all_128[:, 7], x=iters_all, label='GA')
    plt.legend(loc='lower center')
    plt.savefig('NN128 Train vs Iter')
    
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.29)
    plt.subplots_adjust(left=.18)
    plt.title('NN 128: Test Acc vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    sns.lineplot(y=acc_iters_all_128[:, 2], x=iters_all, label='Grad Desc')
    sns.lineplot(y=acc_iters_all_128[:, 4], x=iters_all, label='RHC')
    sns.lineplot(y=acc_iters_all_128[:, 6], x=iters_all, label='SA')
    sns.lineplot(y=acc_iters_all_128[:, 8], x=iters_all, label='GA')
    plt.legend(loc='lower center')
    plt.savefig('NN128 Test vs Iter')
    
    
    acc_iters_all_256 = np.zeros((len(iters_all),9))
    acc_iters_all_256[:,0] = iters_all
    for i, iter in enumerate(iters_all):
        print(iter)
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 50
            lr = 0.001
            nn_gd = mlrose.NeuralNetwork(hidden_nodes=[256], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='gradient_descent', max_iters=max_iters,
                                         max_attempts=max_attempts, curve=True)
            nn_gd_res = nn_gd.fit(train_x, train_y)
            y_pred = nn_gd.predict(test_x)
            y_pred_tr = nn_gd.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_256[i,1] = acc_tr
        acc_iters_all_256[i,2] = acc
        print('GD', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 10
            restarts = 50
            lr = 0.3
            nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[256], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='random_hill_climb', max_iters=max_iters,
                                          max_attempts=max_attempts, restarts=restarts, curve=True)
            nn_rhc_res = nn_rhc.fit(train_x, train_y)
            y_pred = nn_rhc.predict(test_x)
            y_pred_tr = nn_rhc.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_256[i,3] = acc_tr
        acc_iters_all_256[i,4] = acc
        print('RHC', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 500
            init_temp = 0.01
            min_temp = .00001
            exp_const = 0.001
            lr = 0.3
            schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=0.001)
            nn_sa = mlrose.NeuralNetwork(hidden_nodes=[256], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='simulated_annealing', max_iters=max_iters,
                                         max_attempts=max_attempts, schedule=schedule, curve=True)
            nn_sa_res = nn_sa.fit(train_x, train_y)
            y_pred = nn_sa.predict(test_x)
            y_pred_tr = nn_sa.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_256[i,5] = acc_tr
        acc_iters_all_256[i,6] = acc
        print('SA', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 100
            mutation_prob = 0.1
            pop_size = 80
            lr = 0.3
            nn_ga = mlrose.NeuralNetwork(hidden_nodes=[256], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='genetic_alg', max_iters=max_iters,
                                         max_attempts=max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, curve=True)
            nn_ga_res = nn_ga.fit(train_x, train_y)
            y_pred = nn_ga.predict(test_x)
            y_pred_tr = nn_ga.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_256[i,7] = acc_tr
        acc_iters_all_256[i,8] = acc
        print('GA', acc_tr, acc)
    
    acc_res_256 = np.zeros((len(iters_all)+1,9)).astype(object)
    acc_res_256[1:,:] = acc_iters_all_256
    acc_res_256[0,:] = ['iterations', 'GD Train', 'GD Test', 'RHC Train', 'RHC Test', 'SA Train', 'SA Test', 'GA Train', 'GA Test']
    
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.29)
    plt.subplots_adjust(left=.18)
    plt.title('NN 256: Train Acc vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    sns.lineplot(y=acc_iters_all_256[:, 1], x=iters_all, label='Grad Desc')
    sns.lineplot(y=acc_iters_all_256[:, 3], x=iters_all, label='RHC')
    sns.lineplot(y=acc_iters_all_256[:, 5], x=iters_all, label='SA')
    sns.lineplot(y=acc_iters_all_256[:, 7], x=iters_all, label='GA')
    plt.legend(loc='lower center')
    plt.savefig('NN256 Train vs Iter')
    
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.29)
    plt.subplots_adjust(left=.18)
    plt.title('NN 256: Test Acc vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    sns.lineplot(y=acc_iters_all_256[:, 2], x=iters_all, label='Grad Desc')
    sns.lineplot(y=acc_iters_all_256[:, 4], x=iters_all, label='RHC')
    sns.lineplot(y=acc_iters_all_256[:, 6], x=iters_all, label='SA')
    sns.lineplot(y=acc_iters_all_256[:, 8], x=iters_all, label='GA')
    plt.legend(loc='lower center')
    plt.savefig('NN256 Test vs Iter')
    
    
    
    acc_iters_all_512 = np.zeros((len(iters_all),9))
    acc_iters_all_512[:,0] = iters_all
    for i, iter in enumerate(iters_all):
        print(iter)
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 50
            lr = 0.001
            nn_gd = mlrose.NeuralNetwork(hidden_nodes=[512], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='gradient_descent', max_iters=max_iters,
                                         max_attempts=max_attempts, curve=True)
            nn_gd_res = nn_gd.fit(train_x, train_y)
            y_pred = nn_gd.predict(test_x)
            y_pred_tr = nn_gd.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_512[i,1] = acc_tr
        acc_iters_all_512[i,2] = acc
        print('GD', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 10
            restarts = 50
            lr = 0.3
            nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[512], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='random_hill_climb', max_iters=max_iters,
                                          max_attempts=max_attempts, restarts=restarts, curve=True)
            nn_rhc_res = nn_rhc.fit(train_x, train_y)
            y_pred = nn_rhc.predict(test_x)
            y_pred_tr = nn_rhc.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_512[i,3] = acc_tr
        acc_iters_all_512[i,4] = acc
        print('RHC', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 500
            init_temp = 0.01
            min_temp = .00001
            exp_const = 0.001
            lr = 0.3
            schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=0.001)
            nn_sa = mlrose.NeuralNetwork(hidden_nodes=[512], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='simulated_annealing', max_iters=max_iters,
                                         max_attempts=max_attempts, schedule=schedule, curve=True)
            nn_sa_res = nn_sa.fit(train_x, train_y)
            y_pred = nn_sa.predict(test_x)
            y_pred_tr = nn_sa.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_512[i,5] = acc_tr
        acc_iters_all_512[i,6] = acc
        print('SA', acc_tr, acc)
    
        acc = 0
        acc_tr = 0
        for k in range(5):
            max_iters = iter
            max_attempts = 100
            mutation_prob = 0.1
            pop_size = 80
            lr = 0.3
            nn_ga = mlrose.NeuralNetwork(hidden_nodes=[512], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='genetic_alg', max_iters=max_iters,
                                         max_attempts=max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, curve=True)
            nn_ga_res = nn_ga.fit(train_x, train_y)
            y_pred = nn_ga.predict(test_x)
            y_pred_tr = nn_ga.predict(train_x)
            acc += accuracy_score(test_y, y_pred) / 5
            acc_tr += accuracy_score(train_y, y_pred_tr) / 5
        acc_iters_all_512[i,7] = acc_tr
        acc_iters_all_512[i,8] = acc
        print('GA', acc_tr, acc)
    
    acc_res_512 = np.zeros((len(iters_all)+1,9)).astype(object)
    acc_res_512[1:,:] = acc_iters_all_512
    acc_res_512[0,:] = ['iterations', 'GD Train', 'GD Test', 'RHC Train', 'RHC Test', 'SA Train', 'SA Test', 'GA Train', 'GA Test']
    
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.29)
    plt.subplots_adjust(left=.18)
    plt.title('NN 512: Train Acc vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    sns.lineplot(y=acc_iters_all_512[:, 1], x=iters_all, label='Grad Desc')
    sns.lineplot(y=acc_iters_all_512[:, 3], x=iters_all, label='RHC')
    sns.lineplot(y=acc_iters_all_512[:, 5], x=iters_all, label='SA')
    sns.lineplot(y=acc_iters_all_512[:, 7], x=iters_all, label='GA')
    plt.legend(loc='lower center')
    plt.savefig('NN512 Train vs Iter')
    
    sns.set()
    fig, ax = plt.subplots(figsize=(4, 2))
    plt.subplots_adjust(bottom=.29)
    plt.subplots_adjust(left=.18)
    plt.title('NN 512: Test Acc vs Iteration')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    sns.lineplot(y=acc_iters_all_512[:, 2], x=iters_all, label='Grad Desc')
    sns.lineplot(y=acc_iters_all_512[:, 4], x=iters_all, label='RHC')
    sns.lineplot(y=acc_iters_all_512[:, 6], x=iters_all, label='SA')
    sns.lineplot(y=acc_iters_all_512[:, 8], x=iters_all, label='GA')
    plt.legend(loc='lower center')
    plt.savefig('NN512 Test vs Iter')




gd_iter = 1000
rhc_iter = 1000
sa_iter = 10000
ga_iter = 200

time_all_128 = np.zeros((4,5)).astype(object)
time_all_128[0,:] = ['GD 1,000', 'GD 10,000','RHC 1,000', 'SA 10,000', 'GA 200']
avg = 3

start_time = time.time()
for i in range(avg):
    max_iters = 1000
    max_attempts = 50
    lr = 0.001
    nn_gd = mlrose.NeuralNetwork(hidden_nodes=[128], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='gradient_descent', max_iters=max_iters,
                                 max_attempts=max_attempts, curve=True)
    nn_gd_res = nn_gd.fit(train_x, train_y)
    y_pred = nn_gd.predict(test_x)
    y_pred_tr = nn_gd.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all_128[2,0] += acc_tr/avg
    time_all_128[3, 0] += acc/avg
    print('GD', i)
time_all_128[1,0] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 10000
    max_attempts = 50
    lr = 0.001
    nn_gd = mlrose.NeuralNetwork(hidden_nodes=[128], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='gradient_descent', max_iters=max_iters,
                                 max_attempts=max_attempts, curve=True)
    nn_gd_res = nn_gd.fit(train_x, train_y)
    y_pred = nn_gd.predict(test_x)
    y_pred_tr = nn_gd.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all_128[2, 1] += acc_tr /avg
    time_all_128[3, 1] += acc /avg
    print('GD2', i)
time_all_128[1,1] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 1000
    max_attempts = 50
    restarts = 100
    lr = 0.3
    nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[128], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='random_hill_climb', max_iters=max_iters,
                                  max_attempts=max_attempts, restarts=restarts, curve=True)
    nn_rhc_res = nn_rhc.fit(train_x, train_y)
    y_pred = nn_rhc.predict(test_x)
    y_pred_tr = nn_rhc.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all_128[2, 2] += acc_tr /avg
    time_all_128[3, 2] += acc /avg
    print('RHC', i)
time_all_128[1,2] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 10000
    max_attempts = 500
    init_temp = 0.01
    min_temp = .00001
    exp_const = 0.001
    lr = 0.3
    schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=0.001)
    nn_sa = mlrose.NeuralNetwork(hidden_nodes=[128], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='simulated_annealing', max_iters=max_iters,
                                 max_attempts=max_attempts, schedule=schedule, curve=True)
    nn_sa_res = nn_sa.fit(train_x, train_y)
    y_pred = nn_sa.predict(test_x)
    y_pred_tr = nn_sa.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all_128[2, 3] += acc_tr /avg
    time_all_128[3, 3] += acc /avg
    print('SA', i)
time_all_128[1,3] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 200
    max_attempts = 100
    mutation_prob = 0.1
    pop_size = 80
    lr = 0.3
    nn_ga = mlrose.NeuralNetwork(hidden_nodes=[128], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='genetic_alg', max_iters=max_iters,
                                 max_attempts=max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, curve=True)
    nn_ga_res = nn_ga.fit(train_x, train_y)
    y_pred = nn_ga.predict(test_x)
    y_pred_tr = nn_ga.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all_128[2, 4] += acc_tr /avg
    time_all_128[3, 4] += acc /avg
    print('GA', i)
time_all_128[1,4] = (time.time() - start_time)/avg




time_all_512 = np.zeros((4,5)).astype(object)
time_all_512[0,:] = ['GD 1,000', 'GD 10,000','RHC 1,000', 'SA 10,000', 'GA 200']
avg = 3

start_time = time.time()
for i in range(avg):
    max_iters = 1000
    max_attempts = 50
    lr = 0.001
    nn_gd = mlrose.NeuralNetwork(hidden_nodes=[512], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='gradient_descent', max_iters=max_iters,
                                 max_attempts=max_attempts, curve=True)
    nn_gd_res = nn_gd.fit(train_x, train_y)
    y_pred = nn_gd.predict(test_x)
    y_pred_tr = nn_gd.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all_512[2,0] += acc_tr/avg
    time_all_512[3, 0] += acc/avg
    print('GD', i)
time_all_512[1,0] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 10000
    max_attempts = 50
    lr = 0.001
    nn_gd = mlrose.NeuralNetwork(hidden_nodes=[512], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='gradient_descent', max_iters=max_iters,
                                 max_attempts=max_attempts, curve=True)
    nn_gd_res = nn_gd.fit(train_x, train_y)
    y_pred = nn_gd.predict(test_x)
    y_pred_tr = nn_gd.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all_512[2, 1] += acc_tr /avg
    time_all_512[3, 1] += acc /avg
    print('GD2', i)
time_all_512[1,1] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 1000
    max_attempts = 50
    restarts = 100
    lr = 0.3
    nn_rhc = mlrose.NeuralNetwork(hidden_nodes=[512], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='random_hill_climb', max_iters=max_iters,
                                  max_attempts=max_attempts, restarts=restarts, curve=True)
    nn_rhc_res = nn_rhc.fit(train_x, train_y)
    y_pred = nn_rhc.predict(test_x)
    y_pred_tr = nn_rhc.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all_512[2, 2] += acc_tr /avg
    time_all_512[3, 2] += acc /avg
    print('RHC', i)
time_all_512[1,2] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 10000
    max_attempts = 500
    init_temp = 0.01
    min_temp = .00001
    exp_const = 0.001
    lr = 0.3
    schedule = mlrose.ExpDecay(init_temp=init_temp, min_temp=min_temp, exp_const=0.001)
    nn_sa = mlrose.NeuralNetwork(hidden_nodes=[512], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='simulated_annealing', max_iters=max_iters,
                                 max_attempts=max_attempts, schedule=schedule, curve=True)
    nn_sa_res = nn_sa.fit(train_x, train_y)
    y_pred = nn_sa.predict(test_x)
    y_pred_tr = nn_sa.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all_512[2, 3] += acc_tr /avg
    time_all_512[3, 3] += acc /avg
    print('SA', i)
time_all_512[1,3] = (time.time() - start_time)/avg

start_time = time.time()
for i in range(avg):
    max_iters = 200
    max_attempts = 100
    mutation_prob = 0.1
    pop_size = 80
    lr = 0.3
    nn_ga = mlrose.NeuralNetwork(hidden_nodes=[512], activation='relu', bias=True, is_classifier=True, learning_rate=lr, early_stopping=True, algorithm='genetic_alg', max_iters=max_iters,
                                 max_attempts=max_attempts, mutation_prob=mutation_prob, pop_size=pop_size, curve=True)
    nn_ga_res = nn_ga.fit(train_x, train_y)
    y_pred = nn_ga.predict(test_x)
    y_pred_tr = nn_ga.predict(train_x)
    acc = accuracy_score(test_y, y_pred)
    acc_tr = accuracy_score(train_y, y_pred_tr)
    time_all_512[2, 4] += acc_tr /avg
    time_all_512[3, 4] += acc /avg
    print('GA', i)
time_all_512[1,4] = (time.time() - start_time)/avg


fig, ax = plt.subplots(figsize=(8, 2))
ax2 = ax.twinx()
ax2.grid(False)
ax.grid(False)
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('NN Hidden Layer Size 128: Clock Time and Accuracy')
plt.xlabel('Algorithm + Iterations')
ax.set_ylabel('Clock Time (Sec)')
ax2.set_ylabel('Accuracy')
k = np.arange(5)
ax.bar(k, time_all_128[1,:], 0.3)
ax2.set_ylim(bottom=0,top=1)
ax.set_ylim([0,max(time_all_128[1,:])*2])
ax2.scatter(k, time_all_128[3,:], color='red')
ax2.scatter(k, time_all_128[2,:], color='orange')
plt.xticks(k,['GD 1,000', 'GD 10,000','RHC 1,000', 'SA 10,000', 'GA 200'])
for xy in zip(k[:2], time_all_128[2,:2]):
    t = round(xy[1], 3)
    ax2.annotate('Train: %s' % t, xy=xy, textcoords='data', fontsize=10, ha='center', verticalalignment='top')
for xy in zip(k[2:], time_all_128[2,2:]):
    t = round(xy[1], 3)
    ax2.annotate('Train: %s' % t, xy=xy, textcoords='data', fontsize=10, ha='center', verticalalignment='bottom')
for xy in zip(k, time_all_128[3, :]):
    t = round(xy[1], 3)
    ax2.annotate('Test: %s' % t, xy=xy, textcoords='data', fontsize=10, ha='center', verticalalignment='top')
for xy in zip(k, time_all_128[1,:]):
    t = round(xy[1], 2)
    ax.annotate('%s sec' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
plt.savefig('NN128 clock and accuracy')


fig, ax = plt.subplots(figsize=(8, 2))
ax2 = ax.twinx()
ax2.grid(False)
ax.grid(False)
plt.subplots_adjust(bottom=.26)
plt.subplots_adjust(left=.16)
plt.title('NN Hidden Layer Size 512: Clock Time and Accuracy')
plt.xlabel('Algorithm + Iterations')
ax.set_ylabel('Clock Time (Sec)')
ax2.set_ylabel('Accuracy')
k = np.arange(5)
ax.bar(k, time_all_512[1,:], 0.3)
ax2.set_ylim(bottom=0,top=1)
ax.set_ylim([0,max(time_all_512[1,:])*2.3])
ax2.scatter(k, time_all_512[3,:], color='red')
ax2.scatter(k, time_all_512[2,:], color='orange')
plt.xticks(k,['GD 1,000', 'GD 10,000','RHC 1,000', 'SA 10,000', 'GA 200'])
for xy in zip(k[:2], time_all_512[2,:2]):
    t = round(xy[1], 3)
    ax2.annotate('Train: %s' % t, xy=xy, textcoords='data', fontsize=10, ha='center', verticalalignment='top')
for xy in zip(k[2:], time_all_512[2,2:]):
    t = round(xy[1], 3)
    ax2.annotate('Train: %s' % t, xy=xy, textcoords='data', fontsize=10, ha='center', verticalalignment='bottom')
for xy in zip(k, time_all_512[3, :]):
    t = round(xy[1], 3)
    ax2.annotate('Test: %s' % t, xy=xy, textcoords='data', fontsize=10, ha='center', verticalalignment='top')
for xy in zip(k, time_all_512[1,:]):
    t = round(xy[1], 2)
    ax.annotate('%s sec' % t, xy=xy, textcoords='data', fontsize=10, ha='center')
plt.savefig('NN512 clock and accuracy')

# fitness1arr = [7,7,7,7,6,5,4,3,2,1,7,7,7,7,6,5,4,3,2,1,7,7,7,6,6,5,4,3,2,1,7,7,6,6,6,5,4,3,2,1,6,6,6,6,5,4,3,3,2,1,5,5,
#                5,5,4,3,2,2,1,1,4,4,4,4,3,2,1,1,1,4,3,3,3,3,3,2,1,1,4,6,2,2,2,2,2,1,1,4,6,8,1,1,1,1,1,1,4,6,8,10]
# fitness1arr = np.array(fitness1arr).reshape(10,10)
#
# fitness2arr = [10,10,9,8,6,5,4,3,2,1,10,9,8,7,6,5,4,3,2,1,9,8,7,7,6,5,4,3,2,1,8,7,7,6,6,5,4,3,2,1,6,6,6,6,5,4,3,3,2,1,5,
#                5,5,5,4,3,2,2,1,1,4,4,4,4,3,2,1,1,1,3,3,3,3,3,3,2,1,1,3,5,2,2,2,2,2,1,1,3,5,7,1,1,1,1,1,1,3,5,7,7]
# fitness2arr = np.array(fitness2arr).reshape(10,10)
#
# fitness3arr = [5,4,3,2,1,1,2,3,4,5,4,4,3,2,1,1,2,3,4,4,3,3,3,2,1,1,2,3,3,3,2,2,2,2,1,1,2,2,2,2,1,1,1,1,1,1,1,1,1,1,1,1,
#                1,1,1,1,1,1,1,1,2,2,2,2,1,1,2,2,2,2,3,3,3,2,1,1,2,3,3,3,4,4,3,2,1,1,2,3,4,4,5,4,3,2,1,1,2,3,4,5]
# fitness3arr = np.array(fitness3arr).reshape(10,10)
#
# def fitness1(x,y):
#     return fitness1arr[x,:][range(len(x)),y]
#
# def fitness2(x,y):
#     return fitness2arr[x,:][range(len(x)),y]
#
# def fitness3(x,y):
#     time.sleep(5/1000)
#     return fitness3arr[x,:][range(len(x)),y]
#
#
# gen_size = 15
# mutation = 100
# # def genetic_algorithm(generations=50, gen_size = 300, mutation = 100, fitness = fitness1):
# #     gen_x = np.random.randint(0,1000,gen_size)
# #     gen_y = np.random.randint(0,1000,gen_size)
# #
# #     for i in range(generations):
# #
# #         x_mini = (gen_x/100).astype(int)
# #         y_mini = (gen_y/100).astype(int)
# #
# #         fitnesses = fitness(x_mini, y_mini)
# #         survivors = np.random.choice(gen_size,size=int(gen_size*2/3), replace=False, p=fitnesses/sum(fitnesses))
# #         gen_split = np.array_split(survivors,2)
# #         male = gen_split[0]
# #         female = gen_split[1]
# #
# #         if True:
# #             male_x = gen_x[male]
# #             male_y = gen_y[male]
# #             female_x = gen_x[female]
# #             female_y = gen_y[female]
# #
# #             male_x += np.random.normal(0,mutation,int(gen_size/3)).astype(int)
# #             male_x = np.maximum(male_x,0)
# #             male_x = np.minimum(male_x, 999)
# #
# #             male_y += np.random.normal(0, mutation, int(gen_size / 3)).astype(int)
# #             male_y = np.maximum(male_y, 0)
# #             male_y = np.minimum(male_y, 999)
# #
# #             female_x += np.random.normal(0, mutation, int(gen_size / 3)).astype(int)
# #             female_x = np.maximum(female_x, 0)
# #             female_x = np.minimum(female_x, 999)
# #
# #             female_y += np.random.normal(0, mutation, int(gen_size / 3)).astype(int)
# #             female_y = np.maximum(female_y, 0)
# #             female_y = np.minimum(female_y, 999)
# #
# #         gen_x = np.concatenate([male_x,female_x,((male_x+female_x)/2).astype(int)])
# #         gen_y = np.concatenate([female_y,male_y,((male_y+female_y)/2).astype(int)])
# #         print(i)
# #
# #     x_mini = (gen_x / 100).astype(int)
# #     y_mini = (gen_y / 100).astype(int)
# #     fitnesses = fitness1(x_mini, y_mini)
# #
# #     return (gen_x, gen_y, fitnesses)
# def genetic_algorithm(generations=50, gen_size = 9, mutation = 1, fitness = fitness1):
#     gen_x = np.random.randint(0,10,gen_size)
#     gen_y = np.random.randint(0,10,gen_size)
#
#     for i in range(generations):
#
#         x_mini = (gen_x).astype(int)
#         y_mini = (gen_y).astype(int)
#
#         fitnesses = fitness(x_mini, y_mini)
#         survivors = np.argsort(fitnesses)[::-1][0:int(gen_size*2/3)]
#         gen_split = np.array_split(survivors,2)
#         male = gen_split[0]
#         female = gen_split[1]
#
#         if True:
#             male_x = gen_x[male]
#             male_y = gen_y[male]
#             female_x = gen_x[female]
#             female_y = gen_y[female]
#
#             male_x += np.random.normal(0,mutation,int(gen_size/3)).astype(int)
#             male_x = np.maximum(male_x,0)
#             male_x = np.minimum(male_x, 9)
#
#             male_y += np.random.normal(0, mutation, int(gen_size / 3)).astype(int)
#             male_y = np.maximum(male_y, 0)
#             male_y = np.minimum(male_y, 9)
#
#             female_x += np.random.normal(0, mutation, int(gen_size / 3)).astype(int)
#             female_x = np.maximum(female_x, 0)
#             female_x = np.minimum(female_x, 9)
#
#             female_y += np.random.normal(0, mutation, int(gen_size / 3)).astype(int)
#             female_y = np.maximum(female_y, 0)
#             female_y = np.minimum(female_y, 9)
#
#         gen_x = np.concatenate([male_x,female_x,((male_x+female_x)/2).astype(int)])
#         gen_y = np.concatenate([female_y,male_y,((male_y+female_y)/2).astype(int)])
#         print(i)
#
#     x_mini = (gen_x).astype(int)
#     y_mini = (gen_y).astype(int)
#     fitnesses = fitness(x_mini, y_mini)
#
#     return np.array([gen_x, gen_y, fitnesses]).reshape((3,gen_size))
#
# fit1 = genetic_algorithm(fitness=fitness1)
# fit2 = genetic_algorithm(fitness=fitness2)
# fit3 = genetic_algorithm(fitness=fitness3)
#
# def simulan(generations=50, gen_size=9, jump_sd = 100, fitness=fitness1, temp=2):
#     gen_x = np.random.randint(0, 1000, gen_size)
#     gen_y = np.random.randint(0, 1000, gen_size)
#
#     for i in range(generations):
#
#         x_mini = (gen_x / 100).astype(int)
#         y_mini = (gen_y / 100).astype(int)
#
#         fitnesses = fitness(x_mini, y_mini)
#
#         x_jump = gen_x + np.random.normal(0, jump_sd, int(gen_size)).astype(int)
#         x_jump = np.maximum(x_jump, 0)
#         x_jump = np.minimum(x_jump, 999)
#
#         y_jump = gen_y + np.random.normal(0, jump_sd, int(gen_size)).astype(int)
#         y_jump = np.maximum(y_jump, 0)
#         y_jump = np.minimum(y_jump, 999)
#
#         x_mini = (x_jump / 100).astype(int)
#         y_mini = (y_jump / 100).astype(int)
#
#         fitnesses_jump = fitness(x_mini, y_mini)
#         prob =
#
#         gen_split = np.array_split(survivors, 2)
#         male = gen_split[0]
#         female = gen_split[1]
#
#         if True:
#             male_x = gen_x[male]
#             male_y = gen_y[male]
#             female_x = gen_x[female]
#             female_y = gen_y[female]
#
#             male_x += np.random.normal(0, mutation, int(gen_size / 3)).astype(int)
#             male_x = np.maximum(male_x, 0)
#             male_x = np.minimum(male_x, 999)
#
#             male_y += np.random.normal(0, mutation, int(gen_size / 3)).astype(int)
#             male_y = np.maximum(male_y, 0)
#             male_y = np.minimum(male_y, 999)
#
#             female_x += np.random.normal(0, mutation, int(gen_size / 3)).astype(int)
#             female_x = np.maximum(female_x, 0)
#             female_x = np.minimum(female_x, 999)
#
#             female_y += np.random.normal(0, mutation, int(gen_size / 3)).astype(int)
#             female_y = np.maximum(female_y, 0)
#             female_y = np.minimum(female_y, 999)
#
#         gen_x = np.concatenate([male_x, female_x, ((male_x + female_x) / 2).astype(int)])
#         gen_y = np.concatenate([female_y, male_y, ((male_y + female_y) / 2).astype(int)])
#         print(i)
#
#     x_mini = (gen_x / 100).astype(int)
#     y_mini = (gen_y / 100).astype(int)
#     fitnesses = fitness(x_mini, y_mini)
#
#     return np.array([gen_x, gen_y, fitnesses]).reshape((3, gen_size))
#
#
# test = fitness1arr[x_mini,:]
# test1 = test[range(100),y_mini]
# # Press the green button in the gutter to run the script.
#
# # def fitness1(x,y):
# #     return np.square(np.square(x) + y - 11) + np.square(x + np.square(y)-7)
# #
# # def genetic_algorithm(gen_size = 200, mutation = 1, fitness = fitness1):
# #     gen_x = np.random.random(gen_size)
# #     gen_x = gen_x*10-5
# #     gen_y = np.random.random(gen_size)
# #     gen_y = gen_y * 10 - 5
# #
# #     fitnesses = fitness1(gen_x, gen_y)
# #     fitnesses = 1/ (1+fitnesses)
# #     next_gen = np.random.choice(gen_size,size=int(gen_size/2), replace=False, p=fitnesses/sum(fitnesses))
# #     gen_split = np.array_split(next_gen,2)
# #     male = gen_split[0]
# #     female = gen_split[1]
# #
# #     male_x = gen_x[male]
# #     male_y = gen_y[male]
# #     female_x = gen_x[female]
# #     female_y = gen_y
# #

if __name__ == '__main__':
    print('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/

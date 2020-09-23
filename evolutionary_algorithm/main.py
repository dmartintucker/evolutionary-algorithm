
# Global dependencies
import sys
import random
import numpy as np
import pandas as pd
from func_timeout import func_timeout, FunctionTimedOut
from tqdm import tqdm

###############################################################################
###############################################################################
###############################################################################


class EvolutionaryAlgorithm():

    ###############################################################################

    '''
    @param function <Callable>: a callable function whose output is to be minimized
    @param parameters <list>: a list of dictionary input parameters, with each dictionary specifying the name, bounds, and type of the parameter
    @param function_time <int>: the maximum number of seconds to wait for an input function to return an output
    @param algorithm_parameters <dict>: a set of parameters to be passed to the evolutionary algorithm (not to the function to be minimized), which control the evolutionary algorithm's behavior
    '''

    ###############################################################################

    # Initialize class object
    def __init__(self, function, parameters, function_timeout=10,
                 algorithm_parameters={'max_num_iteration': None,
                                       'population_size': 100,
                                       'mutation_probability': 0.1,
                                       'elite_ratio': 0.1,
                                       'crossover_probability': 0.5,
                                       'parents_portion': 0.2,
                                       'crossover_type': 'uniform',
                                       'max_iteration_without_improv': None}):

        # Declare class object name attribute
        self.__name__ = EvolutionaryAlgorithm

        # Declare function reference
        self.f = function

        # Set number of dimensions to be optimized over
        self.dim = int(len(list(parameters)))

        # Set input variable names
        self.var_names = [[p['name']] for p in parameters]

        # Check that parameters object is type dict
        assert(type(parameters) ==
               list), "Error: argument parameters must be a list"

        # Validate input: Check that each item in parameters is a dictionary
        for p in parameters:
            assert(type(p) ==
                   dict), "Error: parameters object must contain only dictionaries"

        # Validate input: Check each input parameter for expected type
        for p in parameters:
            assert(p['type'] in ['bool', 'int', 'real', 'cat']
                   ), "Error: unknown parameter type '{}'".format(p['type'])

        # Validate input: All parameters must have bound and types
        for p in parameters:
            assert(
                p['bounds']), "\nError: every parameter item must have bounds"
            assert(
                p['type']), "\nError: every parameter item must have an explicit type"

            if p['type'] == 'bool':
                assert(
                    p['bounds'] == [0, 1]), "\nError: type 'bool' can only have bounds [0, 1]"

            if p['type'] == 'cat':
                for k in p['bounds']:
                    assert(
                        type(k) == str), "\nError: type 'cat' must have strings as bounds"

        # Create variable bounds object
        self.var_bound = np.array([[x for x in p['bounds']]
                                   for p in parameters], dtype=object)

        # Variable type declaration
        self.var_type = np.array([[p['type']] for p in parameters])

        # Set function timeout
        self.funtimeout = float(function_timeout)

        # Set evolutionary algorithm parameters
        self.param = algorithm_parameters

        # Set initial population size
        self.pop_s = int(self.param['population_size'])

        # Validate input: parent proportion must be between 0 and 1
        assert (0 <= self.param['parents_portion'] <=
                1), "\nError: argument 'parents_portion' must be in range [0,1]"

        # Select initial number of parents
        self.par_s = int(self.param['parents_portion'] * self.pop_s)
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s += 1

        # Set mutation probability
        self.prob_mut = self.param['mutation_probability']

        # Validate input: mutation probability rate must be between 0 and 1
        assert (self.prob_mut <= 1 and self.prob_mut >=
                0), "\nError: parameter 'mutation_probability' must be in range [0,1]"

        # Set & validate crossover rate probability
        self.prob_cross = self.param['crossover_probability']
        assert (self.prob_cross <= 1 and self.prob_cross >=
                0), "\nError: parameter 'crossover_probability' must be in range [0,1]"

        # Set & validate elite ratio
        assert (self.param['elite_ratio'] <= 1 and self.param['elite_ratio'] >= 0),\
            "\nError: parameter 'elite_ratio' must be in range [0,1]"
        trl = self.pop_s * self.param['elite_ratio']
        if trl < 1 and self.param['elite_ratio'] > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)
        assert(self.par_s >= self.num_elit), "\nError: number of parents must be greater than number of generational elites"

        # Set max number of iterations
        if self.param['max_num_iteration'] == None:
            self.iterate = 10
        else:
            self.iterate = int(self.param['max_num_iteration'])

        # Set crossover type
        self.c_type = self.param['crossover_type']
        assert (self.c_type == 'uniform' or self.c_type == 'one_point' or
                self.c_type == 'two_point'),\
            "\nError: parameter 'crossover_type' must be either 'uniform', 'one_point' or 'two_point'"

        # Set early stopping threshold
        self.stop_mniwi = False
        if self.param['max_iteration_without_improv'] == None:
            self.mniwi = self.iterate+1
        else:
            self.mniwi = int(self.param['max_iteration_without_improv'])

    ###############################################################################

    def run(self):

        pop = []
        # solo = np.zeros(self.dim+1)
        var = np.zeros(self.dim)

        for p in range(0, self.pop_s):
            vars = {self.var_names[i][0]: np.nan for i in range(len(self.var_names))}

            for i in range(len(self.var_names)):

                if self.var_type[i][0] == 'int':
                    val = np.random.randint(
                        self.var_bound[i][0], self.var_bound[i][1]+1)
                    vars[self.var_names[i][0]] = val
                    # solo[i] = val

                elif self.var_type[i][0] == 'real':
                    val = self.var_bound[i][0]+np.random.random() * \
                        (self.var_bound[i][1]-self.var_bound[i][0])
                    vars[self.var_names[i][0]] = val
                    # solo[i] = self.var_bound[i][0]+np.random.random() * \
                    #     (self.var_bound[i][1]-self.var_bound[i][0])

                elif self.var_type[i][0] == 'bool':
                    val = random.choice(self.var_bound[i])
                    vars[self.var_names[i][0]] = val
                    #solo[i] = val

                elif self.var_type[i][0] == 'cat':
                    val = random.choice(self.var_bound[i])
                    vars[self.var_names[i][0]] = val
                    #solo[i] = val

            # sys.stdout.write('\n' + str(vars))

            obj = self.sim(vars)
            vars['OBJ'] = obj
            pop.append(vars)

        self.report = []
        self.test_obj = obj
        self.best_variable = {i[0]: vars[i[0]] for i in self.var_names}
        self.best_function = obj
        # sys.stdout.write('\n' + str(self.best_variable))

        t = 1
        counter = 0
        for t in tqdm(range(self.iterate)):

            # Sort population by fitness, ascending
            pop = sorted(pop, key=lambda k: k['OBJ'], reverse=False)

            if pop[0]['OBJ'] < self.best_function:
                counter = 0
                self.best_function = pop[0]['OBJ']
                self.best_variable = {i[0]: pop[0][i[0]]
                                      for i in self.var_names}
            else:
                counter += 1

            self.report.append(pop[0]['OBJ'])

            normobj = np.zeros(self.pop_s)
            minobj = pop[0]['OBJ']
            if minobj < 0:
                normobj = pop[0]['OBJ'] + abs(minobj)
            else:
                normobj = pop[0]['OBJ'].copy()

            maxnorm = np.amax(normobj)
            normobj = maxnorm-normobj+1

            sum_normobj = np.sum(normobj)
            prob = np.zeros(self.pop_s)
            prob = normobj/sum_normobj
            cumprob = np.cumsum(prob)

            par = []

            for k in range(0, self.num_elit):
                par.append(pop[k].copy())
            for k in range(self.num_elit, self.par_s):
                index = np.searchsorted(cumprob, np.random.random())
                par.append(pop[index].copy())

            ef_par_list = []
            par_count = 0
            while par_count == 0:
                for k in range(0, self.par_s):
                    if np.random.random() <= self.prob_cross:
                        ef_par_list.append(k)
                        par_count += 1

            ef_par = [par[i] for i in ef_par_list]

            pop = []
            for k in range(0, self.par_s):
                pop.append(par[k])

            for k in range(self.par_s, self.pop_s, 2):
                r1 = np.random.randint(0, par_count)
                r2 = np.random.randint(0, par_count)
                pvar1 = ef_par[r1]
                pvar2 = ef_par[r2]

            ch = self.cross(pvar1, pvar2, self.c_type)
            ch1 = ch[0]
            ch2 = ch[1]
            # sys.stdout.write('\n' + str(ch))

            ch1 = self.mut(ch1)
            ch2 = self.mutmidle(ch2, pvar1, pvar2)
            obj = self.sim(ch1)
            ch1['OBJ'] = obj
            pop.append(ch1)
            obj = self.sim(ch2)
            ch2['OBJ'] = obj
            pop.append(ch2)

            t += 1

            if counter > self.mniwi:
                pop = sorted(pop, key=lambda k: k['OBJ'], reverse=False)

                if pop[0]['OBJ'] >= self.best_function:
                    t = self.iterate
                    t += 1
                    self.stop_mniwi = True

        pop = sorted(pop, key=lambda k: k['OBJ'], reverse=False)

        if pop[0]['OBJ'] < self.best_function:

            self.best_function = pop[0]['OBJ'].copy()
            self.best_variable = {i[0]: pop[i[0]] for i in self.var_names}

        self.report.append(pop[0]['OBJ'])
        self.output_dict = {'variable': self.best_variable, 'function':
                            self.best_function}
        self.best_params = self.best_variable

        # Write final results to stdout
        sys.stdout.flush()
        sys.stdout.write(
            'Best parameters found: {}'.format(str(self.best_params)))
        sys.stdout.flush()
        sys.stdout.write('\nBest objective output = {}'.format(
            str(self.best_function)))
        sys.stdout.flush()
        if self.stop_mniwi == True:
            sys.stdout.write(
                '\nTerminating algorithm: Exceeded maximum iterations without improvement.')

    ##############################################################################

    def cross(self, x, y, c_type):

        ofs1 = {i[0]: x[i[0]] for i in self.var_names}
        ofs2 = {i[0]: y[i[0]] for i in self.var_names}

        if c_type == 'one_point':
            ran = np.random.randint(0, self.dim)

            for i in range(0, ran):
                ofs1[i[0]] = y[i[0]]
                ofs2[i[0]] = x[i[0]]

        if c_type == 'two_point':
            ran1 = np.random.randint(0, self.dim)
            ran2 = np.random.randint(ran1, self.dim)

            for i in range(ran1, ran2):
                ofs1[i[0]] = y[i[0]]
                ofs2[i[0]] = x[i[0]]

        if c_type == 'uniform':
            for i in self.var_names:
                ran = np.random.random()

                if ran < 0.5:
                    ofs1[i[0]] = y[i[0]]
                    ofs2[i[0]] = x[i[0]]

        return [ofs1, ofs2]

    ###############################################################################

    def mut(self, x):
        var = list(x)

        for i in range(len(self.var_type)):

            if self.var_type[i][0] == 'int':
                ran = np.random.random()

                if ran < self.prob_mut:
                    x[var[i]] = np.random.randint(self.var_bound[i][0],
                                                  self.var_bound[i][1]+1)

                elif self.var_type[i][0] == 'real':
                    x[var[i]] = self.var_bound[i][0]+np.random.random() *\
                        (self.var_bound[i][1]-self.var_bound[i][0])

                elif self.var_type[i][0] == 'bool':
                    x[var[i]] = random.choice(self.var_bound[i])

                elif self.var_type[i][0] == 'cat':
                    x[var[i]] = random.choice(self.var_bound[i])

        return x

    ###############################################################################

    def mutmidle(self, x, p1, p2):

        var = list(x)

        for i in range(len(self.var_type)):
            ran = np.random.random()

            # Integer vars
            if self.var_type[i][0] == 'int':
                if ran < self.prob_mut:
                    if p1[var[i]] < p2[var[i]]:
                        x[var[i]] = np.random.randint(p1[var[i]], p2[var[i]])
                    elif p1[var[i]] > p2[var[i]]:
                        x[var[i]] = np.random.randint(p2[var[i]], p1[var[i]])
                    else:
                        x[var[i]] = np.random.randint(self.var_bound[i][0],
                                                      self.var_bound[i][1]+1)

            # Float vars
            elif self.var_type[i][0] == 'real':
                if ran < self.prob_mut:
                    if p1[var[i]] < p2[var[i]]:
                        x[var[i]] = p1[var[i]]+np.random.random() *\
                            (p2[var[i]]-p1[var[i]])
                    elif p1[var[i]] > p2[var[i]]:
                        x[var[i]] = p2[var[i]]+np.random.random() *\
                            (p1[var[i]]-p2[var[i]])
                    else:
                        x[var[i]] = self.var_bound[i][0]+np.random.random() *\
                            (self.var_bound[i][1]-self.var_bound[i][0])

            # Boolean vars
            elif self.var_type[i][0] == 'bool':
                if ran < self.prob_mut:
                    if p1[var[i]] < p2[var[i]]:
                        x[var[i]] = p1[var[i]]+np.random.random() *\
                            (p2[var[i]]-p1[var[i]])
                    elif p1[var[i]] > p2[var[i]]:
                        x[var[i]] = p2[var[i]]+np.random.random() *\
                            (p1[var[i]]-p2[var[i]])
                    else:
                        x[var[i]] = random.choice(self.var_bound[i])

            # Categorical vars
            elif self.var_type[i][0] == 'cat':
                if ran < self.prob_mut:
                    x[var[i]] = random.choice(self.var_bound[i])

        return x

    ###############################################################################

    def evaluate(self):
        return self.f(self.temp)

    ###############################################################################

    def sim(self, X):
        self.temp = X.copy()
        obj = None

        try:
            obj = func_timeout(self.funtimeout, self.evaluate)

        except FunctionTimedOut:
            print("given function is not applicable")

        assert (obj != None), "Objective function failed to provide output within {} seconds".format(
            str(self.funtimeout))

        return obj

###############################################################################
###############################################################################
###############################################################################

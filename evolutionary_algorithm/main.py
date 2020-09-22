
# Global dependencies
import sys
import time
import random
import numpy as np
from func_timeout import func_timeout, FunctionTimedOut

# Visualization dependencies
import matplotlib.pyplot as plt

###############################################################################
###############################################################################
###############################################################################


class EvolutionaryAlgorithm():

    ###############################################################################

    def __init__(self, function, parameters, function_timeout=10,
                 algorithm_parameters={'max_num_iteration': None,
                                       'population_size': 100,
                                       'mutation_probability': 0.1,
                                       'elite_ratio': 0.01,
                                       'crossover_probability': 0.5,
                                       'parents_portion': 0.3,
                                       'crossover_type': 'uniform',
                                       'max_iteration_without_improv': None}):

        self.__name__ = EvolutionaryAlgorithm
        self.f = function
        self.dim = int(len(list(parameters)))
        self.var_names = [[p['name']] for p in parameters]
        # sys.stdout.write(str(self.var_names))

        # Check that parameters object is type dict
        assert(type(parameters) ==
               list), "Error: argument parameters must be a list"

        # Validate input: Check that each item in parameters is a dictionary
        for p in parameters:
            assert(type(p) ==
                   dict), "Error: parameters object must contain only dictionaries"

        # Validate input: Check each input parameter for expected type
        for p in parameters:
            assert(p['type'] in ['bool', 'int', 'real']
                   ), "Error: unknown parameter type '{}'".format(p['type'])

        for p in parameters:
            assert(
                p['bounds']), "\nError: every parameter item must have bounds"
            assert(
                p['type']), "\nError: every parameter item must have an explicit type"

            if p['type'] == 'bool':
                assert(
                    p['bounds'] == [0, 1]), "\nError: type 'bool' can only have bounds [0, 1]"

        # Create variable bounds object
        self.var_bound = np.array([[x for x in p['bounds']]
                                   for p in parameters])

        # Variable type declaration
        self.var_type = np.array([[p['type']] for p in parameters])

        self.funtimeout = float(function_timeout)
        self.param = algorithm_parameters

        self.pop_s = int(self.param['population_size'])

        assert (0 <= self.param['parents_portion'] <=
                1), "\nError: argument 'parents_portion' must be in range [0,1]"

        self.par_s = int(self.param['parents_portion'] * self.pop_s)
        trl = self.pop_s - self.par_s
        if trl % 2 != 0:
            self.par_s += 1

        self.prob_mut = self.param['mutation_probability']

        assert (self.prob_mut <= 1 and self.prob_mut >=
                0), "\nError: parameter 'mutation_probability' must be in range [0,1]"

        self.prob_cross = self.param['crossover_probability']
        assert (self.prob_cross <= 1 and self.prob_cross >=
                0), "\nError: parameter 'crossover_probability' must be in range [0,1]"

        assert (self.param['elite_ratio'] <= 1 and self.param['elite_ratio'] >= 0),\
            "\nError: parameter 'elite_ratio' must be in range [0,1]"

        trl = self.pop_s * self.param['elite_ratio']
        if trl < 1 and self.param['elite_ratio'] > 0:
            self.num_elit = 1
        else:
            self.num_elit = int(trl)

        assert(self.par_s >= self.num_elit), "\nError: number of parents must be greater than number of generation elites"

        if self.param['max_num_iteration'] == None:
            self.iterate = 0
            for i in range(0, self.dim):
                if self.var_type[i] == 'int':
                    self.iterate += (self.var_bound[i][1] -
                                     self.var_bound[i][0])*self.dim*(100/self.pop_s)
                else:
                    self.iterate += (self.var_bound[i][1] -
                                     self.var_bound[i][0])*50*(100/self.pop_s)
            self.iterate = int(self.iterate)
            if (self.iterate*self.pop_s) > 10000000:
                self.iterate = 10000000/self.pop_s
            sys.stdout.write(
                "\nMaximum iterations set to {}".format(str(self.iterate)))
        else:
            self.iterate = int(self.param['max_num_iteration'])

        self.c_type = self.param['crossover_type']
        assert (self.c_type == 'uniform' or self.c_type == 'one_point' or
                self.c_type == 'two_point'),\
            "\nError: parameter 'crossover_type' must be either 'uniform', 'one_point' or 'two_point'"

        self.stop_mniwi = False
        if self.param['max_iteration_without_improv'] == None:
            self.mniwi = self.iterate+1
        else:
            self.mniwi = int(self.param['max_iteration_without_improv'])

    ###############################################################################

    def run(self):

        pop = np.array([np.zeros(self.dim+1)]*self.pop_s)
        solo = np.zeros(self.dim+1)
        var = np.zeros(self.dim)

        for p in range(0, self.pop_s):

            for i in range(len(self.var_type)):
                if self.var_type[i][0] == 'int':
                    var[i] = np.random.randint(self.var_bound[i][0],
                                               self.var_bound[i][1]+1)
                    solo[i] = var[i].copy()
                elif self.var_type[i][0] == 'real':
                    var[i] = self.var_bound[i][0]+np.random.random() *\
                        (self.var_bound[i][1]-self.var_bound[i][0])
                    solo[i] = var[i].copy()
                else:
                    var[i] = random.choice(self.var_bound[i])
                    solo[i] = var[i].copy()

            vars = {self.var_names[j][0]: var[j]
                    for j in range(len(var))}  # CHANGED
            # sys.stdout.write(str(vars))

            obj = self.sim(vars)
            solo[self.dim] = obj
            pop[p] = solo.copy()

        self.report = []
        self.test_obj = obj
        self.best_variable = var.copy()
        self.best_function = obj

        t = 1
        counter = 0
        while t <= self.iterate:

            self.progress(t, self.iterate, status="Running algorithm...")

            # Sort
            pop = pop[pop[:, self.dim].argsort()]

            if pop[0, self.dim] < self.best_function:
                counter = 0
                self.best_function = pop[0, self.dim].copy()
                self.best_variable = pop[0, : self.dim].copy()
            else:
                counter += 1

            self.report.append(pop[0, self.dim])

            normobj = np.zeros(self.pop_s)

            minobj = pop[0, self.dim]
            if minobj < 0:
                normobj = pop[:, self.dim]+abs(minobj)

            else:
                normobj = pop[:, self.dim].copy()

            maxnorm = np.amax(normobj)
            normobj = maxnorm-normobj+1

            sum_normobj = np.sum(normobj)
            prob = np.zeros(self.pop_s)
            prob = normobj/sum_normobj
            cumprob = np.cumsum(prob)

            par = np.array([np.zeros(self.dim+1)]*self.par_s)

            for k in range(0, self.num_elit):
                par[k] = pop[k].copy()
            for k in range(self.num_elit, self.par_s):
                index = np.searchsorted(cumprob, np.random.random())
                par[k] = pop[index].copy()

            ef_par_list = np.array([False]*self.par_s)
            par_count = 0
            while par_count == 0:
                for k in range(0, self.par_s):
                    if np.random.random() <= self.prob_cross:
                        ef_par_list[k] = True
                        par_count += 1

            ef_par = par[ef_par_list].copy()

            pop = np.array([np.zeros(self.dim+1)]*self.pop_s)

            for k in range(0, self.par_s):
                pop[k] = par[k].copy()

            for k in range(self.par_s, self.pop_s, 2):
                r1 = np.random.randint(0, par_count)
                r2 = np.random.randint(0, par_count)
                pvar1 = ef_par[r1, : self.dim].copy()
                pvar2 = ef_par[r2, : self.dim].copy()

                ch = self.cross(pvar1, pvar2, self.c_type)
                #sys.stdout.write(str(ch)) #####
                ch1 = ch[0].copy()
                ch2 = ch[1].copy()

                ch1 = self.mut(ch1)
                ch2 = self.mutmidle(ch2, pvar1, pvar2)
                solo[: self.dim] = ch1.copy()
                ch1_dict = {self.var_names[j][0]: var[j]
                            for j in range(len(ch1))}
                obj = self.sim(ch1_dict)
                solo[self.dim] = obj
                pop[k] = solo.copy()
                solo[: self.dim] = ch2.copy()
                ch2_dict = {self.var_names[j][0]: var[j]
                            for j in range(len(ch2))}
                obj = self.sim(ch2_dict)
                solo[self.dim] = obj
                pop[k+1] = solo.copy()

            t += 1
            if counter > self.mniwi:
                pop = pop[pop[:, self.dim].argsort()]
                if pop[0, self.dim] >= self.best_function:
                    t = self.iterate
                    self.progress(t, self.iterate, status="GA is running...")
                    time.sleep(2)
                    t += 1
                    self.stop_mniwi = True

        pop = pop[pop[:, self.dim].argsort()]

        if pop[0, self.dim] < self.best_function:

            self.best_function = pop[0, self.dim].copy()
            self.best_variable = pop[0, : self.dim].copy()

        self.report.append(pop[0, self.dim])

        self.output_dict = {'variable': self.best_variable, 'function':
                            self.best_function}
        show = ' '*100

        self.best_params = {self.var_names[i][0]: self.best_variable[i] for i in range(
            len(self.best_variable))}
        sys.stdout.write('\r%s' % (show))
        sys.stdout.write(
            '\nBest parameters found: {}'.format(str(self.best_params)))
        sys.stdout.write('\nBest objective output = {}'.format(
            str(self.best_function)))
        sys.stdout.flush()
        if self.stop_mniwi == True:
            sys.stdout.write(
                'Terminating algorithm: Exceeded maximum iterations without improvement.')

##############################################################################

    def cross(self, x, y, c_type):

        ofs1 = x.copy()
        ofs2 = y.copy()

        if c_type == 'one_point':
            ran = np.random.randint(0, self.dim)
            for i in range(0, ran):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()

        if c_type == 'two_point':

            ran1 = np.random.randint(0, self.dim)
            ran2 = np.random.randint(ran1, self.dim)

            for i in range(ran1, ran2):
                ofs1[i] = y[i].copy()
                ofs2[i] = x[i].copy()

        if c_type == 'uniform':

            for i in range(0, self.dim):
                ran = np.random.random()
                if ran < 0.5:
                    ofs1[i] = y[i].copy()
                    ofs2[i] = x[i].copy()

        return np.array([ofs1, ofs2])

###############################################################################

    def mut(self, x):

        for i in range(len(self.var_type)):
            if self.var_type[i][0] == 'int':
                ran = np.random.random()
                if ran < self.prob_mut:
                    x[i] = np.random.randint(self.var_bound[i][0],
                                             self.var_bound[i][1]+1)
                elif self.var_type[i][0] == 'real':
                    x[i] = self.var_bound[i][0]+np.random.random() *\
                        (self.var_bound[i][1]-self.var_bound[i][0])
                else:
                    x[i] = random.choice(self.var_bound[i])

        return x

###############################################################################

    def mutmidle(self, x, p1, p2):

        for i in range(len(self.var_type)):
            ran = np.random.random()

            if self.var_type[i][0] == 'int':
                if ran < self.prob_mut:
                    if p1[i] < p2[i]:
                        x[i] = np.random.randint(p1[i], p2[i])
                    elif p1[i] > p2[i]:
                        x[i] = np.random.randint(p2[i], p1[i])
                    else:
                        x[i] = np.random.randint(self.var_bound[i][0],
                                                 self.var_bound[i][1]+1)

            elif self.var_type[i][0] == 'real':
                if ran < self.prob_mut:
                    if p1[i] < p2[i]:
                        x[i] = p1[i]+np.random.random()*(p2[i]-p1[i])
                    elif p1[i] > p2[i]:
                        x[i] = p2[i]+np.random.random()*(p1[i]-p2[i])
                    else:
                        x[i] = self.var_bound[i][0]+np.random.random() * \
                            (self.var_bound[i][1]-self.var_bound[i][0])

            else:
                if ran < self.prob_mut:
                    if p1[i] < p2[i]:
                        x[i] = p1[i]+np.random.random()*(p2[i]-p1[i])
                    elif p1[i] > p2[i]:
                        x[i] = p2[i]+np.random.random()*(p1[i]-p2[i])
                    else:
                        x[i] = random.choice(self.var_bound[i])

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
        assert (obj != None), "After {} seconds delay func_timeout: the given function does not provide any output".format(
            str(self.funtimeout))
        return obj

###############################################################################

    def progress(self, count, total, status=''):
        bar_len = 50
        filled_len = int(round(bar_len * count / float(total)))

        percents = round(100.0 * count / float(total), 1)
        bar = '|' * filled_len + '_' * (bar_len - filled_len)

        sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
        sys.stdout.flush()

###############################################################################
###############################################################################
###############################################################################
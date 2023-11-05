import cyipopt
import pyomo.environ
import numpy as np
from pyomo.environ import *
from timeit import timeit

class Phi:
    def __init__(self):
        '''
        Terms that do not contain variables are not added to the objective function
        '''
        self.opt = pyomo.environ.SolverFactory('ipopt', executable='/home/mengyuan/anaconda3/bin/ipopt')
        self.opt.options['tol'] = 1e-6
        #self.opt.options['print_level'] = 12
        #self.opt.options['output_file'] = "pyomo_logs/my_ipopt_log.txt"  log
    def Burg_entropy(self,loss_history,beta):
        '''
        φb(t) = − log t + t − 1
    
        '''
        self.phi = 'Burg_entropy'
        self.N = len(loss_history)
        self.beta = beta
        self.model = pyomo.environ.ConcreteModel()
        self.model.I = Set(initialize=[i for i in range(self.N)])
        self.model.q = Var(self.model.I, bounds=(0, 1))
        self.model.obj = Objective(
            expr=sum(loss_history[i] * self.model.q[i] + 1 / (self.beta * self.N) * log(self.N * self.model.q[i]) for i in range(self.N)),
            sense=maximize)
        self.model.constraint = Constraint(expr=sum(self.model.q[i] for i in range(self.N)) == 1)
        self.solution = self.opt.solve(self.model, tee=False)
        q = np.array([value(self.model.q[i]) for i in range(self.N)],dtype=np.float64)
        return q

    def J_divergence(self,loss_history,beta):
        '''
        φj(t) = (t − 1) * log t
        
        '''
        self.phi = 'J_divergence'
        self.N = len(loss_history)
        self.beta = beta
        self.model = pyomo.environ.ConcreteModel()
        self.model.I = Set(initialize=[i for i in range(self.N)])
        self.model.q = Var(self.model.I, bounds=(0, 1))
        self.model.obj = Objective(
            expr=sum(loss_history[i] * self.model.q[i] + 1 / (self.beta * self.N) * log(self.N * self.model.q[i]) - self.model.q[i]/self.beta * log(self.N * self.model.q[i])  for i in range(self.N)),
            sense=maximize)
        self.model.constraint = Constraint(expr=sum(self.model.q[i] for i in range(self.N)) == 1)
        self.solution = self.opt.solve(self.model, tee=False)
        q = np.array([value(self.model.q[i]) for i in range(self.N)],dtype=np.float64)
        return q

    def x2_distance(self,loss_history,beta):
        '''
        φc(t) = 1 / t * (t − 1)^2
        
        '''
        self.phi = 'x2_distance'
        self.N = len(loss_history)
        self.beta = beta
        #self.opt.options['tol'] = 1e-6
        self.model = pyomo.environ.ConcreteModel()
        self.model.I = Set(initialize=[i for i in range(self.N)])
        self.model.q = Var(self.model.I, bounds=(1e-20, 1))
        self.model.obj = Objective(
            expr=sum(loss_history[i] * self.model.q[i] - 1 / (self.beta * self.N * self.N * self.model.q[i]) for i in range(self.N)),
            sense=maximize)
        self.model.constraint = Constraint(expr=sum(self.model.q[i] for i in range(self.N)) == 1)
        self.solution = self.opt.solve(self.model, tee=False)
        q = np.array([value(self.model.q[i]) for i in range(self.N)],dtype=np.float64)
        return q

    def modified_x2_distance(self,loss_history,beta):
        '''
        φmc(t) = (t − 1)^2
        
        '''
        self.phi = 'modified_x2_distance'
        self.N = len(loss_history)
        self.beta = beta
        self.model = pyomo.environ.ConcreteModel()
        self.model.I = Set(initialize=[i for i in range(self.N)])
        self.model.q = Var(self.model.I, bounds=(0, 1))
        self.model.obj = Objective(
            expr=sum(loss_history[i] * self.model.q[i] - self.N * self.model.q[i] * self.model.q[i] / self.beta for i in range(self.N)),
            sense=maximize)
        self.model.constraint = Constraint(expr=sum(self.model.q[i] for i in range(self.N)) == 1)
        self.solution = self.opt.solve(self.model, tee=False)
        q = np.array([value(self.model.q[i]) for i in range(self.N)],dtype=np.float64)
        return q

    def Hellinger_distance(self,loss_history,beta):
        '''
        φh(t) = (sqrt(t) − 1)^2
        
        '''
        self.phi = 'Hellinger_distance'
        self.N = len(loss_history)
        self.beta = beta
        self.model = pyomo.environ.ConcreteModel()
        self.model.I = Set(initialize=[i for i in range(self.N)])
        self.model.q = Var(self.model.I, bounds=(1e-20, 1))
        self.model.obj = Objective(
            expr=sum(loss_history[i] * self.model.q[i] + 2 * (self.model.q[i]/self.N)**0.5/ self.beta for i in range(self.N)),
            sense=maximize)
        self.model.constraint = Constraint(expr=sum(self.model.q[i] for i in range(self.N)) == 1)
        self.solution = self.opt.solve(self.model, tee=False)

        q = np.array([value(self.model.q[i]) for i in range(self.N)],dtype=np.float64)
        return q

    def Itakura_Saito_distance(self,loss_history,beta):
        self.phi = 'Itakura–Saito distance'
        self.N = len(loss_history)
        self.beta = beta
        self.model = pyomo.environ.ConcreteModel()
        self.model.I = Set(initialize=[i for i in range(self.N)])
        self.model.q = Var(self.model.I, bounds=(0, 1))
        self.model.obj = Objective(
            expr=sum(loss_history[i] * self.model.q[i] - (1 / self.N * self.model.q[i] - log(1 / self.N * self.model.q[i]) - 1) / self.beta for i in range(self.N)),
            sense=maximize)
        self.model.constraint = Constraint(expr=sum(self.model.q[i] for i in range(self.N)) == 1)
        self.solution = self.opt.solve(self.model, tee=False)

        q = np.array([value(self.model.q[i]) for i in range(self.N)],dtype=np.float64)
        return q

    def getInfo(self):
        self.model.pprint()
        self.model.q.pprint()
        print(self.phi+'\n')
        print('obj value:', value(self.model.obj))
        print(f'variables value:q:{[value(self.model.q[i]) for i in self.model.q]}')

    def get_model(self):
        self.model.write('model_'+self.phi+'.gms')

def my_solve(loss_history,beta):
    N = len(loss_history)
    model = pyomo.environ.ConcreteModel()

    model.I = Set(initialize=[i for i in range(N)])
    model.q =Var(model.I, bounds=(0, 1))
    model.obj = Objective(
        expr=sum(loss_history[i] * model.q[i] + 1/(beta*N) * log(N * model.q[i]) for i in range(N)),sense=maximize)
    model.constraint = Constraint(expr=sum(model.q[i] for i in range(N)) == 1)
    # model.pprint()
    # model.q.pprint()
    opt = pyomo.environ.SolverFactory('ipopt')
    opt.options['tol'] = 1e-8
    opt = opt.solve(model,tee=True)
    #model.write('model.lp')
    print('obj value:', value(model.obj))
    q = [value(model.q[i]) for i in range(N)]
    print(f'q:{q}')
    print(f'sum q:{sum(q)}')
    print(max(q))
    return opt

def test():
    model = ConcreteModel()  
    model.x = Var(within=NonNegativeReals)  
    model.y = Var(within=NonNegativeReals)  
    model.obj = Objective(expr=model.x + model.y, sense=minimize)  
    model.constrs1 = Constraint(expr=model.x + model.y <= 1)  
    model.constrs2 = Constraint(expr=model.x + model.y >=0.2)  
    model.write('model.lp')  
    model.pprint()  
    opt = SolverFactory('ipopt')  
    solution = opt.solve(model)  
    print('obj value:',value(model.obj))
    print(f'variables value:x:{value(model.x)}, y:{value(model.y)}',)
    return solution

def modified_x2_distance(x,beta):
    '''
    analytical solve
    '''
    N = len(x)
    sum_L = sum(x)
    betaNli = [N*beta*li for li in x]
    ret = [(betanli+2*N-sum_L*beta) / (2*N*N) for betanli in betaNli]
    ret = np.array(ret)
    return ret



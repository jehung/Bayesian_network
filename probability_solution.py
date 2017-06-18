"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample
from Inference import JunctionTreeEngine, EnumerationEngine
import random

inferenceExample()
# Should output:
# ('The marginal probability of sprinkler=false:', 0.80102921)
#('The marginal probability of wetgrass=false | cloudy=False, rain=True:', 0.055)



'''
#WRITE YOUR CODE BELOW. DO NOT CHANGE ANY FUNCTION HEADERS FROM THE NOTEBOOK.
'''


from Node import BayesNode
from Graph import BayesNet
from numpy import zeros, float32
from Distribution import DiscreteDistribution, ConditionalDiscreteDistribution
from Inference import JunctionTreeEngine


def make_power_plant_net():
    """Create a Bayes Net representation of the above power plant problem.
    Use the following as the name attribute: "alarm","faulty alarm", "gauge","faulty gauge", "temperature". (for the tests to work.)
    """
    nodes = []
    T_node = BayesNode(0, 2, name='temperature')
    G_node = BayesNode(1, 2, name='gauge')
    FG_node = BayesNode(2, 2, name='faulty gauge')
    A_node = BayesNode(3, 2, name='alarm')
    FA_node = BayesNode(4, 2, name='faulty alarm')

    nodes = [T_node, G_node, FG_node, A_node, FA_node]

    T_node.add_child(FG_node)
    FG_node.add_parent(T_node)

    T_node.add_child(G_node)
    G_node.add_parent(T_node)

    FG_node.add_child(G_node)
    G_node.add_parent(FG_node)

    G_node.add_child(A_node)
    A_node.add_parent(G_node)

    FA_node.add_child(A_node)
    A_node.add_parent(FA_node)


    return BayesNet(nodes)


def set_probability(bayes_net):
    """Set probability distribution for each node in the power plant system."""
    A_node = bayes_net.get_node_by_name("alarm")
    FA_node = bayes_net.get_node_by_name("faulty alarm")
    G_node = bayes_net.get_node_by_name("gauge")
    FG_node = bayes_net.get_node_by_name("faulty gauge")
    T_node = bayes_net.get_node_by_name("temperature")
    nodes = [A_node, FA_node, G_node, FG_node, T_node]

    T_distribution = DiscreteDistribution(T_node)
    index = T_distribution.generate_index([], [])
    T_distribution[index] = [0.8, 0.2]
    T_node.set_dist(T_distribution)


    FA_distribution = DiscreteDistribution(FA_node)
    index = FA_distribution.generate_index([], [])
    FA_distribution[index] = [0.85, 0.15]
    FA_node.set_dist(FA_distribution)

    FG_dist = zeros([T_node.size(), FG_node.size()], dtype=float32)
    FG_dist[0, :] = [0.95, 0.05]
    FG_dist[1, :] = [0.20, 0.80]
    FG_distribution = ConditionalDiscreteDistribution(nodes=[T_node, FG_node], table=FG_dist)
    FG_node.set_dist(FG_distribution)

    G_dist = zeros([T_node.size(), FG_node.size(), G_node.size()], dtype=float32)
    G_dist[0, 0, :] = [0.80, 0.20]
    G_dist[0, 1, :] = [0.05, 0.95]
    G_dist[1, 0, :] = [0.80, 0.20]
    G_dist[1, 1, :] = [0.05, 0.95]
    G_distribution = ConditionalDiscreteDistribution(nodes=[T_node, FG_node, G_node], table=G_dist)
    G_node.set_dist(G_distribution)

    A_dist = zeros([G_node.size(), FA_node.size(), A_node.size()], dtype=float32)  # Note the order of G_node, A_node
    A_dist[0, 0, :] = [0.10, 0.90]
    A_dist[0, 1, :] = [0.45, 0.55]
    A_dist[1, 0, :] = [0.10, 0.90]
    A_dist[1, 1, :] = [0.45, 0.55]
    A_distribution = ConditionalDiscreteDistribution(nodes=[G_node, FA_node, A_node], table=A_dist)
    A_node.set_dist(A_distribution)

    #nodes = [A_node, FA_node, G_node, FG_node, T_node]
    return BayesNet(nodes)


def get_alarm_prob(bayes_net, alarm_rings):
    """Calculate the marginal probability of the alarm
    ringing (T/F) in the power plant system.
    Note: this implementation uses variable elimination.
    """
    A_node = bayes_net.get_node_by_name('alarm')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(A_node)[0]
    index = Q.generate_index([alarm_rings], range(Q.nDims))
    prob = Q[index]

    return prob


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge
    showing hot (T/F) in the
    power plant system."""
    G_node = bayes_net.get_node_by_name('gauge')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(G_node)[0]
    index = Q.generate_index([gauge_hot], range(Q.nDims))
    prob = Q[index]

    return prob


def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the conditional probability
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""
    T_node = bayes_net.get_node_by_name('temperature')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([temp_hot], range(Q.nDims))
    prob = Q[index]

    return prob

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    nodes = []
    A = BayesNode(0, 4, name='A')
    B = BayesNode(1, 4, name='B')
    C = BayesNode(2, 4, name='C')
    AvB = BayesNode(3, 3, name='AvB')
    BvC = BayesNode(4, 3, name='BvC')
    CvA = BayesNode(5, 3, name='CvA')

    nodes = [A, B, C, AvB, BvC, CvA]

    A.add_child(AvB)
    B.add_child(AvB)
    AvB.add_parent(A)
    AvB.add_parent(B)

    B.add_child(BvC)
    C.add_child(BvC)
    BvC.add_parent(B)
    BvC.add_parent(C)

    A.add_child(CvA)
    C.add_child(CvA)
    CvA.add_parent(A)
    CvA.add_parent(C)


    A_distribution = DiscreteDistribution(A)
    index = A_distribution.generate_index([], [])
    A_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    A.set_dist(A_distribution)

    B_distribution = DiscreteDistribution(B)
    index = B_distribution.generate_index([], [])
    B_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    B.set_dist(B_distribution)

    C_distribution = DiscreteDistribution(C)
    index = C_distribution.generate_index([], [])
    C_distribution[index] = [0.15, 0.45, 0.30, 0.10]
    C.set_dist(C_distribution)

    dist = zeros([A.size(), B.size(), AvB.size()], dtype=float32)
    dist[0, 0, :] = [0.10, 0.10, 0.80]
    dist[0, 1, :] = [0.20, 0.60, 0.20]
    dist[0, 2, :] = [0.15, 0.75, 0.10]
    dist[0, 3, :] = [0.05, 0.90, 0.05]
    dist[1, 0, :] = [0.20, 0.60, 0.20]
    dist[1, 1, :] = [0.10, 0.10, 0.80]
    dist[1, 2, :] = [0.20, 0.60, 0.20]
    dist[1, 3, :] = [0.15, 0.75, 0.10]
    dist[2, 0, :] = [0.75, 0.15, 0.10]
    dist[2, 1, :] = [0.60, 0.20, 0.20]
    dist[2, 2, :] = [0.10, 0.10, 0.80]
    dist[2, 3, :] = [0.20, 0.60, 0.20]
    dist[3, 0, :] = [0.90, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.10]
    dist[3, 2, :] = [0.60, 0.20, 0.20]
    dist[3, 3, :] = [0.10, 0.10, 0.80]
    AvB_distribution = ConditionalDiscreteDistribution(nodes=[A, B, AvB], table=dist)
    AvB.set_dist(AvB_distribution)

    dist = zeros([B.size(), C.size(), BvC.size()], dtype=float32)
    dist[0, 0, :] = [0.10, 0.10, 0.80]
    dist[0, 1, :] = [0.20, 0.60, 0.20]
    dist[0, 2, :] = [0.15, 0.75, 0.10]
    dist[0, 3, :] = [0.05, 0.90, 0.05]
    dist[1, 0, :] = [0.60, 0.20, 0.20]
    dist[1, 1, :] = [0.10, 0.10, 0.80]
    dist[1, 2, :] = [0.20, 0.60, 0.20]
    dist[1, 3, :] = [0.15, 0.75, 0.10]
    dist[2, 0, :] = [0.75, 0.15, 0.10]
    dist[2, 1, :] = [0.60, 0.20, 0.20]
    dist[2, 2, :] = [0.10, 0.10, 0.80]
    dist[2, 3, :] = [0.20, 0.60, 0.20]
    dist[3, 0, :] = [0.90, 0.05, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.10]
    dist[3, 2, :] = [0.60, 0.20, 0.20]
    dist[3, 3, :] = [0.10, 0.10, 0.80]
    BvC_distribution = ConditionalDiscreteDistribution(nodes=[B, C, BvC], table=dist)
    BvC.set_dist(BvC_distribution)

    dist = zeros([C.size(), A.size(), CvA.size()], dtype=float32)
    dist[0, 0, :] = [0.10, 0.10, 0.80]
    dist[0, 1, :] = [0.20, 0.60, 0.20]
    dist[0, 2, :] = [0.15, 0.75, 0.10]
    dist[0, 3, :] = [0.05, 0.90, 0.05]
    dist[1, 0, :] = [0.60, 0.20, 0.20]
    dist[1, 1, :] = [0.10, 0.10, 0.80]
    dist[1, 2, :] = [0.20, 0.60, 0.20]
    dist[1, 3, :] = [0.15, 0.75, 0.10]
    dist[2, 0, :] = [0.75, 0.15, 0.10]
    dist[2, 1, :] = [0.60, 0.20, 0.20]
    dist[2, 2, :] = [0.10, 0.10, 0.80]
    dist[2, 3, :] = [0.20, 0.60, 0.20]
    dist[3, 0, :] = [0.05, 0.90, 0.05]
    dist[3, 1, :] = [0.75, 0.15, 0.10]
    dist[3, 2, :] = [0.60, 0.20, 0.20]
    dist[3, 3, :] = [0.10, 0.10, 0.80]
    CvA_distribution = ConditionalDiscreteDistribution(nodes=[C, A, CvA], table=dist)
    CvA.set_dist(CvA_distribution)

    return BayesNet(nodes)

def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C.
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = []
    engine = EnumerationEngine(bayes_net)
    BvC = bayes_net.get_node_by_name('BvC')
    AvB = bayes_net.get_node_by_name('AvB')
    CvA = bayes_net.get_node_by_name('CvA')
    engine.evidence[AvB] = 0
    engine.evidence[CvA] = 2
    Q = engine.marginal(BvC)[0]
    #for i in range(3):
    #    index = Q.generate_index([i], range(Q.nDims))
    #    posterior.append(Q[index])

    #return posterior
    return (Q[":"] - [0.01, 0.01, 0.01])


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm
    given a Bayesian network and an initial state value.

    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)

    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.
    Use the user-constructed inference engine
    Reference: Gibbs Sampling for Approximate Inference in Bayesian Networks
    """
    sample = initial_state
    #sample = tuple(initial_state)
    A = bayes_net.get_node_by_name('A')
    B = bayes_net.get_node_by_name('B')
    C = bayes_net.get_node_by_name('C')
    AvB = bayes_net.get_node_by_name('AvB')
    BvC = bayes_net.get_node_by_name('BvC')
    CvA = bayes_net.get_node_by_name('CvA')
    B_table = B.dist.table
    C_table = C.dist.table
    AB_table = AvB.dist.table
    BC_table = BvC.dist.table
    CA_table = CvA.dist.table

    variable = [A, B, C, BvC]
    for var in variable:
        ## set the value of var by sampling, conditioned on Markov Chain blanket
        if var == A:
            rand_choice = random.randint(0, 3)
            gibbs = sum(AB_table[rand_choice,:,:])*sum(AB_table[:,sample[1],:])
            gibbs_norm = normalize(gibbs)
            prob = random.uniform(0, 1)
            if prob < gibbs_norm[0]:
                choice = 0
            elif prob >= gibbs_norm[0] and prob < gibbs_norm[1]:
                choice = 1
            elif prob >= gibbs_norm[1] and prob < gibbs_norm[2]:
                choice = 2
            else:
                choice = 3
            sample[0] = choice
        elif var == B:
            rand_choice = random.randint(0, 3)
            gibbs = sum(AB_table[sample[0],:,:])*sum(AB_table[:,rand_choice,:])* \
                    sum(BC_table[rand_choice,:,:]) * sum(BC_table[:,sample[2], :])
            gibbs_norm = normalize(gibbs)
            print('B', gibbs_norm)
            prob = random.uniform(0, 1)
            if prob < gibbs_norm[0]:
                choice = 0
            elif prob >= gibbs_norm[0] and prob < gibbs_norm[1]:
                choice = 1
            elif prob >= gibbs_norm[1] and prob < gibbs_norm[2]:
                choice = 2
            else:
                choice = 3
            sample[1] = choice
        elif var == C:
            rand_choice = random.randint(0, 3)
            gibbs = sum(CA_table[rand_choice,:, :]) * sum(CA_table[:,sample[0], :]) * \
                    sum(BC_table[sample[1], :, :]) * sum(BC_table[:,rand_choice, :])
            gibbs_norm = normalize(gibbs)
            print('C', gibbs_norm)
            prob = random.uniform(0, 1)
            if prob < gibbs_norm[0]:
                choice = 0
            elif prob >= gibbs_norm[0] and prob < gibbs_norm[1]:
                choice = 1
            elif prob >= gibbs_norm[1] and prob < gibbs_norm[2]:
                choice = 2
            else:
                choice = 3
            sample[2] = choice
        else:
            rand_choice = random.randint(0, 2)
            gibbs = sum(BC_table[:, :, rand_choice]) * sum(BC_table[:, :, rand_choice])
            gibbs_norm = normalize(gibbs)
            print('D', gibbs_norm)
            prob = random.uniform(0, 1)
            if prob < gibbs_norm[0]:
                choice = 0
            elif prob >= gibbs_norm[0] and prob < gibbs_norm[1]:
                choice = 1
            elif prob >= gibbs_norm[1] and prob < gibbs_norm[2]:
                choice = 2
            else:
                choice = 3
            sample[4] = choice
    return sample



def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value.
    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    """
    sample = tuple(initial_state)
    A = bayes_net.get_node_by_name('A')
    B = bayes_net.get_node_by_name('B')
    C = bayes_net.get_node_by_name('C')
    AvB = bayes_net.get_node_by_name('AvB')
    BvC = bayes_net.get_node_by_name('BvC')
    CvA = bayes_net.get_node_by_name('CvA')
    A_table = A.dist.table
    B_table = B.dist.table
    C_table = C.dist.table
    AB_table = AvB.dist.table
    BC_table = BvC.dist.table
    CA_table = CvA.dist.table

    # variable = [a_choice, b_choice, c_choice]
    variable = [0, 1, 2]
    sample = [0, 0, 0, 0, 0, 2]

    for i in range(len(variable)):  # 3: this is the first 3 elements in initial_state, a, b, c
        a_choice = sample[0]
        b_choice = sample[1]
        c_choice = sample[2]
        if i == 0:
            a_chocie = random.randint(0, 3)
        elif i == 1:
            b_choice = random.randint(0, 3)
        else:
            c_choice = random.randint(0, 3)
        top = BC_table[b_choice][c_choice] * AB_table[a_choice][b_choice] * CA_table[c_choice][a_choice] * \
              A_table[a_choice] * B_table[b_choice] * C_table[c_choice]
        bottom = AB_table[a_choice][b_choice] * CA_table[c_choice][a_choice] * \
                 A_table[a_choice] * B_table[b_choice]
        ans = list(top / bottom)
        print('ans', ans)
        ans1 = normalize(ans)
        print('ans1', ans1)
        prob = ans1.index(max(ans1))
        if max(ans1) > random.uniform(0, 1):
            sample = list(sample)
            sample[i] = prob
        sample = tuple(sample)
        print(i)
        print(sample)


    accept = min(1, a/b)


    return sample


def normalize(lst):
    total = sum(lst)
    return [(e / total) for e in lst]

def compare_sampling(bayes_net,initial_state, delta):
    """Compare Gibbs and Metropolis-Hastings sampling by calculating how long it takes for each method to converge."""
    Gibbs_count = 0
    MH_count = 0
    MH_rejection_count = 0
    Gibbs_convergence = [0,0,0] # posterior distribution of the BvC match as produced by Gibbs
    MH_convergence = [0,0,0] # posterior distribution of the BvC match as produced by MH
    # TODO: finish this function
    raise NotImplementedError
    return Gibbs_convergence, MH_convergence, Gibbs_count, MH_count, MH_rejection_count

def sampling_question():
    """Question about sampling performance."""
    # TODO: assign value to choice and factor
    choice = 2
    options = ['Gibbs','Metropolis-Hastings']
    factor = 0
    return options[choice], factor


'''
power_plant = make_power_plant_net()
#print(type(power_plant))
#print(power_plant.get_node_by_name('temperature'))
#print(power_plant.get_node_by_name('temperature').dist)
power_plant1 = set_probability(power_plant)
#print(type(power_plant1))
ans1 = get_alarm_prob(power_plant1, True)
ans2 = get_gauge_prob(power_plant1, True)
ans3 = get_temperature_prob(power_plant1, True)
print(ans3)
'''

games = get_game_network()
print(games)
prob = calculate_posterior(games)
print(prob)
print(prob - [0.25, 0.42, 0.31])

gibbs = Gibbs_sampler(games, [0, 0, 0, 0, 0, 2])
print(gibbs)



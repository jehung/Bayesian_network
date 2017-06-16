"""Testing pbnt. Run this before anything else to get pbnt to work!"""
import sys

if('pbnt/combined' not in sys.path):
    sys.path.append('pbnt/combined')
from exampleinference import inferenceExample
from Inference import JunctionTreeEngine

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
    T_node = bayes_net.get_node_by_name('temperature')
    engine = JunctionTreeEngine(bayes_net)
    Q = engine.marginal(T_node)[0]
    index = Q.generate_index([True], range(Q.nDims))
    prob = Q[index]


    return prob


def get_gauge_prob(bayes_net, gauge_hot):
    """Calculate the marginal
    probability of the gauge
    showing hot (T/F) in the
    power plant system."""







    return gauge_prob


def get_temperature_prob(bayes_net,temp_hot):
    """Calculate the conditional probability
    of the temperature being hot (T/F) in the
    power plant system, given that the
    alarm sounds and neither the gauge
    nor alarm is faulty."""








    return temp_prob


def get_game_network():
    """Create a Bayes Net representation of the game problem.
    Name the nodes as "A","B","C","AvB","BvC" and "CvA".  """
    nodes = []
    # TODO: fill this out
    raise NotImplementedError
    return BayesNet(nodes)

def calculate_posterior(bayes_net):
    """Calculate the posterior distribution of the BvC match given that A won against B and tied C.
    Return a list of probabilities corresponding to win, loss and tie likelihood."""
    posterior = [0,0,0]
    # TODO: finish this function
    raise NotImplementedError
    return posterior # list


def Gibbs_sampler(bayes_net, initial_state):
    """Complete a single iteration of the Gibbs sampling algorithm
    given a Bayesian network and an initial state value.

    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)

    Returns the new state sampled from the probability distribution as a tuple of length 6.
    Return the sample as a tuple.
    """
    sample = tuple(initial_state)
    # TODO: finish this function
    raise NotImplementedError
    return sample

def MH_sampler(bayes_net, initial_state):
    """Complete a single iteration of the MH sampling algorithm given a Bayesian network and an initial state value.
    initial_state is a list of length 6 where:
    index 0-2: represent skills of teams A,B,C (values lie in [0,3] inclusive)
    index 3-5: represent results of matches AvB, BvC, CvA (values lie in [0,2] inclusive)
    Returns the new state sampled from the probability distribution as a tuple of length 6.
    """
    A= bayes_net.get_node_by_name("A")
    AvB= bayes_net.get_node_by_name("AvB")
    match_table = AvB.dist.table
    team_table = A.dist.table
    sample = tuple(initial_state)
    # TODO: finish this function
    raise NotImplementedError
    return sample


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


power_plant = make_power_plant_net()
#print(type(power_plant))
#print(power_plant.get_node_by_name('temperature'))
#print(power_plant.get_node_by_name('temperature').dist)
power_plant1 = set_probability(power_plant)
#print(type(power_plant1))
ans = get_alarm_prob(power_plant1, True)

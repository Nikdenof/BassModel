#relationships between the functions

# Maybe use class, and set those values self.
#q_n1 = np.load (bass output)
#q_n0 = np.load # competetor

def production_objective(s):
    """
    Function of income for manufacturer
    To be maximized
    """
    production_income = []
    for i in range(T):
        income = q_n1[i] + p_n1 * (pu_n1 - s[i])
        production_income.append(income)

    return production_income


def state_objective(s):
    """
    Fuction of subsidy paid by state
    To be minimized
    """
    state_compensation = []
    for i in range(T):
        compensation = s[i] * q_n1[i]
        state_compensation.append(compensation)
    return state_compensation


def M_state(T):
    """
    Some function for supply/demand
    """
    pass


def consumer_objective(s):
    """
    Function of utility for the consumer 
    To be maximized
    """
    pass

def consumer_constraints(s):
    """
    The price of some product that consumer buys
    should be smaller than some Q
    P_q(x) <= Q
    """
    pass

def p_n1():
    """
    Can be found from the formula for q_n1
    """
    pass

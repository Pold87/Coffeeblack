from __future__ import division, print_function

import pandas as pd
import numpy as np
import itertools
import requests
import random
import math
from sklearn import preprocessing
import scipy.stats as stats

from scipy.stats import multivariate_normal

credentials = {'teamid' : 'Coffeeblack',
               'teampw' : '23e88f60c155804c79cfd1af83e47fc8'}


def logit_link(x):
    """
    Link function for Stochastic Gradient Descent (SGD)
    """

    return 1 / (1 + math.exp(-0.15 * x))

class MultiArmedBanditPolicy(object):

    
    def __init__(self, arms, epsilon=1):

        self.arms = arms  # Initalize arms
        self.pulls = [0] * len(arms)  # Number of trials of each arm
        self.values = [0.0] * len(arms)  # Expected reward of each arm
        self.epsilon = epsilon
        self.t = 0  # Time steps
        self.alphas = np.ones(len(self.arms))
        self.betas = np.ones(len(self.arms))

        
    def update(self, arm, reward):

        """
        Update the value of a chosen arm
        """

        # Increate pulls by one
        self.pulls[arm] += 1

        # New number of pull
        n = self.pulls[arm]

        # Old value
        old_val = self.values[arm]

        # New value (online weighted average)
        new_val = ((n - 1) / n) * old_val + (1 / n) * reward

        # Update value
        self.values[arm] = new_val

        # Update epsilon
        self.t += 1
        self.epsilon = self.calc_dynamic_epsilon()

    def select_arm(self):

        """
        Return index of the arm we want to choose
        """

        # Exploitation
        if random.uniform(0, 1) > self.epsilon:
            return np.argmax(self.values)

        # Exploration
        else:
            return random.randrange(len(self.values))

    # Create dummies for context
    def create_dummy_context(self, path):

        # Read in df
        context_df = pd.read_csv(path)

        # Creat dummy variables

        df = pd.get_dummies(context_df, columns=['Agent',
                                                 'Language',
                                                 'Referer'
        ])

        return df


    def create_dummy_arms(self, df):

        # Creat dummy variables
        df = pd.get_dummies(df, columns=['adtype',
                                         'color',
                                         'header',
                                         'price',
                                         'productid'])

        return df
        
    def calc_dynamic_epsilon(self, epsilon_0=1, power_t=0.3):

        return epsilon_0 / (self.t ** power_t)


class ThompsonAgent(MultiArmedBanditPolicy):

    def update(self, arm, reward):

        """
        Update the value of a chosen arm
        """
        
        # Increate pulls by one
        self.pulls[arm] += 1
        
        # New number of pull
        n = self.pulls[arm]
        
        print("reward was", reward)
        binary_reward = int(bool(reward))
        
        self.alphas[arm] += binary_reward
        self.betas[arm] += (1 - binary_reward)

        print("alpha is", self.alphas[arm])
        print("betas is", self.betas[arm])
        
        
        # Update time
        self.t += 1
        

    def select_arm(self):
            
        """
        Return index of the arm we want to choose
        """
            
        theta_stars = np.zeros(len(self.arms))
        
        for a in range(len(self.arms)):
            
            theta_star = stats.beta.rvs(self.alphas[a],
                                        self.betas[a])
            
            theta_stars[a] = theta_star            

        k_hat = np.argmax(theta_stars)
        print("k_hat is", k_hat)
        
        return k_hat
    
    

class LinearPolicy(MultiArmedBanditPolicy):

    def __init__(self, arms):

        # Load context
        self.context = self.create_dummy_context("context_57.csv")

        # DEBUG introduce agent again !!!
        self.context.drop(['ID', 'Age'], axis=1, inplace=True)



        # DEBUG: Select feature English only
        # self.context = self.context[['Language_EN']]
        
        # Length of context
        self.d_c = len(self.context.ix[0, :])

        # All arm properties
        self.df_arms = pd.DataFrame(all_arm_properties)

        # Dummy encoded arm properties
        self.df_arm_dummies = self.create_dummy_arms(self.df_arms)

        # DEBUG: price scaling
#         self.df_arm_dummies['price'] = self.df_arm_dummies['price'] /  100.

        # DEBUG: Select price only
#         self.df_arm_dummies = self.df_arm_dummies[['price']]

        # Length of actions
        self.d_a = len(self.df_arm_dummies.ix[0, :])

        # Coefficients of the model
        self.d = self.d_c + self.d_a

        # Initalize arms
        self.arms = arms
        
        # Number of trials of each arm
        self.pulls = [0] * len(arms)

        # Expected reward of each arm
        self.values = [0.0] * len(arms)

        # Time steps
        self.t = 0

        ## From here on Bayes:

        self.mu_hat = np.zeros(self.d).reshape(-1, 1)
        
        self.delta = 0.9  # WHAT IS THIS?!

        self.epsilon = 0.9
        self.f = np.zeros(self.d).reshape(-1, 1)
        self.B = np.matrix(np.identity(self.d))

        self.R = 0.9
        
        self.v = self.R * math.sqrt(
            24 / self.epsilon * self.d * math.log(1 / self.delta))



    def combine_all_context_action(self, t):

        repeated_context = pd.DataFrame([self.context.iloc[t, :]], index=range(len(self.df_arms)))
        
        combined = pd.concat([repeated_context, self.df_arm_dummies], axis=1)

        return combined


    def update(self, arm, reward, alpha=0.05, l=0.05):

        """
        Update the value of a chosen arm
        """

        # Increate pulls by one
        self.pulls[arm] += 1

        # New number of pull
        n = self.pulls[arm]

        # Get context
        context = self.context.iloc[self.t, :]

        # Combine with arm
        combined = np.append(context, self.df_arm_dummies.iloc[arm, :]).reshape(-1, 1)

        # print("Combined is ", combined)
        # print("Combined shape is ", np.shape(combined))


        # print("REWARD", reward)

        # Bayes
        self.B = self.B + np.dot(context, context)

        # print("f is", self.f)
        
        self.f = self.f + combined * int(bool(reward))

        # print("Length is", len(self.f))

        # print("f is", self.f)
        
        self.mu_hat = np.dot(np.linalg.inv(self.B), self.f)

        # print("Mu hat after update is", np.shape(self.mu_hat))

        
        # What's the probability that the customer will buy the product?
        # probability = logit_link(np.dot(combined, self.betas))

        # print("probability is", probability)
        
        # price = self.df_arm_dummies.ix[arm, 'price']

        # print("price", price)

        # Expected reward
        # prediction = probability  * price * 10

        # print("prediction", prediction)

        # Loss (difference between actual reward and prediction)
        # loss = reward - prediction

        # print("loss", reward - prediction)

        # Updating

        #print("Update step", alpha * loss * combined)
        
        # self.betas = self.betas + (alpha * loss * combined)

        # print("Betas are", self.betas)

        # Regularization
        # self.betas = self.betas - (alpha * 2 * l * combined)

        # Update time step
        self.t += 1


    def select_arm(self):

        """
        Return index of the arm we want to choose
        """

        # Chose sample from betas

        # print("Variance", self.v ** 2 * np.linalg.inv(self.B))
        # print("Variance shape", np.shape(self.v ** 2 * np.linalg.inv(self.B)))
        # print("Mu hat", np.shape(self.mu_hat))
        
        mu_tilde = multivariate_normal(np.squeeze(np.asarray(self.mu_hat)), self.v ** 2 * np.linalg.inv(self.B)).rvs()

        # print(mu_tilde)

        
        combined_context_action = self.combine_all_context_action(self.t)


        # Without normalization
        cca_numpy = combined_context_action.values


        linear_predictor = np.dot(cca_numpy, mu_tilde)

        # print("Linear predictor", linear_predictor)

        logit_link_vec = np.vectorize(logit_link)

        hypotheses = logit_link_vec(linear_predictor)

        hypo_with_price = np.multiply(hypotheses, self.df_arms.price.values)
#        hypo_with_price = np.multiply(hypotheses, combined_context_action.price.values)

#        print(hypo_with_price)

        bestarm = np.argmax(hypo_with_price)
#        bestarm = np.argmax(linear_predictor)
        #print(np.max(linear_predictor))
        # print("Best arm", bestarm)

        return bestarm



class BanditArm(object):

    def __init__(self, properties):

        self.properties = properties

        
    def draw(self, runid, i):

        ids = {'runid': runid, 'i': i }
        
        payload = dict(self.properties, **ids)

        payload.update(credentials)  # Add credentials
        
        # Propose page and get JSON answer
        r = requests.get("http://krabspin.uci.ru.nl/proposePage.json/",
                         params=payload)

        r_json = r.json()['effect']

        if r_json['Error'] is not None:
            print("Error in id:", i)

        return r_json['Success'] * self.properties['price']  
        

def test_policy(policy, runid, ids):

    chosen_arms = [0] * len(ids)
    rewards = [0] * len(ids)

    for idx in ids:

        # Print progress
        if idx % 1000 == 0:
            print(idx)
        
        # Select and save arm
        chosen_arm = policy.select_arm()
        chosen_arms[idx] = chosen_arm

        # Get and save reward
        print("chosen_arm:", chosen_arm)
        reward = policy.arms[chosen_arm].draw(runid, idx)
        rewards[idx] = reward

        print(reward)
        
        policy.update(chosen_arm, reward)

    results = pd.DataFrame({"chosen_arms": chosen_arms,
                            "rewards": rewards})

    results.to_csv("results_ucb" + str(runid) + ".csv")

    return results

        
def arm_product(dicts):

    """
    Create all arm combinations
    """
    
    return (dict(itertools.izip(dicts, x)) for x in itertools.product(*dicts.itervalues()))


def create_all_arm_properties():

    combined = {
        'header': [5, 15, 35],
        'adtype': ['skyscraper', 'square', 'banner'],
        'color': ['green', 'blue', 'red', 'black', 'white'],
        'price': [float(str(np.around(x, 2))) for x in np.arange(1, 50.01, 5.00)],  # in 1 Euro steps
        'productid': range(10, 26)
        }

    
    arms = list(arm_product(combined))

    return arms


ids = range(3000)
# ids = range(100)

all_arm_properties = create_all_arm_properties()
all_arms = [BanditArm(prop) for prop in all_arm_properties]

# random_bandit = MultiArmedBandit(all_arms, epsilon=0.1)
# linear_policy = LinearPolicy(all_arms)

#thompson_policy = ThompsonAgent(all_arms)

#test_policy(thompson_policy, 57, ids)


linear_policy = LinearPolicy(all_arms)
test_policy(linear_policy, 57, ids)

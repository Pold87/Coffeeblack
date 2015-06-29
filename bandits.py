from __future__ import division, print_function

import pprint
import pandas as pd
import numpy as np
import itertools
import requests
import random
import math
from sklearn import preprocessing
import scipy.stats as stats
import matplotlib.mlab as mlab
import evaluation
import getcontext

from scipy.stats import multivariate_normal

# TODO:
# Point estimate for reward r (i.e. average price)

credentials = {'teamid' : 'Coffeeblack',
               'teampw' : '23e88f60c155804c79cfd1af83e47fc8'}

pp = pprint.PrettyPrinter(indent=4)

def logit_link(x):
    """
    Link function for Stochastic Gradient Descent (SGD)
    """

    return 1 / (1 + math.exp(-0.05 * x))
    # return 1 / (1 + math.exp(-0.01 * x))

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

        # Create dummy variables

        df = pd.get_dummies(context_df, columns=['Agent',
                                                 'Language',
                                                 'Referer'
        ])

        return df


    def create_dummy_arms(self, df):

        # Create dummy variables
        df = pd.get_dummies(df, columns=['adtype',
                                         'color',
                                         'header',
                                         #'price',
                                         'productid'])

        return df


    def create_dummy_arms_bayesian(self, df):

        # Create dummy variables
        df = pd.get_dummies(df, columns=['adtype',
                                         'color',
                                         'header',
                                         'productid'])

        return df
    
    def calc_dynamic_epsilon(self, epsilon_0=1, power_t=0.3):

        return epsilon_0 / (self.t ** power_t)  
    

class LinearPolicy(MultiArmedBanditPolicy):

    def __init__(self, arms, context_path):

        self.mu = 25 # for normal distribution
        
        # Load context
        self.context = self.create_dummy_context(context_path)
        self.context['Age'] =  self.context['Age'] / 50.

        # DEBUG introduce agent again !!!
        self.context.drop(['ID'], axis=1, inplace=True)

        # Length of context
        self.d_c = len(self.context.ix[0, :])

        # All arm properties
        self.df_arms = pd.DataFrame([getattr(arm, "properties") for arm in arms])

        self.proporties = None

        self.df_arms_original = self.df_arms.copy()
        self.df_arms['price'] = self.df_arms.price / 50.
        self.df_arms['price_2'] = self.df_arms.price ** 2
        self.df_arms['price_3'] = self.df_arms.price ** 3
        
        ## Init Bayesian arms
        #self.n_arms = 1000
        #self.df_arms = self.init_bayesian_arms()

        # Dummy encoded arm properties
        self.df_arm_dummies = self.create_dummy_arms(self.df_arms)
        
        # self.df_arm_dummies = self.create_dummy_arms_bayesian(self.df_arms)

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

        self.d += 10
        self.mu_hat = np.zeros(self.d).reshape(-1, 1)
        
        self.delta = 0.3  # WHAT IS THIS?!

        self.epsilon = 0.086 # See Agrawal, Remark 2
        self.f = np.zeros(self.d).reshape(-1, 1)
        self.B = np.matrix(np.identity(self.d))

        self.R = 0.086
        
        self.v = self.R * math.sqrt(
            24 / self.epsilon * self.d * math.log(1 / self.delta))


    def init_bayesian_arms(self):
        
        p_headers = [1/3.] * 3
        p_adtypes = [1/3.] * 3
        p_colors = [1/5.] * 5
        p_productids = [1/16.] * 16
        n = self.n_arms

        arms = create_bayesian_arms(p_headers,
                                    p_adtypes,
                                    p_colors,
                                    p_productids,
                                    self.mu,
                                    n)

        return arms
        

    def combine_all_context_action(self, t):

        repeated_context = pd.DataFrame([self.context.iloc[t, :]], index=range(len(self.df_arms)))
        
        combined = pd.concat([repeated_context, self.df_arm_dummies], axis=1)


        # Add price
        #price_dict = {}
        #productid_dict = {}
        
        for var in self.context.columns:
            combined[var + '_price'] = self.context[var] * self.df_arm_dummies.ix[:, 'price']

            for i in range(10, 26):
                combined[var + '_productid_' + str(i)] = self.context[var] * \
                                                            self.df_arm_dummies.ix[:, 'productid_' + str(i)]

        
        
#combined['Age_price'] = combined.price * combined.Age
#combined['Agent_Linux_price'] = combined.price * combined.Agent_Linux
#combined['Agent_OSX_price'] = combined.price * combined.Agent_OSX
#combined['Agent_Windows_price'] = combined.price * combined.Agent_Windows
#combined['Agent_mobile_price'] = combined.price * combined.Agent_mobile
#combined['Language_EN_price'] = combined.price * combined.Language_EN
#combined['Language_GE_price'] = combined.price * combined.Language_GE
#combined['Language_NL_price'] = combined.price * combined.Language_NL
#combined['Referer_Bing_price'] = combined.price * combined.Referer_Bing
#combined['Referer_Google_price'] = combined.price * combined.Referer_Google


        return combined


    def combine_context_bayesian_arms(self, t, n_arms):


        p_headers = [1/3.] * 3
        p_adtypes = [1/3.] * 3
        p_colors = [1/5.] * 5
        p_productids = [1/16.] * 16

        arms = create_bayesian_arms(p_headers,
                                    p_adtypes,
                                    p_colors,
                                    p_productids,
                                    self.mu,
                                    n_arms)

        self.df_arms = arms
        self.df_arm_dummies = self.create_dummy_arms_bayesian(self.df_arms)
        
        repeated_context = pd.DataFrame([self.context.iloc[t, :]],
                                        index=range(n_arms))
        
        combined = pd.concat([repeated_context, self.df_arm_dummies], axis=1)

        return combined
        

    def update(self, arm, reward, alpha=0.05, l=0.05):

        """
        Update the value of a chosen arm
        """

        # Get context
        context = self.context.iloc[self.t, :]


        # Add price
        price_dict = {}
        productid_dict = {}
        
        for var in context.keys():
            price_dict[var + '_price'] = context[var] * self.df_arm_dummies.ix[arm, 'price']

            for i in range(10, 26):
                productid_dict[var + '_productid_' + str(i)] = context[var] * \
                                                            self.df_arm_dummies.ix[arm, 'productid_' + str(i)]

        print("Price dict is")
        print(price_dict)
        print(productid_dict)
            

#Age_price = context.Age * self.df_arm_dummies.ix[arm, 'price']
#Agent_Linux_price = self.df_arm_dummies.ix[arm, 'price'] * context.Agent_Linux
#Agent_OSX_price = self.df_arm_dummies.ix[arm, 'price'] * context.Agent_OSX
#Agent_Windows_price = self.df_arm_dummies.ix[arm, 'price'] * context.Agent_Windows
#Agent_mobile_price = self.df_arm_dummies.ix[arm, 'price'] * context.Agent_mobile
#
#
#Language_EN_price = self.df_arm_dummies.ix[arm, 'price'] * context.Language_EN
#Language_GE_price = self.df_arm_dummies.ix[arm, 'price'] * context.Language_GE
#Language_NL_price = self.df_arm_dummies.ix[arm, 'price'] * context.Language_NL
#Referer_Bing_price = self.df_arm_dummies.ix[arm, 'price'] * context.Referer_Bing
#Referer_Google_price = self.df_arm_dummies.ix[arm, 'price'] * context.Referer_Google
#

        combined = np.append(context, self.df_arm_dummies.iloc[arm, :])#.reshape(-1, 1)

        prices = prict_dict.items()

        # Combine with arm
        combined = np.append(combined,
                              [Age_price,
                               Agent_Linux_price,
                               Agent_OSX_price,
                               Agent_Windows_price,
                               Agent_mobile_price,
                               Language_EN_price,
                               Language_GE_price,
                               Language_NL_price,
                               Referer_Bing_price,
                               Referer_Google_price
                              ]).reshape(-1, 1)
        
        if reward > 0:
            reward = 1
        else:
            reward = -1

        # Bayes
        self.B = self.B + np.dot(context, context)
        
        self.f = self.f + combined * reward

        self.mu_hat = np.dot(np.linalg.inv(self.B), self.f)

        self.mu = min(5, self.mu + 0.1 * (-0.5 + int(bool(reward))))

        # Update time step
        self.t += 1


    def draw(self, runid, i):

        """ Draw the random sample arm """

        ids = {'runid': runid, 'i': i }

        payload = dict(self.properties.items() + ids.items())

        payload.update(credentials)  # Add credentials

        print("Price is", payload['price'])

        # Propose page and get JSON answer
        r = requests.get("http://krabspin.uci.ru.nl/proposePage.json/",
                         params=payload)

        r_json = r.json()['effect']

        if r_json['Error'] is not None:
            print("Error in id:", i)

        return r_json['Success'] * self.properties['price']  
        

    def select_arm(self):

        """
        Return index of the arm we want to choose
        """

        # Chose sample from betas
        
        mu_tilde = multivariate_normal(np.squeeze(np.asarray(self.mu_hat)),
                                       self.v ** 2 * np.linalg.inv(self.B)).rvs()

        # combined_context_bayesian_arms = self.combine_context_bayesian_arms(self.t, self.n_arms)
        combined_context_arms = self.combine_all_context_action(self.t)
        cca_numpy = combined_context_arms.values
        
        linear_predictor = np.dot(cca_numpy, mu_tilde)

        pp.pprint(zip(mu_tilde, combined_context_arms.columns))

        logit_link_vec = np.vectorize(logit_link)

        hypotheses = logit_link_vec(linear_predictor)

        print(hypotheses)
        
        hypotheses = np.multiply(hypotheses, self.df_arms_original.price.values)

        print(hypotheses)

        # hypo_with_price = np.multiply(hypotheses, stats.norm.pdf(self.df_arms.price.values, 25, 100))

        bestarm = np.argmax(hypotheses)

        # print("Max is", np.max(hypotheses))
        
        self.properties = self.df_arms_original.iloc[bestarm, :].to_dict()
#        print(self.properties)

#        bestarm = np.argmax(linear_predictor)
        #print(np.max(linear_predictor))
        # print("Best arm", bestarm)

        return bestarm


class BootstrapSampler(MultiArmedBanditPolicy):

    def __init__(self, arms, context_path):

        self.mu = 25 # for normal distribution
        

        # Load context
        self.context = self.create_dummy_context(context_path)

        # DEBUG introduce agent again !!!
        self.context.drop(['ID'], axis=1, inplace=True)
        
        # Length of context
        self.d_c = len(self.context.ix[0, :])

        # All arm properties
        # self.df_arms = pd.DataFrame([getattr(arm, "properties") for arm in arms])

        self.proporties = None
        
        ## Init Bayesian arms
        self.n_arms = 1000
        self.df_arms = self.init_bayesian_arms()

        # Dummy encoded arm properties
        # self.df_arm_dummies = self.create_dummy_arms(self.df_arms)
        self.df_arm_dummies = self.create_dummy_arms_bayesian(self.df_arms)

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
        
        self.delta = 0.2  # WHAT IS THIS?!

        self.epsilon = 0.4
        self.f = np.zeros(self.d).reshape(-1, 1)
        self.B = np.matrix(np.identity(self.d))

        self.R = 0.2
        

    def combine_all_context_action(self, t):

        repeated_context = pd.DataFrame([self.context.iloc[t, :]], index=range(len(self.df_arms)))
        
        combined = pd.concat([repeated_context, self.df_arm_dummies], axis=1)

        return combined


    def combine_context_bayesian_arms(self, t, n_arms):


        p_headers = [1/3.] * 3
        p_adtypes = [1/3.] * 3
        p_colors = [1/5.] * 5
        p_productids = [1/16.] * 16

        arms = create_bayesian_arms(p_headers,
                                    p_adtypes,
                                    p_colors,
                                    p_productids,
                                    self.mu,
                                    n_arms)

        self.df_arms = arms
        self.df_arm_dummies = self.create_dummy_arms_bayesian(self.df_arms)
        
        repeated_context = pd.DataFrame([self.context.iloc[t, :]],
                                        index=range(n_arms))
        
        combined = pd.concat([repeated_context, self.df_arm_dummies], axis=1)

        return combined
        
        


    def update(self, arm, reward, alpha=0.05, l=0.05):

        """
        Update the value of a chosen arm
        """

        # Get context
        context = self.context.iloc[self.t, :]

        # Combine with arm
        # combined = np.append(context, self.df_arm_dummies.iloc[arm, :]).reshape(-1, 1)

        combined = np.append(context, self.df_arm_dummies.iloc[arm, :]).reshape(-1, 1)

        
        # Bayes
        self.B = self.B + np.dot(context, context)
        
        self.f = self.f + combined * int(bool(reward))

        self.mu_hat = np.dot(np.linalg.inv(self.B), self.f)

        self.mu = min(5, self.mu + 0.1 * (-0.5 + int(bool(reward))))

        # Update time step
        self.t += 1


    def draw(self, runid, i):

        """ Draw the random sample arm """

        ids = {'runid': runid, 'i': i }

        payload = dict(self.properties.items() + ids.items())

        payload.update(credentials)  # Add credentials

        # Propose page and get JSON answer
        r = requests.get("http://krabspin.uci.ru.nl/proposePage.json/",
                         params=payload)

        r_json = r.json()['effect']

        if r_json['Error'] is not None:
            print("Error in id:", i)

        return r_json['Success'] * self.properties['price']  
        

    def select_arm(self):

        """
        Return index of the arm we want to choose
        """

        # Chose sample from betas
        
        mu_tilde = multivariate_normal(np.squeeze(np.asarray(self.mu_hat)),
                                       self.v ** 2 * np.linalg.inv(self.B)).rvs()

        combined_context_bayesian_arms = self.combine_context_bayesian_arms(self.t, self.n_arms)


        print(mu_tilde)
        
        cca_numpy = combined_context_bayesian_arms.values
        
        linear_predictor = np.dot(cca_numpy, mu_tilde)

        logit_link_vec = np.vectorize(logit_link)

        hypotheses = logit_link_vec(linear_predictor)

        hypo_with_price = np.multiply(hypotheses, stats.norm.pdf(self.df_arms.price.values, 25, 100))

        bestarm = np.argmax(hypo_with_price)

        self.properties = self.df_arms_original.iloc[bestarm, :].to_dict()

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
        'price': [float(str(np.around(x, 2))) for x in np.arange(1.99, 50.01, 1.00)],  # in 1 Euro steps
        'productid': range(10, 26)
        }

    
    arms = list(arm_product(combined))

    return arms


def create_bayesian_arms(p_headers, p_adtypes, p_colors, p_productids, mu, n):

    arms = []
    
    for i in range(n):

        header_msk = np.random.multinomial(1, p_headers)
        headers = np.array([5, 15, 35])
        header = headers[np.where(header_msk)][0]

        adtype_msk = np.random.multinomial(1, p_adtypes)
        adtypes = np.array(['skyscraper', 'square', 'banner'])
        adtype = adtypes[np.where(adtype_msk)][0]

        color_msk = np.random.multinomial(1, p_colors)
        colors = np.array(['green', 'blue', 'red', 'black', 'white'])
        color = colors[np.where(color_msk)][0]

        productid_msk = np.random.multinomial(1, p_productids)
        productids = np.array(range(10, 26))
        productid = productids[np.where(productid_msk)][0]
        
        price = float(
            str(
                np.around(
                    np.min([50, 
                    np.max(
                        [0, np.random.normal(mu, 10)]
                    )]),
                    2)))

        combined = {
            'header': header,
            'adtype': adtype,
            'color': color, 
            'productid': productid,
            'price': price,
            }

        arms.append(combined)

    arms_df = pd.DataFrame(arms)

    return arms_df


def main():

    all_arm_properties = create_all_arm_properties()
    all_arms = [BanditArm(prop) for prop in all_arm_properties]

    for i in range(10000):
        print(all_arms[1903].draw(57, i))
        
    
    # print(all_arms[432])
    # getcontext.proposepage()
    #big_df = join_df(57)
    #big_df.to_csv("context_57.csv", index=False)


if __name__ == "__main__":

    main()

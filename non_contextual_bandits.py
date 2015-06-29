
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

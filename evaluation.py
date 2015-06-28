import pandas as pd
import numpy as np
import bandits

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
        # reward = policy.arms[chosen_arm].draw(runid, idx)

        reward = policy.draw(runid, idx)
        
        rewards[idx] = reward

        print(reward)
        
        policy.update(chosen_arm, reward)

    results = pd.DataFrame({"chosen_arms": chosen_arms,
                            "rewards": rewards})

    results.to_csv("results" + str(runid) + ".csv")

    return results


runid = 188

df = pd.read_csv("results/results_ucb" + str(runid) + ".csv")

print(df.rewards.mean())


def main():

    ids = range(10000)

    all_arm_properties = bandits.create_all_arm_properties()
    all_arms = [bandits.BanditArm(prop) for prop in all_arm_properties]

    linear_policy = bandits.LinearPolicy(all_arms, "contexts/context_188.csv")
    test_policy(linear_policy, 188, ids)

if __name__ == "__main__":

    main()

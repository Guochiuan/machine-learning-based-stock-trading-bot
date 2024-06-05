""""""
"""  		  	   		  		 			  		 			 	 	 		 		 	
Template for implementing QLearner  (c) 2015 Tucker Balch  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Copyright 2018, Georgia Institute of Technology (Georgia Tech)  		  	   		  		 			  		 			 	 	 		 		 	
Atlanta, Georgia 30332  		  	   		  		 			  		 			 	 	 		 		 	
All Rights Reserved  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Template code for CS 4646/7646  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Georgia Tech asserts copyright ownership of this template and all derivative  		  	   		  		 			  		 			 	 	 		 		 	
works, including solutions to the projects assigned in this course. Students  		  	   		  		 			  		 			 	 	 		 		 	
and other users of this template code are advised not to share it with others  		  	   		  		 			  		 			 	 	 		 		 	
or to make it available on publicly viewable websites including repositories  		  	   		  		 			  		 			 	 	 		 		 	
such as github and gitlab.  This copyright statement should not be removed  		  	   		  		 			  		 			 	 	 		 		 	
or edited.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
We do grant permission to share solutions privately with non-students such  		  	   		  		 			  		 			 	 	 		 		 	
as potential employers. However, sharing with other current or future  		  	   		  		 			  		 			 	 	 		 		 	
students of CS 7646 is prohibited and subject to being investigated as a  		  	   		  		 			  		 			 	 	 		 		 	
GT honor code violation.  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
-----do not edit anything above this line---  		  	   		  		 			  		 			 	 	 		 		 	
  		  	   		  		 			  		 			 	 	 		 		 	
Student Name: Tucker Balch (replace with your name)  		  	   		  		 			  		 			 	 	 		 		 	
GT User ID: gwang383 (replace with your User ID)  		  	   		  		 			  		 			 	 	 		 		 	
GT ID: 903760738 (replace with your GT ID)  		  	   		  		 			  		 			 	 	 		 		 	
"""

import random as rand

import numpy as np


class QLearner(object):
    def __init__(
            self,
            num_states=1000,
            num_actions=3,
            alpha=0.2,
            gamma=0.95,
            rar=0.5,
            radr=0.99,
            dyna=200,
            verbose=False,
    ):
        self.verbose = verbose
        self.num_states = num_states
        self.num_actions = num_actions
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q_table = np.zeros([self.num_states, self.num_actions])
        self.reward = np.zeros([self.num_states, self.num_actions])
        self.transition = np.ones([num_states, num_actions, num_states])

    def querysetstate(self, s):

        self.s = s
        self.a = np.random.randint(self.num_actions)
        return self.a

    def query(self, s_prime, r):
        if self.dyna > 0:
            self.reward[self.s, self.a] = (1 - self.alpha) * self.reward[self.s, self.a] + self.alpha * r
            self.transition[self.s, self.a, s_prime] = self.transition[self.s, self.a, s_prime] + 1

            for i in range(self.dyna):
                star_s = np.random.randint(self.num_states)
                star_a = np.random.randint(self.num_actions)
                new_s = np.argmax(self.transition[star_s, star_a])

                self.q_table[star_s, star_a] = (1 - self.alpha) * self.q_table[star_s, star_a] \
                                               + self.alpha * self.reward[star_s, star_a] + self.alpha * self.gamma * self.find_best_reward(new_s)

                # self.q_table[star_s, star_a] = (1 - self.alpha) * self.q_table[star_s, star_a] \
                #                                + self.alpha * self.reward[star_s, star_a] + self.alpha * self.gamma * max(self.q_table[new_s])

        elif self.dyna < 0:
            print("dyna cannot be negative, so we set it to 0")

        self.q_table[self.s, self.a] = (1 - self.alpha) * self.q_table[self.s, self.a] + \
                                       self.alpha * r + self.alpha * self.gamma * self.find_best_reward(s_prime)

        # self.q_table[self.s, self.a] = (1 - self.alpha) * self.q_table[self.s, self.a] + \
        #                                self.alpha * r + self.alpha * self.gamma * max(self.q_table[s_prime])


        self.a = np.argmax(self.q_table[s_prime])
        if np.random.random() <= self.rar:
            self.a = np.random.randint(self.num_actions)

        self.rar *= self.radr
        self.s = s_prime
        return self.a

    def test_query(self, s_prime):
        action = np.argmax(self.q_table[int(s_prime)])
        return action



    # find best reward if s is given, and a is unknown
    def find_best_reward(self, s):
        return self.q_table[s, np.argmax(self.q_table[s])]
        # return np.max(self.q_table[s])
    def author(self):
        return "gwang383"

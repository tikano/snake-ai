import numpy as np
import utils


class Agent:
    def __init__(self, actions, Ne=40, C=40, gamma=0.7, display_width=18, display_height=10):
        # HINT: You should be utilizing all of these
        self.actions = actions
        self.Ne = Ne  # used in exploration function
        self.C = C
        self.gamma = gamma
        self.display_width = display_width
        self.display_height = display_height
        self.reset()
        # Create the Q Table to work with
        self.Q = utils.create_q_table()
        self.N = utils.create_q_table()
        
    def train(self):
        self._train = True
        
    def eval(self):
        self._train = False

    # At the end of training save the trained model
    def save_model(self, model_path):
        utils.save(model_path, self.Q)
        utils.save(model_path.replace('.npy', '_N.npy'), self.N)

    # Load the trained model for evaluation
    def load_model(self, model_path):
        self.Q = utils.load(model_path)

    def reset(self):
        # HINT: These variables should be used for bookkeeping to store information across time-steps
        # For example, how do we know when a food pellet has been eaten if all we get from the environment
        # is the current number of points? In addition, Q-updates requires knowledge of the previously taken
        # state and action, in addition to the current state from the environment. Use these variables
        # to store this kind of information.
        self.points = 0
        self.s = None
        self.a = None
    
    def update_n(self, state, action):
        # TODO - MP11: Update the N-table. 
        if state != None and action != None:
            food_dir_x = state[0]
            food_dir_y = state[1]
            adjoining_wall_x = state[2]
            adjoining_wall_y = state[3]
            adjoining_body_top = state[4]
            adjoining_body_bottom = state[5]
            adjoining_body_left = state[6]
            adjoining_body_right = state[7]
            self.N[food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, action] = self.N[food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, action] + 1
            
            
        pass

    def update_q(self, s, a, r, s_prime):
        # TODO - MP11: Update the Q-table. 
        if s != None and a != None:
            food_dir_x = s[0]
            food_dir_y = s[1]
            adjoining_wall_x = s[2]
            adjoining_wall_y = s[3]
            adjoining_body_top = s[4]
            adjoining_body_bottom = s[5]
            adjoining_body_left = s[6]
            adjoining_body_right = s[7]
            food_dir_xp = s_prime[0]
            food_dir_yp = s_prime[1]
            adjoining_wall_xp = s_prime[2]
            adjoining_wall_yp = s_prime[3]
            adjoining_body_topp = s_prime[4]
            adjoining_body_bottomp = s_prime[5]
            adjoining_body_leftp = s_prime[6]
            adjoining_body_rightp = s_prime[7]
            alpha = self.C/(self.C + self.N[food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, a])
            currentq = self.Q[food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, a]
            maximum = -1000000
            for i in range(4):
                newq = self.Q[food_dir_xp, food_dir_yp, adjoining_wall_xp, adjoining_wall_yp, adjoining_body_topp, adjoining_body_bottomp, adjoining_body_leftp, adjoining_body_rightp, i]
                if newq > maximum: maximum = newq
            update = currentq + alpha * (r + self.gamma * maximum - currentq)
            self.Q[food_dir_x, food_dir_y, adjoining_wall_x, adjoining_wall_y, adjoining_body_top, adjoining_body_bottom, adjoining_body_left, adjoining_body_right, a] = update
            

    def act(self, environment, points, dead):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        :param points: float, the current points from environment
        :param dead: boolean, if the snake is dead
        :return: chosen action between utils.UP, utils.DOWN, utils.LEFT, utils.RIGHT

        Tip: you need to discretize the environment to the state space defined on the webpage first
        (Note that [adjoining_wall_x=0, adjoining_wall_y=0] is also the case when snake runs out of the playable board)
        '''
       
        s_prime = self.generate_state(environment)
        food_dir_xp = s_prime[0]
        food_dir_yp = s_prime[1]
        adjoining_wall_xp = s_prime[2]
        adjoining_wall_yp = s_prime[3]
        adjoining_body_topp = s_prime[4]
        adjoining_body_bottomp = s_prime[5]
        adjoining_body_leftp = s_prime[6]
        adjoining_body_rightp = s_prime[7]
        reward = -0.1
        if(points > self.points): reward = 1
        elif(dead): 
            reward =  -1
            
        
        
            
        

        
        # TODO - MP12: write your function here
        if(self.s != None and self.a != None):
            
            self.update_n(self.s, self.a)
            self.update_q(self.s, self.a, reward, s_prime)
        
        
        maximum = -100000
        action = 1;
        last = -1;
        
        if(dead):
            self.reset()
            self.points = points
            return utils.UP
        
        for i in range(0, 4):
            if(self.N[s_prime + (i,)] < self.Ne): 
                action = i
                last = i
            elif(self.Q[s_prime + (i,)] >= maximum):
                maximum = self.Q[s_prime + (i,)]
                if last == -1:
                    action = i

            
        self.a = action
        self.s = s_prime
        self.points = points
            
        return action


    def generate_state(self, environment):
        '''
        :param environment: a list of [snake_head_x, snake_head_y, snake_body, food_x, food_y, rock_x, rock_y] to be converted to a state.
        All of these are just numbers, except for snake_body, which is a list of (x,y) positions 
        '''
        # TODO - MP11: Implement this helper function that generates a state given an environment 
        returnstate = [0, 0, 0, 0, 0, 0, 0, 0]
        snake_head_x = environment[0]
        snake_head_y = environment[1]
        snake_body = environment[2]
        food_x = environment[3]
        food_y = environment[4]
        rock_x = environment[5]
        rock_y = environment[6]
        if food_x == snake_head_x: returnstate[0] = 0
        elif food_x > snake_head_x: returnstate[0] = 2
        else: returnstate[0] = 1
        
        if food_y == snake_head_y: returnstate[1] = 0
        elif food_y > snake_head_y: returnstate[1] = 2
        else: returnstate[1] = 1
        
        if snake_head_x == 1 or (snake_head_x == rock_x + 2 and snake_head_y == rock_y): returnstate[2] = 1
        elif snake_head_x == self.display_width - 2 or (snake_head_x == rock_x - 1 and snake_head_y == rock_y): returnstate[2] = 2
        
        if snake_head_y == 1 or (snake_head_y == rock_y + 1 and rock_x <= snake_head_x <= rock_x + 1): returnstate[3] = 1
        elif snake_head_y == self.display_height - 2 or (snake_head_y == rock_y - 1 and rock_x <= snake_head_x <= rock_x + 1): returnstate[3] = 2
        
        if (snake_head_x, snake_head_y - 1) in snake_body: returnstate[4] = 1
        if (snake_head_x, snake_head_y + 1) in snake_body: returnstate[5] = 1
        if (snake_head_x - 1, snake_head_y) in snake_body: returnstate[6] = 1
        if (snake_head_x + 1, snake_head_y) in snake_body: returnstate[7] = 1
        
        return tuple(returnstate)

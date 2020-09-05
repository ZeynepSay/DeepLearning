import numpy as np
import math
import pylab 


class CartPoleSystem:
    def __init__(self):
       
        self.control_interval = 0.01 # control interval
        self.init_condition = np.array([-1.0, 0.25, 0.3, -0.7]) #position,  velocity,  angle,  angular velocity
        self.n_vector = np.array([0.004, 0.04, 0.001, 0.01]) # noise vector
        self.cart_position = [-0.1,0.1]  # cart position  
        self.cart_angle = [-0.05,0.05]  # pole angle 
        self.mass_cart = 6   
        self.mass_pole = 3 
        self.gravity = 9.81 
        self.length = 0.8  # pole length
        self.cart_pole = self.mass_pole + self.mass_cart
        self.mass_length = self.mass_pole*self.length
    

    def get_positions(self):
         
        sin = np.sin
        cos = np.cos 
        #Initiliaze state vector 
        new_x = self.init_condition[0]        #position of the cart along the track
        new_x_dot = self.init_condition[1]    #velocity of Cart
        new_angle = self.init_condition[2]    #cart angle
        new_theta_dot = self.init_condition[3] #angle velocity- angle change between cart and pole -
        new_position = []
        cart_force = 0
        N = 1/self.control_interval
        zero_mean = np.array([0, 0, 0, 0])
        
        for i in range(int(N)):
            
            n_vector = np.random.normal(zero_mean, self.n_vector)
            angle_acceleration = (self.gravity*sin(new_angle)*(self.cart_pole) - (cart_force + self.mass_length*(new_theta_dot**2)*sin(new_angle))*cos(new_angle)) / ((4/3)*self.length*(self.cart_pole) - self.mass_length*cos(new_angle)**2)
            position_acceleration = (cart_force - self.mass_length*(angle_acceleration*cos(new_angle) - 
                                sin(new_angle)*new_theta_dot**2))/(self.cart_pole)
            new_x = new_x + self.control_interval*new_x_dot + n_vector[0]
            new_x_dot = new_x_dot + self.control_interval*position_acceleration + n_vector[1]
            new_angle = new_angle + self.control_interval*new_theta_dot + n_vector[2]
            new_theta_dot = new_theta_dot + self.control_interval*angle_acceleration + n_vector[3]
            new_position.append([new_x, new_x_dot, new_angle, new_theta_dot])
            
        return new_position 
    
    def get_moves(self,k1,k2,k3,k4, t_steps):
        reward = 0
        new_x = self.init_condition[0]   #cart position
        new_x_dot = self.init_condition[1] #cart velocity
        new_angle = self.init_condition[2] #cart angle
        new_theta_dot = self.init_condition[3] #-angle velocity-angle change between cart and pole 
        new_position = []
        points = []
        next_moves = []
        N = t_steps
        sin = np.sin
        cos = np.cos
        zero_mean = np.array([0, 0, 0, 0])
      
        for i in range(N):
            feedback_policy= k1*new_x + k2*new_x_dot + k3*new_angle + k4*new_theta_dot
            cart_force = min(100,max(-100, feedback_policy))
            angle_acceleration = (self.gravity*sin(new_angle)*(self.cart_pole) - (cart_force + self.mass_length*(new_theta_dot**2)*sin(new_angle))*cos(new_angle)) /((4/3)*self.length*(self.cart_pole) - self.mass_length*cos(new_angle)**2)
            position_acceleration = (cart_force - self.mass_length*(angle_acceleration*cos(new_angle) - 
                                sin(new_angle)*new_theta_dot**2))/(self.cart_pole)
            n_vector = np.random.normal(zero_mean, self.n_vector)
            new_x = new_x + self.control_interval*new_x_dot + n_vector[0]
            new_x_dot = new_x_dot + self.control_interval*position_acceleration + n_vector[1]
            new_angle = new_angle + self.control_interval*new_theta_dot + n_vector[2]
            new_theta_dot = new_theta_dot + self.control_interval*angle_acceleration + n_vector[3]
            new_position.append([new_x, new_x_dot, new_angle, new_theta_dot])
            next_moves.append(cart_force)
            if not (self.cart_angle[0] <= new_angle and new_angle <= self.cart_angle[1]):
                reward = -1
            elif not (self.cart_position[0] <= new_x and new_x <= self.cart_position[1]):
                reward = -1
            else:
                reward = 0
            points.append(reward)
            if abs(new_angle) > 1 or abs(new_x) > 5:
                point_final = -(N-i)
                return points, new_position, next_moves, point_final
            
        return points, new_position, next_moves, point_final
    
    def get_value(self, val):
        return 1.0 / (1.0 + np.exp(-val))
    
    def get_policy(self, new_position, new_angle):
        move_r = self.get_value(np.dot(new_position, np.transpose(new_angle)))
        return [1-move_r, move_r]
    
    def get_gradient(self, points, next_moves, new_position, new_angle):
        '''calcutate policy with gradient'''
        init_val_g = 0
        e_length = len(points)
        for i in range(e_length):
            policy_value = self.get_policy(new_position[i], new_angle)
            action_value = next_moves[i]
            reward_value = sum(points[i::])
            if action_value == 0:
                policy_with_gradient = - policy_value[1] * np.asarray(new_position[i]) * reward_value
            else:
                policy_with_gradient = policy_value[0] * np.asarray(new_position[i]) * reward_value
            init_val_g = init_val_g + policy_with_gradient
        return init_val_g
    
    def learning_policy(self, new_angle, t_steps, e_steps, init_steps):
        '''  '''
        for i in range(e_steps):
            points, new_position, next_moves, trained_points = self.get_moves(new_angle[0], new_angle[1], new_angle[2], new_angle[3], t_steps)
            print(len(points))
            policy_gradient = self.get_gradient(points, next_moves, new_position, new_angle)
            step_size = init_steps / (1 + i)
            new_angle = new_angle + step_size * policy_gradient

        return new_angle
    
    def policy_estimate(self, new_angle, t_steps, e_steps):
        noise = []
        reward_i = []
        init_val_g = [0.0,0.0,0.0,0.0]
        for i in range(len(new_angle)):
            points, new_position, next_moves, trained_points = self.get_moves(new_angle[0], new_angle[1], new_angle[2], new_angle[3], t_steps)
            init_reward = points[len(points)-1]
            for e in range(e_steps):
                angle_next = np.random.rand()*self.n_vector[i]*2-self.n_vector[i]
                noise.append(angle_next)
                new_angle[i] = new_angle[i] + angle_next
                points, new_position, next_moves, trained_points = self.get_moves(new_angle[0], new_angle[1], new_angle[2], new_angle[3], t_steps)
                reward_next = points[len(points)-1]
                reward_i.append(reward_next)
            init_val_g[i] = np.sum(np.asarray(noise)*(np.asarray(reward_i)-init_reward))/np.sum(np.asarray(noise))**2
        return init_val_g


def draw_Gradient():

    cart_pole_env = CartPoleSystem()
    new_position = cart_pole_env.get_positions()
    new_position = np.asarray(new_position)
   
    new_x = np.linspace(0,1,100)
    pylab.plot(new_x, new_position[:,0], 'purple', label='Position of Cart')
    pylab.plot(new_x, new_position[:,1], 'green', label='Velocity of Cart')
    pylab.plot(new_x, new_position[:,2], 'brown', label='Angle of Cart')
    pylab.plot(new_x, new_position[:,3], 'darkblue', label='Anguler velocity of Cart')
    print("\nExercise 6.1 Visualize state trajectory over time")
    pylab.grid()
    pylab.legend(loc='upper left')
    pylab.ylim(-1.5, 3.0)
    pylab.show()
  
def draw_states():
  
    t_steps = 1000
    cart_pole_env = CartPoleSystem()
    k1 = 1.0
    k2 = 1.0
    k3 = 1.0
    k4 = 1.0
    reward, new_position, next_moves, new_point = cart_pole_env.get_moves(k1,k2,k3,k4, t_steps)
    new_position = np.asarray(new_position)
   
    new_x = np.linspace(0,1,len(new_position))
    pylab.plot(new_x, new_position[:,0], 'purple', label='Position of Cart')
    pylab.plot(new_x, new_position[:,1], 'green', label='Velocity of Cart')
    pylab.plot(new_x, new_position[:,2], 'brown', label='Angle of Cart')
    pylab.plot(new_x, new_position[:,3], 'darkblue', label='Anguler velocity of Cart')
    print("\nExercise 6.2 Visualize State trajectory of the system")
    pylab.grid()
    pylab.legend(loc='upper left')
    pylab.ylim(-1.5, 3.0)
    pylab.show()
    print("\n Reward ")
    print( str(new_point))
    print("\n Timesteps")
    print(str(len(new_position)))
 
    
    episode_maximum = 10
    new_angle = np.array([k1,k2,k3,k4])
    alpha = 0.01
    init_steps = 0.01
    new_angle = cart_pole_env.learning_policy(new_angle, t_steps, episode_maximum, init_steps)
    print("\nExercise 6.2 Visualize State trajectory of the system")
    print(new_angle)
    points, new_position, next_moves, new_point = cart_pole_env.get_moves(new_angle[0], new_angle[1], new_angle[2], new_angle[3], t_steps)
    print("\n Reward ")
    print( str(new_point))
    print("\n Timesteps")
    print(str(len(new_position)))
    
    new_position = np.asarray(new_position)
    new_x = np.linspace(0,len(new_position)*0.01,len(new_position))
    pylab.plot(new_x, new_position[:,0], 'purple', label='Position of Cart')
    pylab.plot(new_x, new_position[:,1], 'green', label='Velocity of Cart')
    pylab.plot(new_x, new_position[:,2], 'brown', label='Angle of Cart')
    pylab.plot(new_x, new_position[:,3], 'darkblue', label='Anguler velocity of Cart')
    print("\nExercise 6.3 Visualize states over time")
    pylab.grid()
    pylab.legend(loc='upper left')
    pylab.ylim(-4, 4.0)
    pylab.show()
    print("\nExercise 6.3 State trajectory of the system")
    print(new_angle)
    
    
    
    
    
    

if __name__ == "__main__":
    draw_Gradient()
    draw_states()
    
   
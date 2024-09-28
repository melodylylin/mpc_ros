import numpy as np
try:
    from ompl import base as ob
    from ompl import geometric as og
except ImportError:
    print("Could not find OMPL")
    raise ImportError("Run inside docker!!")
import matplotlib.pyplot as plt 

class ValidityChecker(ob.StateValidityChecker):
    '''A class to check if an obstacle is in collision or not.
    '''
    def __init__(self, si, obs_list, obs_r, robot_radius=0.5):
        '''
        Intialize the class object, with the current map and mask generated
        from the transformer model.
        :param si: an object of type ompl.base.SpaceInformation
        :param CurMap: A np.array with the current map.
        :param MapMask: Areas of the map to be masked.
        '''
        super().__init__(si)
        self.obs_list = obs_list
        self.obs_r = obs_r
        self.robot_r = robot_radius

    def isValid(self, state):
        '''
        Check if the given state is valid.
        :param state: An ob.State object to be checked.
        :returns bool: True if the state is valid.
        '''
        for (ox, oy) in self.obs_list:
            d = (ox - state[0])**2 + (oy-state[1])**2
            if d <= (self.obs_r + self.robot_r)**2:
                return False  # collision

        return True  # safe

def plot_circle(x, y, obs_r, color="-b"):  # pragma: no cover
    deg = list(range(0, 360, 5))
    deg.append(0)
    xl = [x + obs_r * np.cos(np.deg2rad(d)) for d in deg]
    yl = [y + obs_r * np.sin(np.deg2rad(d)) for d in deg]
    plt.plot(xl, yl, color)

def plan_rrt_star(start, goal, obs_list, obs_r, robot_r, map_bound, plan_time = 0.1):
    # Define the space
    space = ob.RealVectorStateSpace(2)
    bounds = ob.RealVectorBounds(2)
    bounds.setLow(0,-map_bound[0])
    bounds.setLow(1,-map_bound[1])
    bounds.setHigh(0, map_bound[0]) # Set width bounds (x)
    bounds.setHigh(1, map_bound[1]) # Set height bounds (y)
    space.setBounds(bounds)


    # Define the SpaceInformation object.
    si = ob.SpaceInformation(space)
    validity_checker = ValidityChecker(si, obs_list, obs_r, robot_r)
    si.setStateValidityChecker(validity_checker)

    # Create a simple setup
    ss = og.SimpleSetup(si)

    # Use RRT*
    planner = og.RRTstar(si)

    # Set the start and goal states:
    start_state = ob.State(space)
    start_state[0] = start[0]
    start_state[1] = start[1]
    goal_state = ob.State(space)
    goal_state[0] = goal[0]
    goal_state[1] = goal[1]
    ss.setStartAndGoalStates(start_state, goal_state, 0.1)
    ss.setPlanner(planner)

    # Attempt to solve within the given time
    solved = ss.solve(plan_time)
    if ss.haveExactSolutionPath():
        print("Found solution")
        path = [
            [ss.getSolutionPath().getState(i)[0], ss.getSolutionPath().getState(i)[1]]
            for i in range(ss.getSolutionPath().getStateCount())
            ]
        # Define path
        # Get the number of interpolation points
        num_points = int(4*ss.getSolutionPath().length())#//(dist_resl*32))
        ss.getSolutionPath().interpolate(num_points)
        path_obj = ss.getSolutionPath()
        path_interpolated = np.array([
            [path_obj.getState(i)[0], path_obj.getState(i)[1]] 
            for i in range(path_obj.getStateCount())
            ])
        status = True
    else:
        path = [[start[0], start[1]], [goal[0], goal[1]]]
        path_interpolated = []
        status = False

    return path, path_interpolated, status    


def main():
    print("Dubins path planner sample start!!")
    import matplotlib.pyplot as plt
    map_bound = (10,10)
    obs_list = [(4,0), (8,5), (6,9), (2, -4), (8,-5), (6,-9), (5, -6)]

    start = [0.0, 0.0]
    goal = [5.0,-7.5]
    obs_r = 0.5
    robot_r = 0.5

    path, path_interp, status = plan_rrt_star(start, goal, obs_list, obs_r, robot_r, map_bound)


    for (ox, oy) in obs_list:
        plot_circle(ox, oy, obs_r)

    for p in path:
        plt.plot(p[0], p[1], "xb")


    plt.plot(start[0], start[1], "xr")
    plt.plot(goal[0], goal[1], "xr")
    plt.axis("equal")
    plt.grid(True)
    plt.show()   



if __name__ == '__main__':
    main()
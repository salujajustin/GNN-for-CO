import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class tsp_generator:
    def __init__(self, map_size = (100, 100)):
        self.map_x = map_size[0]
        self.map_y = map_size[1]
        self.tsp_coords = []
        self.opt_tour = None
        self.max_coordinates = max(self.map_x,self.map_y)
        self.min_coordinates = min(self.map_x,self.map_y)
    
    def generate(self, TSP_nodes, seed_gen = 0 , distribution="uniform", mode = np.float):
        self.num_nodes = TSP_nodes
        self.seed_gen = seed_gen
        self.distribution = distribution
        np.random.seed(self.seed_gen)
        if distribution == 'uniform':
            tsp_coords = np.random.uniform(0,self.min_coordinates,size = (self.num_nodes,2))
            tsp_coords = np.array(tsp_coords,dtype = mode)        
        if distribution == 'normal':
            mu = self.min_coordinates//2
            sigma = int(self.min_coordinates*0.2)
            tsp_coords = np.random.normal(mu,sigma,size = (self.num_nodes,2))
            tsp_coords = np.array(tsp_coords,dtype = mode)
        self.tsp_coords = abs(tsp_coords).astype(float)
        return self.tsp_coords 
    
    
    def visualize_tsp(self, save_plt = False, fig_name = 'tsp_gen'):
        if len(self.tsp_coords) == 0:
            print("generate TSP first")
        else:
            fig, ax = plt.subplots(nrows=1, ncols=1)
            ax.scatter(self.tsp_coords[:,0],self.tsp_coords[:,1], c= 'r')            
            plt.show()
            if save_plt:
                fig.savefig(fig_name+'_'+self.distribution+'_'+str(self.seed_gen),dpi = 300)        
            plt.close()
        
    def write_tsp(self,data_name = 'tsp_coords'):
        node_id = np.arange(1,self.tsp_coords.shape[0]+1).reshape([-1,1])
        out_arr = np.concatenate((node_id,self.tsp_coords),axis = 1)
        names = ['node_id','x_coord','y_coord']
        df = pd.DataFrame(out_arr, index=None, columns=names)
        file_name = data_name+'_'+self.distribution+'_'+str(self.seed_gen)+'.csv'
        contents = df.to_csv(index=False, header=False, sep=',')
        with open(file_name, "w") as f:
        	f.write("NAME: " + data_name + "\n")
        	f.write("TYPE: TSP\n")
        	f.write("COMMENT: " + str(self.num_nodes) + " locations\n")
        	f.write("DIMENSION: " + str(self.num_nodes) + "\n")
        	f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
        	f.write("NODE_COORD_SECTION\n")
        	f.write(contents)
        f.close()

    
    def solve_tsp(solver):
        """
        Maybe incorporate the solvers in this calss too?
        we can write a function to save the TSP optimal path too
        """
        #from concorde.tsp import TSPSolver
		#data = "tsp_coords_uniform_0.csv" 
		#solver = TSPSolver.from_tspfile(data)
		#solution = solver.solve()

#        self.opt_tour = 
        return None
    
    def save_opt_tour():
        """
        Save the optimal tour as a csv file
        """
        return None
    
if __name__ == "__main__":
    tsp = tsp_generator()
    for i in range(1):
        tsp_coords = tsp.generate(TSP_nodes = 50 ,seed_gen = i, distribution = 'uniform',mode = np.int)
        tsp.visualize_tsp(save_plt =False)
        tsp.write_tsp()
    

class Node:
    position=[]
    type="vehicle"
    P_n=0.6 # w
    E_n=0
    bandwidth=1 # MHz
    def __init__(self,pos=[0,0,0],type="vehicle"):
        self.position=pos
        self.type=type
        if(self.type=="vehicle"):
            self.E_n=4e7 # cycle/s
            self.P_n=1 # w
        else:
            self.E_n=2e7 # cycle/s
            self.P_n = 0.6  # w

import numpy as np
class Multivariate_Optimization:
    def __init__(self,start,direction,range):
        self.start=np.array(start)
        self.direction=np.array(direction)
        self.range=range
    
    def bracketing(self,N,fn):
        for i in range(N-2):
            #print("ROUND",i)
            if fn(self.w_all[i]) >= fn(self.w_all[i+1]) <= fn(self.w_all[i+2]):
                #print("ROUND",i)
                return np.array([self.w_all[i],self.w_all[i+2]]),i
                
        if fn(self.w_all[0]) > fn(self.w_all[-1]):
            print(f"Highest Value at {self.w_all[0]}")
            return self.w_all[0]
        else:
            print(f"Highest Value at {self.w_all[-1]}")
            return self.w_all[-1]

    def region_elimination(self, range, e, fn):
        a= range[0]
        b= range[1]
        #print(a,b)
        #print((a-b)/4)
        L = (b-a)/4
        #print(f"L:{L}")

        if abs(np.average(L))<e:
            return (a+b)/2
        
        wm = (a+b)/2

        w1 = a + L
        w2 = b - L
        
        #print(f"w1:{w1}\nw2:{w2}\nwm:{wm}")

        if fn(w1) < fn(wm):
            return self.region_elimination(np.array([a, wm]), e, fn)
        elif fn(w2) < fn(wm):
            return self.region_elimination(np.array([wm, b]), e, fn)
        else:
            return self.region_elimination(np.array([w1,w2]), e, fn)

    def line_search(self,N,e,fn):
        self.alpha = np.linspace(0,self.range,N)
        self.w_all=[]
        for i in self.alpha:
            self.w = self.start+self.direction*i
            self.w_all+=[list(self.w)]
        self.w_all=np.array(self.w_all)
        #print(self.bracketing(N,fn))
        #print(self.w_all)
        if len(self.bracketing(N,fn))==2:
            new_value, i = self.bracketing(N,fn)
            #print(new_value)
            min_value = self.region_elimination(new_value, e, fn)
            return min_value,self.alpha[i]
        else:
            return self.bracketing(N,fn)
        

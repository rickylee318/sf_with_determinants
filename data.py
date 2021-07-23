import numpy as np
import pandas as pd
from scipy.stats import truncnorm


class Data:
    """empirical US bank data or simulation data in 4SFwD
    """
    def __init__(self):
        pass
    
    def raw_data(self, data_name):
        data = pd.read_csv('data/'+data_name)
        data.index = data['Unnamed: 0']
        data.drop(columns = 'Unnamed: 0', inplace=True)
        return data
        
    def _numpy_triu2(self, df):          
        a = df.values
        r,c = np.triu_indices(a.shape[1],0)
        cols = df.columns
        nm = [cols[i]+cols[j] for i,j in zip(r,c)]
        return pd.DataFrame(a[:,r] * a[:,c], columns=nm, index = df.index)

    def _sim_param(self):
        self.true_beta = np.array([1,1])
        self.true_delta = np.array([-0.3,0.5])
        self.true_xi = np.array([0.5,-0.3])
        self.true_z = np.array([-1,1])
        self.true_gamma = np.array([1,-1])
        self.true_delta = np.array([1,1])
        self.true_xi = np.array([1,1])
        self.true_z = np.array([1,1])
        self.true_gamma = np.array([1,1])
        self.true_sigma_v=0.5
        self.true_sigma_alpha = 0.5

    def simulate(self,N,T):
        NT = N * T
        K = 2
        #gen_param
        self._sim_param()
        #data simulation
        x = np.concatenate([np.ones([NT,1]),np.random.normal(0,1,[NT,1])],axis=1)
        pi = np.concatenate([np.ones([NT,1]),np.random.normal(0,1,[NT,1])],axis=1)
        w = np.concatenate([np.ones([N,1]),np.random.normal(0,1,[N,1])],axis=1)

        myclip_a = 0
        my_mean = np.dot(pi,self.true_delta)
        my_std = np.exp(np.dot(pi,self.true_xi)/2)
        a, b = (myclip_a - my_mean) / my_std, np.inf
        u = truncnorm.rvs(a,b,loc = my_mean, scale = my_std)

        myclip_a = 0
        my_mean = np.dot(w,self.true_z)
        my_std = np.exp(np.dot(w,self.true_gamma)/2)
        a, b = (myclip_a - my_mean) / my_std, np.inf
        eta = truncnorm.rvs(a,b,loc = my_mean, scale = my_std)

        v = np.random.normal(0, self.true_sigma_v,[NT,])
        alpha = np.random.normal(0, self.true_sigma_alpha, [N,])

        y = np.dot(x,self.true_beta) + v + u + np.kron(eta, np.ones([T,])) + np.kron(alpha,np.ones([T,]))
        return y,x,w,pi

    def preprcessing(self, data_name, per_det, tra_det):
        data = self.raw_data(data_name)
        #ensure price homogeneity
        data['TC'] = data['TC'] - data['p1']
        data['p2'] = data['p2'] - data['p1']
        data['p3'] = data['p3'] - data['p1']
        df = data[['p2','p3','y1','y2','y3','y4','y5','T']]
        df1 = self._numpy_triu2(df)
        data = pd.concat([data[['p2','p3','y1','y2','y3','y4','y5','T','TA','TC','E/A','ROA']],df1], axis=1)

        data = data.iloc[np.tile(np.array([0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1], dtype = bool),int(data.shape[0]/20)),:]
        data['T'] = data['T']-8
        data.loc[:,'y1T'] = data['T'] * data['y1']
        data.loc[:,'y2T'] = data['T'] * data['y2']
        data.loc[:,'y3T'] = data['T'] * data['y3']
        data.loc[:,'y4T'] = data['T'] * data['y4']
        data.loc[:,'y5T'] = data['T'] * data['y5']
        #data.loc[:,'p1T'] = data['T'] * data['p1']
        data.loc[:,'p2T'] = data['T'] * data['p2']
        data.loc[:,'p3T'] = data['T'] * data['p3']
        data.loc[:,'TT'] = data['T'] ** 2
        T = 12

        #4
        npl = pd.read_csv('data/NPL')
        npl.index = npl['Unnamed: 0']
        data = pd.concat([data,npl.loc[data.index.unique().tolist()][['NPL']]], axis=1)

        NT = data.shape[0]
        N =  int(NT/ T)
        names = [xx for xx in data.columns.tolist() if xx not in ['TA','TC','E/A','ROA','NPL']]
        y = np.array(data.TC)
        x = np.array(data[names])
        x = np.concatenate([np.ones([NT,1]),x],axis=1)
        w = []
        for i in data.index.unique().tolist():
            w.append(data[per_det].loc[i].mean(axis=0))
        w = np.concatenate([np.ones([N,1]),pd.DataFrame(w)],axis=1)
        pi = np.array(data[tra_det])
        pi = np.concatenate([np.ones([NT,1]),pi],axis=1)
        return y,x,w,pi
        
if __name__ == '__main__':
    main()









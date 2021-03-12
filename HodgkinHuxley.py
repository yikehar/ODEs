"""
Hodgkin Huxley model
Copied from
https://qiita.com/tagut_19/items/5b55f0fb3b9034127699
"""

import numpy as np
import matplotlib.pyplot as plt

class BP:
    def __init__(self):
        self.Cm = 1.0
        self.gNa = 120.
        self.ENa = 45.
        self.gk = 36.
        self.Ek = -82.
        self.gL = 0.30
        self.EL = -54.387
        #-54.387??
        #単位はmV

        #それぞれの，self.Vを引数にした関数の宣言
    def am(self):
        return (0.1*(self.V+45.)/(1 - np.exp(-(self.V+45.)/10.)))
    def bm(self):
        return (4. * np.exp(-(self.V+70.)/18.))
    def bh(self):
        return 1./(1.+np.exp(-(self.V+40.)/10.))
    def ah(self):
        return 0.07 * np.exp(-(self.V+70.)/20.)
    def an(self):
        return 0.01 * (self.V+60.)/(1-np.exp(-(self.V+60.)/10))
    def bn(self):
        return 0.125* np.exp(-(self.V+70.)/80.)

    def Calculated_m_inf(self):
        return self.Calculated_inf(self.am(),self.bm())

    def Calculated_h_inf(self):
        return self.Calculated_inf(self.ah(),self.bh())

    def Calculated_n_inf(self):
        return self.Calculated_inf(self.an(),self.bn())

    def Calculated_inf(self,alpha,beta):
        return alpha/(alpha + beta)

    def calc(self):
        t_list = []
        V_list = []
        #リストの宣言
        t = 0

        #諸々の定数の宣言
        #以下の値を色々変える
        self.V = -69.996
        #-69.996[mV]で記述されている，初期値のこと
        self.dt = 0.01
        self.n = 1000

        #初期値をリストに格納する
        t_list.append(t)
        V_list.append(self.V)

        #以下の初期値は，inftyを計算しないといけない
        m = self.Calculated_m_inf()
        h = self.Calculated_h_inf()
        n = self.Calculated_n_inf()
       # print(self.V +"self.V")
        for i in range(self.n):
            y= np.sin(t)
            self.Istim = 30 * np.where(y>0,1,0)

            m = m +  (self.am()*(1-m)-self.bm()*m) * self.dt
            #これらは前のステップの値によって変わっていく
            h =  h + (self.ah() * (1-h) - self.bh()* h) * self.dt
            n =n +  (self.an() * (1-n) - self.bn() * n) * self.dt
            INa = self.gNa * (m**3) * h * (self.ENa - self.V)
            IK = self.gk * (n**4) * (self.Ek - self.V)
            IL = self.gL * (self.EL - self.V)
            #ここまでの計算は，ただの次のステップのself.Vの計算のための未知数
            self.V =  self.V + (INa + IK + IL + self.Istim)/self.Cm * self.dt
            #基本的なh-h方程式を差分法によって近似する
            #self.Vの更新
            t =  t + self.dt
            #dtだけ時間tを進める
            t_list.append(t)
            #進められたtを格納する
            V_list.append(self.V)

           # print(self.V)
           # print(m)
            #このステップで計算されたself.Vを格納する
            #解をもとめてプロットすることができないので，この方法をとるが，解があるならそのままプロットしてもよい

        #計算が終わったらプロットしてもらう
        plt.plot(t_list,V_list,label = 'H-H simulation')
        plt.xlabel('t[ms]' )
        plt.ylabel('V[mV]')


bp = BP() #インスタンスを作成
bp.calc()
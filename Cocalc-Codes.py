import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
import pandas as pd
from scipy.integrate import odeint
from PIL import Image


header = st.container()
dataset = st.container()
features = st.container()
description = st.container()
sird = st.container()
explanation = st.container()

def model(Sz,Ir,Iz,Rr,T):
#שיטת אוילר
        h = 0.1
        t= np.arange(0,T,h)#זמן
        beta = Ir#מקדם תחלואה
        gemma = Rr#מקדם החלמה

        I0 = 20
        R0 = 0
        S0 = 1180

        S = t*0
        I = t*0
        R = t*0

        S[0] = Sz
        I[0] = Iz
        R[0] = 0

        for i in range(0,len(t)-1):
            S[i+1] = (-beta * I[i] * S[i])*h+S[i]#בהנחה שהאוכלוסייה אינה גדלה, כמות הרגישים יכולה רק לקטון, או להישאר ניטרלית
            I[i+1] = (beta * I[i] * S[i] - gemma * I[i])*h+I[i]#ישנה גדילה חיובית ממחלקת הרגישים, וירידה לעומת מחלקת המבריאים
            R[i+1] = (gemma * I[i])*h + R[i]

        plt.grid()
        return np.array([[t,S],[t,I],[t,R]])




with header:

    st.title('welcome')

with dataset:
    st.header('This website was made by Gil Peleg and Michal Kalmanovich as a part of our S.I.R Research essay')
    st.text('We wrote some codes to simulate the S.I.R Model with Graphs, than used a Python ')
    st.text("library called steamlit in order to make the simulation a responsive website.")
    st.text('')
    st.header('')
    st.header('')
    sel_col,slide_col= st.columns(2)





with features:

    st.text('')
    st.header('S.I.R Simulator')
    S = slide_col.slider('The S value (the population thats vulnerable)',min_value=0, max_value = 10000,value = 1200,step=40)
    I = slide_col.slider('The I value (the infected population)',min_value=0, max_value = S,value =20,step=10)
    T =slide_col.slider('simulation run time (in days)',min_value=0, max_value = 10000,value =1000,step=500)
    Rr = sel_col.text_input("recovery rate",value=0.01)
    Ir = sel_col.text_input("infection rate",value=0.00005)
    if st.button("run s.i.r"):
        fig = plt.figure()
        ###plots
        data = model(float(S),float(Ir),float(I),float(Rr),T)
        #plt.style.use('dark_background')
        plt.plot(data[0,0],data[0,1], 'r', label="S")
        plt.plot(data[1,0],data[1,1], 'b', label="I")
        plt.grid()
        plt.plot(data[2,0],data[2,1], 'g', label="R")
        plt.legend(fontsize=17)
        #title and axis labels
        #fig, ax = plt.subplots()
        plt.xlim(0,T)
        plt.ylim(0,int((S+I)*1.2))
        plt.title('S.I.R Model Simulation', fontsize=18)
        plt.xlabel('Time (days)', fontsize=14)
        plt.ylabel('Population', fontsize=14)
        st.write(fig)
        plt.show()


with description:
    st.header('')
    st.header('')
    st.header('')
    st.header('')


def SIRD(ls, t, beta, gamma, mu):
        S = ls[0]
        I = ls[1]
        R = ls[2]
        D = ls[3]
        dSdt = -beta * S * I
        dIdt = beta * S * I - gamma * I - mu * I
        dRdt = gamma * I
        dDdt = mu * I
        
        return [dSdt, dIdt, dRdt, dDdt]

        
def sird_func(Sz,Ir,Iz,Rr,Dr,T):


    # Set initial conditions
    S0 = 9000
    I0 = 20
    R0 = 0
    D0 = 0

    # Set parameters
    beta = 0.00005#מקדם תחלואה
    gamma = 0.01#מקדם החלמה
    mu = 0.01# מקדם תמותה
    time = T
    h = 0.01
    t = np.arange(0,time,h)
    # Solve the SIRD model using odeint
    Q = odeint(SIRD, [Sz, Iz, R0, D0], t, args=(Ir,Rr,Dr))

    # Plot the results

    return np.array([[t,Q[:,0]],[t,Q[:,1]],[t,Q[:,2]],[t,Q[:,3]]])

with sird:
    time = 100
    sel_col2,slide_col2= st.columns(2)
    st.text('')
    st.header('S.I.R.D Simulator')
    S = slide_col2.slider('The S value in s.i.r.d(the population thats vulnerable)',min_value=0, max_value = 10000,value = 9000,step=40)
    I = slide_col2.slider('The I value in s.i.r.d(the infected population)',min_value=0, max_value = S,value =20,step=10)
    Rr = sel_col2.text_input("recovery rate in s.i.r.d",value=0.01)
    Dr = sel_col2.text_input("Death rate in s.i.r.d",value=0.01)
    Ir = sel_col2.text_input("infection rate in s.i.r.d",value=0.00005)
    Ti =slide_col2.slider('simulation run time in s.i.r.d(in days)',min_value=0, max_value = 3000,value =100,step=100)
    if st.button("run s.i.r.d"):
        fig = plt.figure()
        ###plots
        data =sird_func(float(S),float(Ir),float(I),float(Rr),float(Dr),Ti)
        #plt.style.use('dark_background')
        plt.plot(data[0,0],data[0,1], 'r', label="S")
        plt.plot(data[1,0],data[1,1], 'b', label="I")
        plt.plot(data[2,0],data[2,1], 'g', label="R")
        plt.plot(data[3,0],data[3,1], 'orange', label="D")
        plt.legend(fontsize=17)
        #title and axis labels
        #fig, ax = plt.subplots()
        plt.xlim(0,Ti)
        plt.grid()
        plt.ylim(0,int((S+I)*1.2))
        plt.title('S.I.R.D Model Simulation', fontsize=18)
        plt.xlabel('Time (days)', fontsize=14)
        plt.ylabel('Population', fontsize=14)
        st.write(fig)
        plt.show()

with explanation:
    st.header('')
    st.header('')
    st.header('')
    st.header('')
    st.header('development process')
    st.text('we wrote the solutions to the diffrential equations in cocalc,than')
    st.text("implemented them on graphs. after writing the simulation in cocalc,")
    st.text(' and exporting it to vs-code, we used a library called streamlit to write')
    st.text('the simulation in a website protocol, and we hosted it on Gil'"'"'s local')
    st.text(' network for testing and debugging. after the website was ready,we')
    st.text(' uploded it to github, added a requirments file(requirments file says')
    st.text(" what requirments does the code need to run [librarys, installations, ext...])")
    st.text('than used the hosting services porvided by streamlit to make the')
    st.text(' website online and public')
    img = Image.open('a.jpeg')
    st.image(img)

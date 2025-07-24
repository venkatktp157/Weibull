#!/usr/bin/env python
# coding: utf-8

# In[1]:

import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy.special import gamma
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import math
# import warnings
# warnings.filterwarnings('ignore')

# In[2]:
from auth import load_authenticator
from logger import setup_logger

# === Authentication ===
authenticator = load_authenticator()
name, auth_status, username = authenticator.login('Login', 'main')
logger = setup_logger()

if auth_status:
    # === Password Reset Check ===
    if authenticator.credentials["usernames"][username].get("password_reset", False):
        st.warning("🔒 You are required to reset your password.")
        if st.button("Change Password"):
            authenticator.reset_password(username)
            st.success("✅ Password updated. Please log in again.")
            st.stop()
    
    authenticator.logout('Logout', 'main')

    st.write(f"Welcome *{name}* 👋")
    logger.info(f"User {username} logged in successfully")

    #st.set_page_config(layout='wide')

    st.title('MACHINERY FAILURE ANALYSIS')
    st.subheader('WEIBULL DISTRIBUTION')

    #DATASET SECTION
    # Sidebar - Upload live dataset
    with st.sidebar.header('1. Upload your CSV data'):
        uploaded_file = st.sidebar.file_uploader("Upload your Live data CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read the uploaded file into a DataFrame
        df = pd.read_csv(uploaded_file)
        
        # Impute null values to 0
        df.fillna(0, inplace=True)
        
        # Display the filtered DataFrame
        st.markdown('**1.1. Glimpse of Live dataset**')
        st.write(df)
        
        # Sort the dataset by 'RUNNING hours' in ascending order
        df_sorted = df.sort_values(by="cycles", ascending=True).reset_index(drop=True)
        
        # Add a rank column starting from 1
        df_sorted["Rank"] = range(1, len(df_sorted) + 1)
        
        df_sorted['Median Ranks'] = (df_sorted['Rank'] - 0.3) / (len(df_sorted['Rank']) + 0.4)
        
        df_sorted['Reciprocal MR'] = 1/ (1-df_sorted['Median Ranks'])
        
        df_sorted['dual log MR'] = np.log(np.log(df_sorted['Reciprocal MR']))
        
        df_sorted['log Run hrs'] = np.log(df_sorted['cycles'])
        
        # Data for plotting
        X = df_sorted['log Run hrs']
        y = df_sorted['dual log MR']
        
        # Calculate the linear trendline
        coefficients = np.polyfit(X, y, 1)  # Linear fit (degree 1)
        linear_trendline = np.poly1d(coefficients)

        # Calculate the R² value
        y_pred = linear_trendline(X)
        ss_total = np.sum((y - np.mean(y))**2)  # Total sum of squares
        ss_residual = np.sum((y - y_pred)**2)   # Residual sum of squares
        r_squared = 1 - (ss_residual / ss_total)

        # Create the plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot the data
        ax.plot(X, y, label="Linearised Median ranks vs Running hrs.", color='g', marker='o')

        # Add the linear trendline to the plot
        ax.plot(X, y_pred, label="Linear Trendline", color='r')

        # Adding the equation of the trendline and R² value
        equation = f"y = {coefficients[0]:.2f}X + {coefficients[1]:.2f}\nR² = {r_squared:.3f}"
        ax.text(0.05, 0.85, equation, transform=ax.transAxes, fontsize=10, verticalalignment='top', 
            bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor='white'))

        # Adding title and labels
        ax.set_title("Median Ranks vs Running Hour Intervals")
        ax.set_xlabel("Running Hours")
        ax.set_ylabel("Median Ranks")

        # Adding a legend in a non-clashing position
        ax.legend(loc='lower right')

        # Displaying the grid
        ax.grid(True)

        # Show the plot in Streamlit
        st.pyplot(fig)
        
        if r_squared >= 0.95:
            st.markdown('The data satisfies a Weibull distribution')
        else:
            st.markdown('The data does not satisfy a Weibull distribution, check data quality or other distributions like Lognormal,Exponential etc.,')
            
        
        # Add a constant to the independent variable (for the intercept term)
        X = sm.add_constant(X)

        # Fit the linear regression model
        model = sm.OLS(y, X).fit()

        # Get the regression results
        intercept = model.params['const']
        coefficient = model.params['log Run hrs']
        p_value = model.pvalues['log Run hrs']
        
        # Full summary of the regression model
        st.write(model.summary())
        
        beta = coefficient  
        # Shape factor Beta (beta = 0 is constant failure rate, beta>1 is increasing failure rate #& beta<1 is decreasing failure rate)
        st.markdown(f'The Shape character "beta" is: {round(beta, 2)}')
        
        char_life = np.exp(-intercept/beta)  #Characteristic life eta
        st.markdown(f'The Characteristic Life "eta" as computed by Weibull method is: {round(char_life, 2)} hours')
        
        est_life = char_life*gamma((1/beta)+1)
        st.markdown(f'The Estimated Life as computed by Weibull method is: {round(est_life, 2)} hours')
        
        df_sorted['Reliability R(t)'] = np.exp(-((df_sorted['cycles']/char_life)**beta))
        
        df_sorted['Fail rate'] = (beta/char_life)*((df_sorted['cycles']/char_life)**(beta-1))
        
        df_sorted['Fail density'] = df_sorted['Reliability R(t)'] * df_sorted['Fail rate']
        
        #B10 Life (at 90% reliability) which states that 10% of equipment would have failed by then.

        B10_life = char_life * (-np.log(0.9))**(1/beta)
        
        st.markdown(f"B10 Life: {B10_life:.2f}")
        
        cycles_monthly = st.number_input("Enter the running cycles per month: ", min_value=1.0, step=25.0) # Running hrs per month for equipment
        
        usage = B10_life/cycles_monthly
        
        st.markdown(f'The equipment usage will last: {round(usage, 2)} months based on B10 life')

        if usage < 12.0:        
            st.markdown('Based on warranty terms, quality of spares to be checked and/or manufacturer must improve design to improve reliability')
        else:
            st.markdown('Based on current usage, equipment is able to run troublefree for at least one year')
        #------------------------------------------------------------------------- 
        
        # Create the Reliability plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plotting the Reliability Curve
        ax.plot(df_sorted['cycles'], df_sorted['Reliability R(t)'], label="Reliability Curve", color='g', marker='o')

        # Adding title and labels
        ax.set_title("Reliability vs Running Hour Intervals")
        ax.set_xlabel("Running Hours")
        ax.set_ylabel("Reliability")

        # Adding a legend
        ax.legend()

        # Displaying a grid
        ax.grid(True)

        # Show the plot in Streamlit
        st.pyplot(fig)
        
        #-------------------------------------------------------------------------------
        
        # Create the Failure rate plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plotting the Reliability Curve
        ax.plot(df_sorted['cycles'], df_sorted['Fail rate'], label="Failure Rate Curve", color='navy', marker='o')

        # Adding title and labels
        ax.set_title("Failure Rate vs Running Hour Intervals")
        ax.set_xlabel("Running Hours")
        ax.set_ylabel("Failure Rate")

        # Adding a legend
        ax.legend()

        # Displaying a grid
        ax.grid(True)

        # Show the plot in Streamlit
        st.pyplot(fig)
        
        #------------------------------------------------------------------------------------------
        
        # Create the Reliability plot
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plotting the Reliability Curve
        ax.plot(df_sorted['cycles'], df_sorted['Fail density'], label="Probability Density Failure Curve", color='maroon', marker='o')

        # Adding title and labels
        ax.set_title("Probability Density Failure vs Running Hour Intervals")
        ax.set_xlabel("Running Hours")
        ax.set_ylabel("Probability Density Failure")

        # Adding a legend
        ax.legend()

        # Displaying a grid
        ax.grid(True)

        # Show the plot in Streamlit
        st.pyplot(fig)
        
        #-------------------------------------------------------------------------------------------------
        
        MTTF = est_life
        st.markdown(f'Mean Time to Failure (MTTF) is: {round(est_life, 2)} hours')

        def interpret_weibull_stats(beta, char_life, est_life, B10_life, usage):
            inferences = []

            # Beta Interpretation
            if beta > 1:
                inferences.append("🔺 Increasing failure rate over time — typical of aging or wear-out mechanisms.")
            elif beta < 1:
                inferences.append("🔻 Decreasing failure rate — possible early-life failures or infant mortality.")
            else:
                inferences.append("➖ Constant failure rate — suggests random failures or ideal scenario.")

            # Characteristic Life & MTTF
            inferences.append(f"📌 Characteristic Life (η): {round(char_life, 2)} hrs — 63.2% of equipment fails by this point.")
            inferences.append(f"📌 Mean Time to Failure (MTTF): {round(est_life, 2)} hrs — average operational life across machines.")

            # B10 Life & Usage
            inferences.append(f"📉 B10 Life: {round(B10_life, 2)} hrs — 10% failure threshold.")
            inferences.append(f"⏳ Estimated usage before failure (at {int(cycles_monthly)} hrs/month): {round(usage, 2)} months.")

            if usage < 12.0:
                inferences.append("⚠️ Equipment does not meet a 1-year reliability benchmark. Review warranty or improve design.")
            else:
                inferences.append("✅ Equipment passes a 1-year reliability test under current usage conditions.")

            return inferences
    
            # Generate and display live Weibull inferences
            inferences = interpret_weibull_stats(beta, char_life, est_life, B10_life, usage)

            st.subheader("📊 Dynamic Inference Summary")
            for item in inferences:
                st.markdown(f"- {item}")
            
    else:
        st.info('Awaiting for csv file to be uploaded.')      
    
elif auth_status == False:
    st.error("Username/password is incorrect ❌")

elif auth_status == None:
    st.warning("Please enter your username and password 🔐")
    


# Loading required packages - 
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import streamlit as st
from PIL import Image
from statsmodels.tsa.statespace.sarimax import SARIMAX
from scipy.interpolate import interp1d

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ HOME PAGE

def home_page():
    st.write("""
             # Time Series Analysis
             
             This is our TSA app.
             """)
    
    st.write(" ")
    
    about_expander = st.beta_expander("About")
    
    about_expander.write("""
                         Time Series Analysis!  
             
                         This app serves two purposes -  
                         * **Australia Airline Traffic:** This page shows the analysis done by the author on tweets of NBA teams, to assist the marketing team. They can examine the sentiment of the tweets and the overall trends/polarity of the tweets. This can assist them in making an informed decision about where to allocate their spend.
                         * **User Area:** Users can use this area to upload their own data or a set of data and do simple text processing with the help of a simple UI.
                         
                         """)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~ ANALYSIS PAGE

# FUNCTIONS - 


def get_city_data(data, tsa_column):
    if tsa_column != "~ All Cities ~":
        return data[data['City2'] == tsa_column]
    else:
        return data

# KNN interpolation
def knn_mean(ts, n):
    out = np.copy(ts)
    for i, val in enumerate(ts):
        if np.isnan(val):
            n_by_2 = np.ceil(n/2)
            lower = np.max([0, int(i-n_by_2)])
            upper = np.min([len(ts)+1, int(i+n_by_2)])
            ts_near = np.concatenate([ts[lower:i], ts[i:upper]])
            out[i] = np.nanmean(ts_near)
    return out

def plot_boxplots(df, tsa_column, box_col1, box_col2):
    total_passengers = df.groupby([df.index]).agg({"Passenger_Trips" : "sum"})
    
    total_passengers['year'] = [d.year for d in total_passengers.index]
    total_passengers['month'] = [d.strftime('%b') for d in total_passengers.index]
    # years = total_passengers['year'].unique()
    # Plotting Year wise
    temp = total_passengers["2000":"2020"]
    fig, axes = plt.subplots(1, 1, figsize=(10,7), dpi= 80)
    sns.boxplot(x='year', y='Passenger_Trips', data=temp,
                ax=axes)
    axes.set_ylabel('Number of Passengers')
    axes.set_xlabel('Year')
    axes.set_title('Box Plot for every Year', fontsize=18)
    axes.yaxis.set_major_formatter(ticker.EngFormatter())
    plt.xticks(rotation=60)
    
    box_col1.pyplot()
    
    # Plotting Monthly
    temp = total_passengers["2016":"2019"]
    fig, axes = plt.subplots(1, 1, figsize=(10,7), dpi= 80)
    sns.boxplot(x='month', y='Passenger_Trips', data=temp)
    axes.set_title('Box Plot for every Month', fontsize=18)
    axes.set_ylabel('Number of Passengers')
    axes.set_xlabel('Month')
    axes.yaxis.set_major_formatter(ticker.EngFormatter()) 
    box_col2.pyplot()

def get_forecast(df, forecast_col2):
    
    df = df.groupby([df.index]).agg({"Passenger_Trips" : "sum"})
    df = df.dropna()
    #df['Passenger_Trips'] = knn_mean(df['Passenger_Trips'], 6)
    
    model = SARIMAX(np.log(df["Passenger_Trips"]), order = (2,1,1), seasonal_order = (0, 1, 1, 12))
    results = model.fit()
    
    fcast = np.exp(results.get_forecast('2021-09').summary_frame())
    fig, ax = plt.subplots(figsize=(15, 5))
    df.loc['2015':]['Passenger_Trips'].plot(ax=ax)
    fcast['mean'].plot(ax=ax, style='k--')
    ax.fill_between(fcast.index, fcast['mean_ci_lower'], fcast['mean_ci_upper'], color='k', alpha=0.1)
    ax.set_xlabel('Year', fontsize=12)
    ax.set_ylabel('Number of Passengers', fontsize=12)
    plt.title('Prediction of Future Data', fontsize=18)
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    forecast_col2.pyplot()
    
    fig, ax = plt.subplots(figsize=(15, 5))
    plt.plot(df[df.index.to_series().between('2019-01-01', '2019-12-01')]['Passenger_Trips'], color='green')
    plt.plot(np.exp(results.predict(start='2019-01-01', end='2019-12-01')), color='red')
    plt.title('Comparison of Actual and Predicted data', fontsize=18)
    ax.set_ylabel('Number of Passengers', fontsize=12)
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    #plt.xticks(rotation=70)
    forecast_col2.pyplot()


def airline_page():
    st.title("Airline Analysis Page")
    
    st.write("""
             EXPLAIN!  
             
             Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
             """)
    
    st.write("")
    about_expander = st.beta_expander("About")
    about_expander. write("""
                          Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
                          """)
    st.write("")
    
    # Plotting the graphs for yearly and monthly - 
    # Initial overview - 
    st.write("""
             ## Initial Overview of Airline Data 
             
             Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
             
             """)
    
    image_col1, image_col2, image_col3 = st.beta_columns((1,2,1))
    
    # ~~~~~ GRAPH
    # Setting the image - 
    image = Image.open('Data_Analysis/smooth_monthly.png')
    
    image_col2.image(image, use_column_width=True)
    
    st.write("""
             Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
             """)
    
    st.write(" ")
    
    st.write("""
             ### Now since we have an idea of the airline trend, the user can further explore below. The selectbox can be used to explore data from different cities.
             """)
    
    
    st.write(" ")
    col1, col2, col3 = st.beta_columns((1,3,1))
    
    
    city_option = ["~ All Cities ~"]
    df_cities = list(data["City2"].unique())
    col_options = city_option + df_cities
    tsa_column = col2.selectbox('Choose the city you want to explore', col_options, index = 0)
    col2.write(" ")
    
    st.write("""
             * Below we can see 2 graphs. The first one one is yearly ... And second is monthly variations in the past 5 years.
             """)
    st.write(" ")
    # Get city data - 
    temp = get_city_data(data, tsa_column)
    
    box_col1, box_col2 = st.beta_columns((1,1))
    plot_boxplots(temp, tsa_column, box_col1, box_col2)
    st.write(" ")
    st.write("""
             Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
             """)
    st.write(" ")
    # Showing the best model according to all cities - 
    forecast_col1, forecast_col2, forecast_col3 = st.beta_columns((1,1.5, 1))
    get_forecast(temp, forecast_col2)
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ USER PAGE

def get_user_data():
    st.sidebar.write('---')
    # Select Box to ask for input - 
    input_preference = st.sidebar.selectbox("Input file", ["~ Select ~", "Input from Computer", "Use Example file"])
    user_data = 2
    if input_preference == "Input from Computer":
        uploaded_file = st.sidebar.file_uploader('Upload your input CSV file', type = ['csv'])
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            user_data = 1
    elif input_preference == "Use Example file":
        df = pd.read_csv('dom_citypairs_web.csv')
        user_data = 0
    return input_preference, user_data, df

def get_tsa_column(user_data, df, col2):
    if user_data == 1:
        col_options = ['~ Select a column ~']
        df_columns = list(df.columns)
        col_options = col_options + df_columns
        tsa_column = col2.selectbox('Choose the column containing time-series data', col_options, index = 0)
    elif user_data == 0:
        tsa_column = col2.selectbox('Choose the column containing time-series data', ['Passenger_Trips', 'Aircraft_Trips', 'Seats'], index = 0)
        col2. write("""
                    ** We will be doing Time-Series Analysis on the 'Passenger_Trips' column.**
                    """)
    return tsa_column

def get_time_col_pref(user_data, col2):
    if user_data == 1:
        time_col_pref = col2.selectbox('Time data is available in - ', ['~ Select ~', 'Single column', 'Multiple columns'], index = 0)
    elif user_data == 0:
        time_col_pref = col2.selectbox('Time data is available in - ', ['Multiple columns'], index = 0)
        col2. write("""
                    ** In the input file we have time data in different columns.**
                    """)
    return time_col_pref

def get_date_time(time_col_pref, df, user_data_temp):
    if time_col_pref == 'Single column':
        col1, col2 = st.beta_columns((1,1))
        date_freq = get_date_freq(user_data_temp, df, col1)
        date_columns = get_single_date_col(df, col2)
    elif time_col_pref == 'Multiple columns':
        col1, col2, col3 = st.beta_columns((1,2,1))
        date_freq = get_date_freq(user_data_temp, df, col2)
        col1, col2 = st.beta_columns((1,1))
        date_columns = get_mutiple_cols(df, col1, col2)
    check = 1
    for col in date_columns:
        if col == '~ Select a column ~':
            check = 0
    if check == 1 and date_columns != []:    
        pass
    else:
        col1, col2, col3 = st.beta_columns((1,1,1))
        col2.info('Select the date/time columns')
    return date_freq, date_columns

def get_date_freq(user_data, df, col):
    # Getting the frequency - 
    date_freq = col.selectbox('Choose the frequency of date time - ', ['Daily','Monthly', 'Yearly'])
    return date_freq

def get_single_date_col(df, col):
    # Getting the date column - 
    col_options = ['~ Select a column ~']
    df_columns = list(df.columns)
    col_options = col_options + df_columns
    date_column = col.selectbox('Choose the column containing the date column', col_options, index = 0)
    date_column = [date_column]
    return date_column

def get_time_cols(user_time, options, col):
    time_col = col.selectbox(f'Select the {user_time} column -', options)
    time_col = [time_col]
    return time_col

def get_mutiple_cols(df, col1, col2):
    # user_times = col1.multiselect('Select the columns/information you have - ', ['Day', 'Month', 'Year'])
    col1.write("Select the date/time you have in your data -")
    day_check = col1.checkbox('Day')
    month_check = col1.checkbox('Month')
    year_check = col1.checkbox('Year')
    col_options = ['~ Select a column ~']
    df_columns = list(df.columns)
    col_options = col_options + df_columns
    selected_columns = []
    if day_check:
        selected_column = get_time_cols('Day', col_options, col2)
        selected_columns = selected_columns + selected_column
    if month_check:
        selected_column = get_time_cols('Month', col_options, col2)
        selected_columns = selected_columns + selected_column
    if year_check:
        selected_column = get_time_cols('Year', col_options, col2)
        selected_columns = selected_columns + selected_column
    return selected_columns
    

<<<<<<< HEAD
       
=======
# Plot pacf and acf plots
def plot_autocorrelation(df, n):
    plot_acf(df.diff(n).dropna(), lags=25, zero=False)
    plot_pacf(df.diff(n).dropna(),  lags=25, zero=False)
    
def adfuller(df):
    print(adfuller(df.diff(i).dropna())[1])
    
# Hyperparameter tuning
    
def hyperparameter_tuning(S, p_range=3, q_range=3, P_range=3, Q_range=3, d=1):
    order_aic_bic = []
    # Loop over p values from 0-2
    for p in range(p_range):
        # Loop over q values from 0-2
        for q in range(q_range):
            for P in range(P_range):
                for Q in range(Q_range):
                    try:
                        
                        #create and fit ARMA(p,q) model
                        model = SARIMAX(np.log(airline_pt['Passenger_Trips']), order=(p,d,q), seasonal_order=(P,d,Q,S))
                        results = model.fit()
                        
                        # Append order and results tuple
                        order_aic_bic.append((p,q,P, Q, results.aic, results.bic))
                    except:
                        order_aic_bic.append((p,q,P,Q,None, None))
      

    # Construct DataFrame from order_aic_bic
    order_df = pd.DataFrame(order_aic_bic, 
                        columns=['p', 'q', 'P', 'Q', 'AIC', 'BIC'])
    order_df = order_df.sort_values('AIC').reset_index(drop=True)


>>>>>>> 364edb50b4e80f2addcddaf137cf7ed7a52ea1bb
def user_page():
    st.title("User Basic Page")
    st.write(" ")
    st.write("""
             Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
             """)
    st.write("")
    about_expander = st.beta_expander("About")
    about_expander.write("""
                         Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
                         """)
    
    st.write(" ")
    st.write("""
             ## Let's get started!
             """)
    st.write("""
             Officiis eligendi itaque labore et dolorum mollitia officiis optio vero. Quisquam sunt adipisci omnis et ut. Nulla accusantium dolor incidunt officia tempore. Et eius omnis. Cupiditate ut dicta maxime officiis quidem quia. Sed et consectetur qui quia repellendus itaque neque. Aliquid amet quidem ut quaerat cupiditate. Ab et eum qui repellendus omnis culpa magni laudantium dolores.
             """)
    
    input_preference, user_data, input_df = get_user_data()
        
    if input_preference != "~ Select ~" and user_data != 2:
        if input_df is not None:
            
            # Dividing screen into 3 parts to 
            col1, col2, col3 = st.beta_columns((1,4,1))
            col2.write("")
            col2.write('Below we can see first 100 rows from the data.')
            
            col2.dataframe(input_df.iloc[0:100])
            
            # Ask for values column and data/time column - 
            st.write(" ")
            st.write("""
                     Great, now let's move on to the next step.  
                     Let's get the data into required format for time series analysis. In this application, we only take single column (think of word.)   
                     Select the column that selects the time-series data which you want to explore.
                     """)
            
            col1, col2, col3 = st.beta_columns((1,4,1))
            tsa_column = get_tsa_column(user_data, input_df, col2)
            
            if tsa_column != "~ Select a column ~":
                
                st.write(" ")
                st.write("""
                         Now that we have selected the time_series data column, let's select the column containing our date-time info.  
                         It's normal that sometimes, we have time information in a single column or in different columns.  
                         """)
                         
                col1, col2, col3 = st.beta_columns((1,4,1))
                time_col_pref = get_time_col_pref(user_data, col2)
                
                if time_col_pref != "~ Select ~":
                    
                    date_freq, date_columns = get_date_time(time_col_pref, input_df, user_data)
                    st.write(date_freq, date_columns)
                    
                else:
                    col2.info("Select an option to get time data")
            else:
                col2.info("Select the column for Time Series Analysis")
            
    else:
        st.info('Awaiting input file. (Use Input file select box in the sidebar to input a file)')
    
    
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ MAIN PAGE

# Setting the page layout -
st.set_page_config(layout = 'wide', page_title = "TSA App")
st.set_option('deprecation.showPyplotGlobalUse', False)


# Loading in the data - 
data = pd.read_csv("dom_citypairs_web.csv", parse_dates = [[10, 11]], infer_datetime_format = True, index_col = 0)

# Choosing the desired columns - 
data = data[["City1", "City2", "Passenger_Trips"]]
data["City2"] = data["City2"].apply(lambda x: x.title())


# Setting the image - 
image = Image.open('airline3.png')
if image.mode != "RGB":
    image = image.convert('RGB')
# Setting the image width -
st.image(image, use_column_width=True)

navigation_option = st.sidebar.selectbox("Navigation Tab", ['Home Page', 'Airline Page', 'Basic TSA'])

if navigation_option == "Home Page":
    home_page()
elif navigation_option == "Airline Page":
    airline_page()
elif navigation_option == "Basic TSA":
    user_page()
                             

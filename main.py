import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

header = st.container()
dataset = st.container()
features = st.container()
trainingModel = st.container()


# Caching the data
@st.cache
def get_data(filename):
    user_data = pd.read_csv(filename)
    return user_data


with header:
    st.title("Welcome to My Dashboard")
    st.text("Determining the features that Heavily affect user Engagement")


with dataset:
    st.header("The Telecommunication DataSet")
    st.text("I created this version of the dataset from 10_academy's week 1 project dataset")
    user_data = get_data('data/user_overview_data.csv')
    st.write(user_data.head(5))

    st.subheader('Total data usage per user in the telecommunication industry')
    data_usage_distribution = pd.DataFrame(user_data['Total_UL_and_DL_(Bytes)'].value_counts()).head(50)
    st.bar_chart(data_usage_distribution)

with features:
    st.header("Features I created")
    # Creating lists
    st.markdown('* **First Feature**: The first feature that I created for better user analytics ')
    st.markdown('* **Second Feature**: The first feature that I created for better user analytics ')

with trainingModel:
    st.header("GO Machine Learning!")
    st.text("Here you get to choose the hyperparameter of the model to see how the affect the accuracy")

    #Create columns
    sel_col, disp_col = st.columns(2)

    #Create Slider for user input
    max_depth = sel_col.slider('What should be the depth of the column?', min_value=10, max_value=100, value=20, step=10)

    #Create a dropdown menu
    n_estimators = sel_col.selectbox("How many treea should be used", options=[100,200,300, "No limits"], index=0)
    # Adding the list of features one can select and work with
    sel_col.text_input('The following is a list of features you can work with')
    sel_col.write(user_data.columns)

    #Text Input
    input_feature = sel_col.text_input('Which feature would you like to work with', 'Total_UL_and_DL_(Bytes)')

    # Adding our models
    if n_estimators == 'No limits':
        regr = RandomForestRegressor(max_depth=max_depth)
    else:
        regr = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)

    x = user_data[[input_feature]]
    y = user_data[['Total_UL_and_DL_(Bytes)']]
    regr.fit(x,y)
    prediction = regr.predict(y)

    disp_col.subheader("The Mean Absolute Error is:")
    disp_col.write(mean_absolute_error(y, prediction))

    disp_col.subheader("The Mean Squared Error is:")
    disp_col.write(mean_squared_error(y, prediction))

    disp_col.subheader("The R squared Score of the model is::")
    disp_col.write(r2_score(y, prediction))



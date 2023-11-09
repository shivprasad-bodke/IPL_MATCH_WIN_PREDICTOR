import altair as alt
import plotly.express as px
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pickle
import seaborn as sns
import pandas as pd
import datetime
from IPython.display import display
import statistics as stat


# icon and title
st.set_page_config(page_title=" IPL MATCH Win Predictor", page_icon=":bar_chart:",initial_sidebar_state="expanded")



# Add some CSS styles to the title
st.markdown(
    f"""
    <style>
        h1 {{
            color: #0072B2;
            text-align: center;
        }}
    </style>
    """,
    unsafe_allow_html=True
)




teams = ['Sunrisers Hyderabad',
         'Mumbai Indians',
         'Royal Challengers Bangalore',
         'Kolkata Knight Riders',
         'Kings XI Punjab',
         'Chennai Super Kings',
         'Rajasthan Royals',
         'Delhi Capitals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
          'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
          'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
          'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
          'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
          'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open("pipe.pkl", 'rb'))


# TITLE OF PAGE
st.sidebar.markdown('<h1>Match Win Predictor</h1>', unsafe_allow_html=True)


col1, col2 = st.columns(2)

with col1:
    batting_team = st.sidebar.selectbox('Select the batting team', sorted(teams))
with col2:
    bowling_team = st.sidebar.selectbox('Select the bowling team', (teams))

selected_city = st.sidebar.selectbox('Select host city', sorted(cities))

target = st.sidebar.number_input('Target')

col3, col4, col5 = st.columns(3)

with col3:
    score = st.sidebar.number_input('Score')
with col4:
    overs = st.sidebar.number_input('Overs completed')
with col5:
    wickets = st.sidebar.number_input('Wickets out')





# DEFAULT
df = pd.read_csv("temp.csv")



st.header("IPL Match Prediction Dashboard")

# POPULARITY TEAMS IN 2023
st.header("IPL TEAM POPULARITY : ")
# Define the popularity data for each IPL team in 2023
popularity = {'Chennai Super Kings': 20, 'Mumbai Indians': 25, 'Royal Challengers Bangalore': 15, 
              'Kolkata Knight Riders': 10, 'Sunrisers Hyderabad': 8, 'Rajasthan Royals': 7, 
              'Delhi Capitals': 5, 'Punjab Kings': 5}

# Create a pie chart using Matplotlib
fig, ax = plt.subplots()
ax.pie(popularity.values(), labels=popularity.keys(), autopct='%1.1f%%')

# Set the title for the graph
ax.set_title('IPL Team Popularity')

# Display the graph in Streamlit
st.pyplot(fig)




st.header("Wins by each team")
operator_counts = df['winner'].value_counts().reset_index().head(10)
operator_counts.columns = ['Operator', 'count']
chart = alt.Chart(operator_counts).mark_bar().encode(
    x=alt.X('count', title='Count'),
    y=alt.Y('Operator', title='TEAMS')
).properties(

)
st.altair_chart(chart, use_container_width=True)

#  LINE CHART
st.header("Number of Runs_left by MatchId : ")
Fatalities = df.groupby('match_id')['runs_left'].count().head(100)
st.line_chart(Fatalities)

# LINE CHART
st.header("Extra Runs in a Match")
fig = plt.figure(figsize=(10,5))
plt.plot(df['extra_runs'].unique())
plt.xlabel("Overs")
plt.ylabel("Extra_Runs")
plt.legend("Extra_Runs")
plt.show()
st.pyplot(fig)


#  PROBABILITY SHOWING
if st.sidebar.button('Predict Probability'):
    runs_left = target - score
    balls_left = 120 - (overs*6)
    wickets = 10 - wickets
    crr = score/overs
    rrr = (runs_left*6)/balls_left

    input_df = pd.DataFrame({'batting_team': [batting_team], 'bowling_team': [bowling_team], 'city': [selected_city], 'runs_left': [runs_left], 'balls_left': [balls_left], 'wickets': [wickets], 'total_runs_x': [target], 'crr': [crr], 'rrr': [rrr]})

    result = pipe.predict_proba(input_df)
    loss = result[0][0]
    win = result[0][1]
    st.sidebar.header(batting_team + "- " + str(round(win*100)) + "%")
    st.sidebar.header(bowling_team + "- " + str(round(loss*100)) + "%")


    # CHARTS SHOWING PIE
    data = pd.DataFrame({
        'Winning': [batting_team, bowling_team],'Percentage': [round(win*100), round(loss*100)]
        })

    # Create pie chart
    st.header("Pie Chart with Percentage Labels : ")
    fig = px.pie(data, values='Percentage', names='Winning',hole=0.5, color_discrete_sequence=px.colors.qualitative.Set3)

    # Render chart
    st.plotly_chart(fig, use_container_width=True)

    

    # Create a sample dataframe
    data = {
        'Winning': [batting_team, bowling_team],
        'Percentage': [round(win*100), round(loss*100)]
    }
    df = pd.DataFrame(data)

    # Set up the bar chart using Altair
    bars = alt.Chart(df).mark_bar().encode(
        x='Winning',
        y='Percentage'
    )

    # Set the chart's title and axis labels
    chart = bars.properties(
        
        width=alt.Step(80)
    )

    
    # Display the chart in Streamlit
    st.header("Sample Bar Chart :")
    st.altair_chart(chart, use_container_width=True)

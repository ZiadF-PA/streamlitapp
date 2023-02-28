
# import libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
from PIL import Image
import datetime as dt
import seaborn as sns
from typing import List, Dict
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import xgboost as xg
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as MSE
import pickle as pickle
from sklearn import preprocessing


# Run-chart
def plot_completion_forecast(df):
    total_time_forecast = df["Planned Duration"].sum()
    df["cumulative_forecast_time"] = df["Planned Duration"].cumsum()
    df["cumulative_completion_percentage"] = df["cumulative_forecast_time"]/total_time_forecast
    df["completion_uncertainty"] = df["Planned Duration"]*0.05
    
    df["delay_duration"] = (df["Actual Start Date"] - df["Planned start date"])
    df["delay_duration"] = df["delay_duration"].fillna("0 days")
    df["cumulative_delay"]=pd.Timedelta("0 days")
    df["cumulative_delay"] = df["delay_duration"].cumsum()

    df["updated_forecast"] = df["Planned start date"] + df["cumulative_delay"]

    fig = go.Figure([
        go.Scatter(
            name='Planned',
            x=df['Planned start date'],
            y=df['cumulative_completion_percentage'],
            mode='lines',
            line=dict(color='rgb(31, 119, 180)', width=6),
        ),
        go.Scatter(
            name='Upper Bound',
            x=df['Planned start date'],
            y=df['cumulative_completion_percentage']+df['completion_uncertainty'],
            mode='lines',
            marker=dict(color="#444"),
            line=dict(width=0),
            showlegend=False
        ),
        go.Scatter(
            name='Lower Bound',
            x=df['Planned start date'],
            y=df['cumulative_completion_percentage']-df['completion_uncertainty'],
            marker=dict(color="#444"),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(80, 60, 150, 0.4)',
            fill='tonexty',
            showlegend=False
            ),
        go.Scatter(
            name='Updated Forecast',
            x=df['updated_forecast'],
            y=df['cumulative_completion_percentage'],
            mode='lines',
            line=dict(color='rgb(0, 200, 80)',  width=4),
        )
    ])
    fig.update_xaxes(title="Date")
    fig.update_yaxes(title="Woring Hours")
    # fig.show()
    return fig

def estimate_required_time(historic_data_df, xgb_model_loaded, task_request, res_request):

    # Prepare data
    X, y = historic_data_df.iloc[:, :-1], historic_data_df.iloc[:, -1]
    encoder_task = preprocessing.LabelEncoder()
    encoder_task.fit(X.loc[:,"Task"])

    encoder_res = preprocessing.LabelEncoder()
    encoder_res.fit(X.loc[:,"Resource"])
    
    task_request = encoder_task.transform([task_request])
    res_request = encoder_res.transform([res_request])

    inarray = np.array([task_request, res_request]).reshape(1,-1)
    # Perform prediction
    return round(float(xgb_model_loaded.predict(inarray)[0]), 2)


# Gantt-chart
def preprocess_data(df):
    df = df[['Task ID','Assigned to','Planned start date',
              'Planned Duration', 'Planned Finish Date']]
    df['start'] = pd.to_datetime(df['Planned start date'],format = "%d/%m/%Y")
    df['end'] = pd.to_datetime(df['Planned Finish Date'],format = "%d/%m/%Y")

    df['Completion'] =   np.random.randint(0,100,len(df))
    df = df.rename(columns={"Planned Duration": "duration"})

    df=df.sort_values(by='Task ID', ascending=True)
    # df=df.sort_values(by='start', ascending=True)
    # df = df.reset_index(drop=True)
    return df

def assign_colors(elements: List[str]) -> Dict[str, str]:
    colors = sns.color_palette(n_colors=len(elements))
    color_map = {}
    for i, element in enumerate(elements):
        color_map[element] = colors[i]
    return color_map

def gantt_plot(df):
    c_dict = assign_colors(list(df['Assigned to'].unique()))
    # unique_resource = list(df['Assigned to'].unique())
    
    # c_dict={'Resource 1':'red', 'Resource 2':'green',
    #         'Resource 3':'blue', 'Resource 4':'yellow'}
    #project level variables
    p_start=df.start.min()
    p_end=df.end.max()
    p_duration=(p_end-p_start).days+1
    #Add relative date
    df['rel_start']=df.start.apply(lambda x: (x-p_start).days)
    df['w_comp']=round(df.Completion*df.duration/100,2)
    #Create custom x-ticks and x-tick labels
    x_ticks=[i for i in range(p_duration+1)]
    x_labels=[(p_start+dt.timedelta(days=i)).strftime('%d-%b') 
              for i in x_ticks]
    yticks=[i for i in range(len(df['Task ID']))]
    plt.figure(figsize=(12,7))
    plt.title('Gantt Chart:Project Recruitment', size=18)
    for i in range(df.shape[0]):
        color=c_dict[df['Assigned to'][i]]
        plt.barh(y=yticks[i], left=df.rel_start[i], 
                 width=df.duration[i], alpha=0.4, 
                 color=color)
        plt.barh(y=yticks[i], left=df.rel_start[i], 
                 width=df.w_comp[i], alpha=1, color=color,
                label=df['Assigned to'][i])
        plt.text(x=df.rel_start[i]+df.w_comp[i],
                 y=yticks[i],
                 s=f'{df.Completion[i]}%')
        
    plt.gca().invert_yaxis()
    plt.xticks(ticks=x_ticks[::3], labels=x_labels[::3])
    plt.yticks(ticks=yticks, labels=df['Task ID'])
    plt.grid(axis='x')
    plt.xlabel('Date',fontsize=16)
    plt.ylabel('Task ID',fontsize=16)
    #fix legends
    handles, labels = plt.gca().get_legend_handles_labels()
    handle_list, label_list = [], []
    for handle, label in zip(handles, labels):
        if label not in label_list:
            handle_list.append(handle)
            label_list.append(label)
    plt.legend(handle_list, label_list, fontsize='medium', 
               title='Assigned to', title_fontsize='large')
    # plt.show()
    # return plt
    plt.savefig("./../plots/gantt_chart.png")


# Page about - Application Description
def page1():
    # with st.expander(" ", expanded=True):    
    st.title("Ingenuity Progress Pathway Planner")
    # Display image
    image = Image.open('./../images/01_01_start_page.png')
    st.image(image, caption='')


# Page 2 - Plots
def page2(rc_df, gc_df):
    # Create run chart using Plotly
    # Sidebar filters
    with st.expander("Data with forecast", expanded=True):    
        st.table(rc_df)


# Page 2 - Plots
def page3(rc_df, gc_df, historic_data_df, xgb_model_loaded):
    # Create run chart using Plotly
    # Sidebar filters
    st.sidebar.markdown(" ")
    st.sidebar.markdown(" ")
    st.sidebar.markdown(" ")
    st.sidebar.markdown("---")


    # Create and display run chart
    st.title('Ingenuity Progress Pathway Planner')

    # workstream filter
    workstream_list = list(set(rc_df["Request"].tolist()))
    workstream_list.sort(reverse=False)
    # workstream_sel = st.sidebar.multiselect(label='Select Workstream', options=workstream_list, default=workstream_list)
    workstream_sel = st.sidebar.selectbox(label='Select Workstream', options=workstream_list)
    fig_run = plot_completion_forecast(rc_df[rc_df["Request"]==workstream_sel])
      
    rc_sub_df = rc_df[rc_df["Request"]==workstream_sel]
    # rc_sub_df = rc_df[rc_df["Request"].isin(workstream_sel)]

    # resource filter
    resource_list = list(set(rc_sub_df["Assigned to"].tolist()))
    # resource_list = list(set(rc_df["Assigned to"].tolist()))
    resource_list.sort(reverse=False)
    # resource_sel = st.sidebar.multiselect(label='Select Resource', options=resource_list, default=resource_list)    
    resource_sel = st.sidebar.selectbox(label='Select Resource', options=resource_list)    

    print(f" --------- workstream_sel: {workstream_sel}") # TODO debug 

    with st.expander("Project Progress and Forecasting", expanded=True):    
        st.plotly_chart(fig_run,use_container_width=True)
        rc_col1, rc_col2, rc_col3, rc_col4, rc_col5 = st.columns(5)
        rc_col1.metric('Mean', rc_sub_df['Planned Duration'].mean().round(2), "")
        rc_col2.metric("Standard Deviation", rc_sub_df['Planned Duration'].std().round(2), "")
        rc_col3.metric("ML forecast time", estimate_required_time(historic_data_df, xgb_model_loaded, workstream_sel, resource_sel), "")

    # Create and display Gantt chart
    fig_gantt = px.timeline(gc_df, x_start='start_time', x_end='end_time', y='task_name', color='task_name')
    with st.expander("Task Planning Gant-Chart", expanded=True):    
        # st.plotly_chart(fig_gantt, use_container_width=True)
        dfnew = preprocess_data(rc_sub_df)
        # dfnew = preprocess_data(rc_df)
        # filter for resources
        dfnew = dfnew[dfnew['Assigned to']==resource_sel]
        dfnew.reset_index(inplace=True)
        # dfnew = dfnew[dfnew['Assigned to'].isin(resource_sel)]
        # display the gantt-chart
        gantt_plot(dfnew)
        image = Image.open('./../plots/gantt_chart.png')
        st.image(image, caption='')


# App
def app():
    # Load data
    rc_df = pd.read_excel("./../data/Example_Progress_Tracking_Data_02.xlsx",skiprows=1)
    # rc_df = pd.read_csv('./../data/Work Task Tracking Tool.csv')
    gc_df = pd.read_csv('./../data/app_4_example_data_uk_date_time.csv')
    historic_data_df = pd.read_csv("./../data/Historic_Data.csv")

    #load model
    xgb_model_loaded = pickle.load(open("./../models/xgb_trained_model.pkl", "rb"))

    # Define the date and time format string
    rc_dt_format = '%m/%d/%Y' # %H:%M:%S'
    gc_dt_format = '%m/%d/%Y %H:%M' #:%S'
    # Convert date column to datetime format
    rc_df['Planned start date'] = pd.to_datetime(rc_df['Planned start date'], format=rc_dt_format)
    rc_df['Planned Finish Date'] = pd.to_datetime(rc_df['Planned Finish Date'], format=rc_dt_format)
    rc_df['Actual Start Date'] = pd.to_datetime(rc_df['Actual Start Date'], format=rc_dt_format)
    rc_df['Actual Completion Date'] = pd.to_datetime(rc_df['Actual Completion Date'], format=rc_dt_format)
    rc_df['Forecast Completion Date'] = pd.to_datetime(rc_df['Forecast Completion Date'], format=rc_dt_format)

    gc_df['start_time'] = pd.to_datetime(gc_df['start_time'], format=gc_dt_format)
    gc_df['end_time'] = pd.to_datetime(gc_df['end_time'], format=gc_dt_format)

    # Perform calculation

    # Set
    st.set_page_config(page_title="Ingenuity Progress Pathway Planner", layout="wide")
    
    # Define pages
    pages = {
        "Description": page1,
        "Table": page2,
        "Plots": page3,
    }
    
    # Sidebar navigation
    page = st.sidebar.selectbox("Choose page", options=list(pages.keys()))
    
    # Display selected page with inputs_df as argument for Page 2
    if page == "Table":
       page2(rc_df, gc_df)
    elif page == "Plots":
       page3(rc_df, gc_df, historic_data_df, xgb_model_loaded)
    else:
        pages[page]()
    
# Run app
if __name__ == "__main__":
    app()



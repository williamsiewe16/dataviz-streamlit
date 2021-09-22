import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

uber_path = "dataset/uber-raw-data-apr14.csv"
ny_path = "dataset/ny-trips-data.csv"

def myDecorator(function):
    def modified_function(df):
        time_ = time.time()
        res = function(df)
        time_ = time.time()-time_
       # with open(f"{function.__name__}_exec_time.txt","w") as f:
       #     f.write(f"{time_}")
        return res
    return modified_function


@st.cache
def load_data(path):
    df = pd.read_csv(path)
    return df


@myDecorator
@st.cache
def df1_data_transformation(df_):
    df = df_.copy()
    df["Date/Time"] = df["Date/Time"].map(pd.to_datetime)

    def get_dom(dt):
        return dt.day
    def get_weekday(dt):
        return dt.weekday()
    def get_hours(dt):
        return dt.hour

    df["weekday"] = df["Date/Time"].map(get_weekday)
    df["dom"] = df["Date/Time"].map(get_dom)
    df["hours"] = df["Date/Time"].map(get_hours)

    return df


@myDecorator
@st.cache
def df2_data_transformation(df_):
    df = df_.copy()
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])
    df["tpep_dropoff_datetime"] = pd.to_datetime(df["tpep_dropoff_datetime"])

    def get_hours(dt):
        return dt.hour

    df["hours_pickup"] = df["tpep_pickup_datetime"].map(get_hours)
    df["hours_dropoff"] = df["tpep_dropoff_datetime"].map(get_hours)

    return df

@st.cache(allow_output_mutation=True)
def frequency_by_dom(df):
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set_title("Frequency by DoM - Uber - April 2014")
    ax.set_xlabel("Date of the month")
    ax.set_ylabel("Frequency")
    ax = plt.hist(x=df.dom, bins=30, rwidth=0.8, range=(0.5,30.5))
    return fig

@st.cache
def map_data(df):
    df_ = df[["Lat","Lon"]]
    df_.columns=["lat","lon"]
    return df_

@st.cache(allow_output_mutation=True)
def data_by(by,df):
    def count_rows(rows):
        return len(rows)
    
    if by == "dom":
        fig, ax = plt.subplots(1,2, figsize=(10,6))
        ax[0].set_ylim(40.72,40.75)
        ax[0].bar(x=sorted(set(df["dom"])),height=df[["dom","Lat"]].groupby("dom").mean().values.flatten())
        ax[0].set_title("Latitude moyenne par jour du mois")

        ax[1].set_ylim(-73.96,-73.98)
        ax[1].bar(x=sorted(set(df["dom"])),height=df[["dom","Lon"]].groupby("dom").mean().values.flatten(), color="orange")
        ax[1].set_title("Longitude moyenne par jour du mois")
        return fig
    
    elif by == "hours":
        fig, ax= plt.subplots(figsize=(10,6))
        ax = plt.hist(x=df.hours, bins=24, range=(0.5,24))
        return fig
    
    elif by == "dow":
        fig, ax= plt.subplots(figsize=(10,6))
        ax = plt.hist(x=df.weekday, bins=7, range=(-5,6.5))
        return fig
    
    elif by == "dow_xticks":
        fig, ax= plt.subplots(figsize=(10,6))
        ax.set_xticklabels('Mon Tue Wed Thu Fri Sat Sun'.split())
        ax.set_xticks(np.arange(7))
        ax = plt.hist(x=df.weekday, bins=7, range=(0,6))
        return fig
    
    else:
        pass

@st.cache
def group_by_wd(df):    
    def count_rows(rows):
        return len(rows)
    grp_df = df.groupby(["weekday","hours"]).apply(count_rows).unstack()
    return grp_df

@st.cache(allow_output_mutation=True)
def grp_heatmap(grp_df):
    fig, ax= plt.subplots(figsize=(10,6))
    ax = sns.heatmap(grp_df)
    return fig

@st.cache(allow_output_mutation=True)
def lat_lon_hist(df,fusion=False):
    lat_range = (40.5,41)
    lon_range = (-74.2,-73.6)

    if fusion:
        fig, ax = plt.subplots()
        ax1 = ax.twiny()
        ax.hist(df.Lon, range=lon_range, color="yellow")
        ax.set_xlabel("Latitude")
        ax.set_ylabel("Frequency")

        ax1.hist(df.Lat, range=lat_range)
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Frequency")
        return fig
    
    else:
        fig, ax = plt.subplots(1,2, figsize=(10,5))


        ax[0].hist(df.Lat, range=lat_range, color="red")
        ax[0].set_xlabel("Latitude")
        ax[0].set_ylabel("Frequence")

        ax[1].hist(df.Lon, range=lon_range, color="green")
        ax[1].set_xlabel("Longitude")
        ax[1].set_ylabel("Frequence")
        return fig

@st.cache(allow_output_mutation=True)
def display_points(data, color=None):
    fig, ax= plt.subplots(figsize=(10,6))
    ax = sns.scatterplot(data=data) if color == None else sns.scatterplot(data=data, color=color)
    return fig

@st.cache(allow_output_mutation=True)
def passengers_graphs_per_hour(df):
    fig, ax = plt.subplots(2,2, figsize=(10,6))

    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks(np.arange(24))

    ax[0,0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","passenger_count"]].groupby("hours_pickup").sum().values.flatten(), color="red")
    ax[0,0].set_title("Total Number of passengers per pickup hour")

    ax[0,1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","passenger_count"]].groupby("hours_pickup").mean().values.flatten(), color="yellow")
    ax[0,1].set_title("Average Number of passengers per pickup hour")

    ax[1,0].bar(x=sorted(set(df["hours_pickup"])), height=df["hours_pickup"].value_counts().sort_index().values.flatten(), color="green")
    ax[1,0].set_title("Total number of passages per pickup hour")
    return fig

@st.cache(allow_output_mutation=True)
def passengers_graphs_per_dropoff_hour(df):
    fig, ax = plt.subplots(2,2, figsize=(12,6))

    for ax_ in ax:
        for ax__ in ax_:
            ax__.set_xticks(np.arange(24))

    ax[0,0].bar(x=sorted(set(df["hours_dropoff"])), height=df[["hours_dropoff","passenger_count"]].groupby("hours_dropoff").sum().values.flatten())
    ax[0,0].set_title("Total Number of passengers per dropoff hour")

    ax[0,1].bar(x=sorted(set(df["hours_dropoff"])), height=df[["hours_dropoff","passenger_count"]].groupby("hours_dropoff").mean().values.flatten(), color="black")
    ax[0,1].set_title("Average Number of passengers per dropoff hour")

    ax[1,0].bar(x=sorted(set(df["hours_dropoff"])), height=df["hours_dropoff"].value_counts().sort_index().values.flatten(), color="orange")
    ax[1,0].set_title("Total number of passages per dropoff hour")
    return fig

@st.cache(allow_output_mutation=True)
def amount_graphs_per_hour(df):
    fig, ax = plt.subplots(1,2, figsize=(12,6))

    for ax_ in ax:
        ax_.set_xticks(np.arange(24))

    ax[0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","trip_distance"]].groupby("hours_pickup").sum().values.flatten(), color="grey")
    ax[0].set_title("Total trip distance per pickup hour")

    ax[1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","trip_distance"]].groupby("hours_pickup").mean().values.flatten())
    ax[1].set_title("Average trip distance per pickup hour")
    return fig

@st.cache(allow_output_mutation=True)
def distance_graphs_per_hour(df):
    fig, ax = plt.subplots(1,2, figsize=(10,6))

    for ax_ in ax:
        ax_.set_xticks(np.arange(24))

    ax[0].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","total_amount"]].groupby("hours_pickup").sum().values.flatten(), color="lime")
    ax[0].set_title("Total amount per hour")

    ax[1].bar(x=sorted(set(df["hours_pickup"])), height=df[["hours_pickup","total_amount"]].groupby("hours_pickup").mean().values.flatten(), color="pink")
    ax[1].set_title("Average amount per hour")
    return fig

@st.cache(allow_output_mutation=True)
def corr_heatmap(df):
    fig, ax = plt.subplots(figsize=(10,6))
    ax = sns.heatmap(df.corr())
    return fig


def Uber_dataset():
    #Uber-raw-data-apr14 dataset
    st.title("Uber-raw-data-apr14 dataset")

    ## Load the Data
    st.text(" ")
    st.header("Load the Data")
    df1 = load_data(uber_path)
    if st.checkbox('Show dataframe'):
        df1


    ## Perform Data Transformation
    df1_ = df1_data_transformation(df1)

    ## Visual representation

    #
    st.text(" ")
    st.text(" ")
    st.header("Visual representation")
    if st.checkbox("Show graphs"):
        st.text(" ")
        st.markdown("`Fréquence par jour du mois`")
        st.pyplot(frequency_by_dom(df1_))

        #
        st.text(" ")
        st.markdown("`Visualisation des points sur une carte`")
        st.map(map_data(df1_))

        #
        st.text(" ")
        st.markdown("`Latitude et longitude moyenne par jour du mois`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(data_by("dom",df1_))


        #
        st.text(" ")
        st.markdown("`Visualisation des données par heure`")
        st.pyplot(data_by("hours",df1_))

        #
        st.text(" ")
        st.markdown("`Visualisation des données par jour de la semaine`")
        st.pyplot(data_by("dow",df1_))

        #
        st.text(" ")
        st.markdown("`Visualisation des données par jour de la semaine avec les noms des jours en abcisse`")
        st.pyplot(data_by("dow_xticks",df1_))


    st.text(" ")
    st.text(" ")
    ## Performing Cross Analysis
    st.header("Performing Cross Analysis")

    if st.checkbox('Show cross analysis'):

        #
        grp_df = group_by_wd(df1_)

        #
        st.text(" ")
        st.markdown("`Carte de chaleur avec les données groupées`")
        st.pyplot(grp_heatmap(grp_df))

        #
        st.text(" ")
        st.markdown("`Histogramme de la latitude et de la longitude`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(lat_lon_hist(df1_))

        #
        st.text(" ")
        st.markdown("`Fusions des histogrammes de latitude et de longitude`")
        st.pyplot(lat_lon_hist(df1_, fusion=True))

        #
        st.text(" ")
        st.markdown("`affichage de la latitude des points sur un graphique`")
        st.pyplot(display_points(df1_.Lat))

        #
        st.text(" ")
        st.markdown("`affichage de la longitude des points sur un graphique`")
        st.pyplot(display_points(df1_.Lon, color="orange"))


def Ny_dataset():
    #ny-trips-data dataset
    st.text(" ")
    st.text(" ")
    st.title("Ny-trips-data dataset")

    ## Load the Data
    st.text(" ")
    st.text(" ")
    st.header("Load the Data")
    st.text(" ")
    df2 = load_data(ny_path)
    if st.checkbox('Show dataframe 2'):
        df2


    ## Perform Data Transformation
    st.text(" ")
    df2_ = df2_data_transformation(df2)


    ## Visual representation

    #
    st.text(" ")
    st.text(" ")
    st.header("Visual representation")
    if st.checkbox('Show graphs 2'):
        st.text(" ")
        st.text(" ")
        st.markdown("`Nombre total, moyen de passagers et nombre total de passages par heure de départ`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(passengers_graphs_per_hour(df2_))

        #
        st.text(" ")
        st.text(" ")
        st.markdown("`Nombre total, moyen de passagers et nombre total de passages par heure de d'arrivée`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(passengers_graphs_per_dropoff_hour(df2_))

        #
        st.text(" ")
        st.text(" ")
        st.markdown("`Montant total et montant moyen perçu en fonction de l'heure de départ`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(amount_graphs_per_hour(df2_))

        #
        st.text(" ")
        st.text(" ")
        st.markdown("`Distance totale parcourue et distance moyenne en fonction de l'heure de départ`")
        plt.gcf().subplots_adjust(wspace = 0.3, hspace = 0.5)
        st.pyplot(distance_graphs_per_hour(df2_))




    ## Performing Cross Analysis

    #
    st.text(" ")
    st.text(" ")
    st.header("Performing Cross Analysis")
    if st.checkbox('Show cross analysis 2'):
        st.text(" ")
        st.markdown("`Carte de chaleur permettant de visualiser la corrélation entre les différentes features`")
        st.pyplot(corr_heatmap(df2_.corr()))

        #
        st.text(" ")
        st.markdown("`Carte de chaleur montrant la corrélation entre le nombre de passagers, la ditance totale, le montant du tarif, du pourboire et le montant total groupés par heure de départ`")
        grp = df2_[["passenger_count", "hours_pickup", "trip_distance", "fare_amount", "tip_amount", "total_amount"]].groupby("hours_pickup").sum()
        st.pyplot(corr_heatmap(grp.corr()))


def main():

    choice = st.sidebar.selectbox(
    'Choisissez votre dataset',
    ('Uber-raw-data-apr14 dataset', 'Ny-trips-data dataset'))

    if choice == 'Uber-raw-data-apr14 dataset':
        Uber_dataset()
    else:
        Ny_dataset()
    



    

main()

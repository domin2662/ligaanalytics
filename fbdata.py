
import streamlit as st
import pandas as pd
import seaborn as sns

from pandas import json_normalize

import matplotlib.pyplot as plt
from matplotlib.patches import Arc
import numpy as np

import re, json, requests


#st.beta_set_page_config(layout="wide")
x = st.sidebar.header('⚽ Analytics')  # 👈 this is a widget

####
file_to_charge2 = st.sidebar.selectbox('temporada',('Temporada 2019-2020','Temporada 2018-2019','Temporada 2017-2018','Temporada 2016-2017'))

if file_to_charge2 == 'Temporada 2018-2019':
    file_to_charge = '4.json'
elif file_to_charge2 == 'Temporada 2019-2020':
    file_to_charge = '42.json'
elif file_to_charge2 == 'Temporada 2017-2018':
    file_to_charge = '1.json'
elif file_to_charge2== 'Temporada 2016-2017':
    file_to_charge = '2.json'

'''

'''
### busqueda de partidos ###

home_team = st.sidebar.selectbox('Equipo juega en casa:',( 'Barcelona','Atlético Madrid','Athletic Bilbao', 'Celta Vigo', 'Deportivo Alavés', 'Deportivo La Coruna', 'Eibar', 'Espanyol', 'Getafe', 'Girona', 'Granada', 'Las Palmas', 'Leganés', 'Levante', 'Málaga', 'Osasuna', 'Rayo Vallecano', 'Real Betis', 'Real Madrid', 'Real Sociedad', 'Real Valladolid', 'Sevilla', 'Sporting Gijón', 'Valencia', 'Villarreal'))
away_team = st.sidebar.selectbox('Equipo fuera de casa:',('Athletic Bilbao', 'Atlético Madrid', 'Barcelona', 'Celta Vigo', 'Deportivo Alavés', 'Deportivo La Coruna', 'Eibar', 'Espanyol', 'Getafe', 'Girona', 'Granada', 'Las Palmas', 'Leganés', 'Levante', 'Málaga', 'Osasuna', 'Rayo Vallecano', 'Real Betis', 'Real Madrid', 'Real Sociedad', 'Real Valladolid', 'Sevilla', 'Sporting Gijón', 'Valencia', 'Villarreal'))
#json_normalize(my_data3, sep='_').assign(match_id=file_name[:-5])


#### ESCUDO DEL EQUIPO - EJEMPLO



###strestre
url = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/matches/11/' + file_to_charge
resp = requests.get(url)
st.title('⚽ DATOS POR TEMPORADA ⚽:')
dfpartidos = json.loads(resp.text)

#dfpartidos = json.load(open(os.path.expanduser('~/Desktop/DATOS/open-data-master/data/matches/11/' + file_to_charge), 'r', encoding='utf-8'))




#DATAFRAME DE LA TEMPORADA
FIELDS = ['match_id','match_week',"home_team.home_team_name",'away_team.away_team_name','home_score','away_score','referee.name']
dfdef = pd.json_normalize(dfpartidos)
p = dfdef[FIELDS]

#p = p.sort_values(by=['match_week']


p = p.set_index('match_week')

p = p.sort_values(by=['match_week'])


#DATAFRAME SEGÚN EQUIPOS FILTRADOS

l = p.loc[p['home_team.home_team_name'] == home_team, :]
l = l.loc[l['away_team.away_team_name'] == away_team, :]




#TABLE WITH DATA OF THE SELECTED SEASON:
st.subheader('📋 Datos de la  {} :'.format(file_to_charge2))
st.write(p)




### GANADOS Y PERDIDOS

st.subheader('Arbitros pare el {} en {}'.format(home_team,file_to_charge2))

f = p.loc[p['home_team.home_team_name'] == home_team, :]

m = f.groupby(['match_week'])

p['count']=1
referee_graph = p.groupby(['referee.name']).count()['count'].sort_values(ascending=True)

col1,col2 =st.beta_columns(2)

col1.dataframe(referee_graph)

## - GRÁFICO DE LOS ARBITROS - ##

referee_graph.plot.bar()

st.set_option('deprecation.showPyplotGlobalUse', False)
#st.pyplot()



match_idfil = l['match_id']

ndpart = int(l['match_id'])


#### DATA FRAME
file_name = str(ndpart) + '.json'
#my_data3 = json.load(open(os.path.expanduser('~/Desktop/DATOS/open-data-master/data/events/' + file_name), 'r', encoding='utf-8'))
#dfpt = json_normalize(my_data3, sep='_').assign(match_id=file_name[:-5])
playername = st.sidebar.selectbox('SELECTOR PLAYER', ('Lionel Andrés Messi Cuccittini','Philippe Coutinho Correia', 'Sergi Roberto Carnicer', 'Nélson Cabral Semedo','Antoine Griezmann','Ivan Rakitić','Anssumane Fati','Ricard Puig Martí'))

#pass_df = dfpt.loc[dfpt['type_name'] == 'Pass', :].copy()
#pass_df.dropna(inplace=True, axis=1)
#pass_df = pass_df.loc[pass_df['player_name'] == playername, :]

#dff = pd.DataFrame.from_dict(pass_df, orient='columns')
#st.write('Pases de', playername.upper())
#st.write(dff)


## BAR CHART##

# Aqui hacemo el filtro al barcelona
st.header('Gráficos por equipos:')

my_exp = st.beta_expander('Visualización')
with my_exp:

    mfc = p.loc[p['home_team.home_team_name'] == home_team, :]



    ### TITLE

    ### BAR CHART ###


    st.write(mfc)
    st.subheader('Gráficos {} en casa'.format(home_team))
    allcolumns = mfc.columns.tolist()
    typeofplot= 'bar'
    select_column_names  = ['home_score','away_score']
    select_column_names2  = ['away_score']




    if typeofplot == 'bar':
        custdat =mfc[select_column_names]
        st.bar_chart(custdat)


    elif typeofplot:
        custdat =mfc[select_column_names].plot(kind=type)
        st.write(custdat)
        st.pyplot()


    # UNIQUE BAR CHART
    mec = p.loc[p['away_team.away_team_name'] == away_team, :]


    st.subheader('Gráficos {} fuera de casa'.format(away_team))
    allcolumns = mec.columns.tolist()
    typeofplot= 'bar'
    select_column_names = ['home_score','away_score']


    if typeofplot == 'bar':
        custdat =mec[select_column_names]
        st.bar_chart(custdat)
    elif typeofplot:
        custdat =mec[select_column_names].plot(kind=type)
        st.write(custdat)
        st.pyplot()

###############################################################
################ PRUEBA DE GRAFICOS ###########################




mfc = p.loc[p['home_team.home_team_name'] == home_team, :]

away_df = mfc.drop('away_score', axis=1) \
                .rename(columns={'home_score': 'test'}) \
                .merge(pd.DataFrame(
                        {'Category': list(pd.np.repeat('home_score', len(mfc)))}),
                        left_index=True,
                        right_index=True)

home_df = mfc.drop('home_score', axis=1)\
               .rename(columns={'away_score': 'test'})\
               .merge(pd.DataFrame(
                   {'Category': list(pd.np.repeat('away_score', len(mfc)))}),
                   left_index=True,
                   right_index=True)





df_revised = pd.concat([home_df, away_df])


# Display original df and grouped bar chart
#st.write(df_revised)

my_exp3 = st.beta_expander('Goles marcados y recibidos en casa por el {}'.format(home_team))
with my_exp3:
    sns.set_theme(style="whitegrid")

    sns.barplot(x="test", y="away_team.away_team_name", hue="Category", data = df_revised, ci="sd" ,n_boot=100, saturation=0.9,units=20,palette="Blues_d")
    st.pyplot()

my_exp31 = st.beta_expander('Vert Goles marcados y recibidos en casa por el {}'.format(home_team))
with my_exp31:
    sns.set_theme(style="whitegrid")
    sns.barplot(x="away_team.away_team_name", y="test", hue="Category", data = df_revised, ci="hd" ,n_boot=100, saturation=0.9,units=20,color="salmon")
    st.pyplot()



##### CAMPO ####

def createPitch(length, width, unity, linecolor):  # in meters
    # Code by @JPJ_dejong

    """
    creates a plot in which the 'length' is the length of the pi$tch (goal to goal).
    And 'width' is the width of the pitch (sideline to sideline).
    Fill in the unity in meters or in yards.
    """
    # Set unity
    if unity == "meters":
        # Set boundaries
        if length >= 120.5 or width >= 75.5:
            return (str("Field dimensions are too big for meters as unity, didn't you mean yards as unity?\
                       Otherwise the maximum length is 120 meters and the maximum width is 75 meters. Please try again"))
        # Run program if unity and boundaries are accepted
        else:
            # Create figure
            fig = plt.figure()
            # fig.set_size_inches(7, 5)
            ax = fig.add_subplot(1, 1, 1)

            # Pitch Outline & Centre Line
            plt.plot([0, 0], [0, width], color=linecolor)
            plt.plot([0, length], [width, width], color=linecolor)
            plt.plot([length, length], [width, 0], color=linecolor)
            plt.plot([length, 0], [0, 0], color=linecolor)
            plt.plot([length / 2, length / 2], [0, width], color=linecolor)

            # Left Penalty Area
            plt.plot([16.5, 16.5], [(width / 2 + 16.5), (width / 2 - 16.5)], color=linecolor)
            plt.plot([0, 16.5], [(width / 2 + 16.5), (width / 2 + 16.5)], color=linecolor)
            plt.plot([16.5, 0], [(width / 2 - 16.5), (width / 2 - 16.5)], color=linecolor)

            # Right Penalty Area
            plt.plot([(length - 16.5), length], [(width / 2 + 16.5), (width / 2 + 16.5)], color=linecolor)
            plt.plot([(length - 16.5), (length - 16.5)], [(width / 2 + 16.5), (width / 2 - 16.5)], color=linecolor)
            plt.plot([(length - 16.5), length], [(width / 2 - 16.5), (width / 2 - 16.5)], color=linecolor)

            # Left 5-meters Box
            plt.plot([0, 5.5], [(width / 2 + 7.32 / 2 + 5.5), (width / 2 + 7.32 / 2 + 5.5)], color=linecolor)
            plt.plot([5.5, 5.5], [(width / 2 + 7.32 / 2 + 5.5), (width / 2 - 7.32 / 2 - 5.5)], color=linecolor)
            plt.plot([5.5, 0.5], [(width / 2 - 7.32 / 2 - 5.5), (width / 2 - 7.32 / 2 - 5.5)], color=linecolor)

            # Right 5 -eters Box
            plt.plot([length, length - 5.5], [(width / 2 + 7.32 / 2 + 5.5), (width / 2 + 7.32 / 2 + 5.5)],
                     color=linecolor)
            plt.plot([length - 5.5, length - 5.5], [(width / 2 + 7.32 / 2 + 5.5), width / 2 - 7.32 / 2 - 5.5],
                     color=linecolor)
            plt.plot([length - 5.5, length], [width / 2 - 7.32 / 2 - 5.5, width / 2 - 7.32 / 2 - 5.5], color=linecolor)

            # Prepare Circles
            centreCircle = plt.Circle((length / 2, width / 2), 9.15, color=linecolor, fill=False)
            centreSpot = plt.Circle((length / 2, width / 2), 0.8, color=linecolor)
            leftPenSpot = plt.Circle((11, width / 2), 0.8, color=linecolor)
            rightPenSpot = plt.Circle((length - 11, width / 2), 0.8, color=linecolor)

            # Draw Circles
            ax.add_patch(centreCircle)
            ax.add_patch(centreSpot)
            ax.add_patch(leftPenSpot)
            ax.add_patch(rightPenSpot)

            # Prepare Arcs
            leftArc = Arc((11, width / 2), height=18.3, width=18.3, angle=0, theta1=308, theta2=52, color=linecolor)
            rightArc = Arc((length - 11, width / 2), height=18.3, width=18.3, angle=0, theta1=128, theta2=232,
                           color=linecolor)

            # Draw Arcs
            ax.add_patch(leftArc)
            ax.add_patch(rightArc)
            # Axis titles

    # check unity again
    elif unity == "yards":
        # check boundaries again
        if length <= 95:
            return (str("Didn't you mean meters as unity?"))
        elif length >= 131 or width >= 101:
            return (str("Field dimensions are too big. Maximum length is 130, maximum width is 100"))
        # Run program if unity and boundaries are accepted
        else:
            # Create figure
            fig = plt.figure()
            # fig.set_size_inches(7, 5)
            ax = fig.add_subplot(1, 1, 1)

            # Pitch Outline & Centre Line
            plt.plot([0, 0], [0, width], color=linecolor)
            plt.plot([0, length], [width, width], color=linecolor)
            plt.plot([length, length], [width, 0], color=linecolor)
            plt.plot([length, 0], [0, 0], color=linecolor)
            plt.plot([length / 2, length / 2], [0, width], color=linecolor)

            ## the following lines of code will create
            ## the goal-post at both side of the pitch
            plt.plot([-3, 0], [(width / 2) - 5, (width / 2) - 5], color=linecolor)
            plt.plot([-3, 0], [(width / 2) + 5, (width / 2) + 5], color=linecolor)
            plt.plot([-3, -3], [(width / 2) - 5, (width / 2) + 5], color=linecolor)
            plt.plot([length + 3, length + 3], [(width / 2) - 5, (width / 2) + 5], color=linecolor)
            plt.plot([length, length + 3], [(width / 2) - 5, (width / 2) - 5], color=linecolor)
            plt.plot([length, length + 3], [(width / 2) + 5, (width / 2) + 5], color=linecolor)

            # Left Penalty Area
            plt.plot([18, 18], [(width / 2 + 18), (width / 2 - 18)], color=linecolor)
            plt.plot([0, 18], [(width / 2 + 18), (width / 2 + 18)], color=linecolor)
            plt.plot([18, 0], [(width / 2 - 18), (width / 2 - 18)], color=linecolor)

            # Right Penalty Area
            plt.plot([(length - 18), length], [(width / 2 + 18), (width / 2 + 18)], color=linecolor)
            plt.plot([(length - 18), (length - 18)], [(width / 2 + 18), (width / 2 - 18)], color=linecolor)
            plt.plot([(length - 18), length], [(width / 2 - 18), (width / 2 - 18)], color=linecolor)

            # Left 6-yard Box
            plt.plot([0, 6], [(width / 2 + 7.32 / 2 + 6), (width / 2 + 7.32 / 2 + 6)], color=linecolor)
            plt.plot([6, 6], [(width / 2 + 7.32 / 2 + 6), (width / 2 - 7.32 / 2 - 6)], color=linecolor)
            plt.plot([6, 0], [(width / 2 - 7.32 / 2 - 6), (width / 2 - 7.32 / 2 - 6)], color=linecolor)

            # Right 6-yard Box
            plt.plot([length, length - 6], [(width / 2 + 7.32 / 2 + 6), (width / 2 + 7.32 / 2 + 6)], color=linecolor)
            plt.plot([length - 6, length - 6], [(width / 2 + 7.32 / 2 + 6), width / 2 - 7.32 / 2 - 6], color=linecolor)
            plt.plot([length - 6, length], [(width / 2 - 7.32 / 2 - 6), width / 2 - 7.32 / 2 - 6], color=linecolor)

            # Prepare Circles; 10 yards distance. penalty on 12 yards
            centreCircle = plt.Circle((length / 2, width / 2), 10, color=linecolor, fill=False)
            centreSpot = plt.Circle((length / 2, width / 2), 0.8, color=linecolor)
            leftPenSpot = plt.Circle((12, width / 2), 0.8, color=linecolor)
            rightPenSpot = plt.Circle((length - 12, width / 2), 0.8, color=linecolor)

            # Draw Circles
            ax.add_patch(centreCircle)
            ax.add_patch(centreSpot)
            ax.add_patch(leftPenSpot)
            ax.add_patch(rightPenSpot)

            # Prepare Arcs
            leftArc = Arc((11, width / 2), height=20, width=20, angle=0, theta1=312, theta2=48, color=linecolor)
            rightArc = Arc((length - 11, width / 2), height=20, width=20, angle=0, theta1=130, theta2=230,
                           color=linecolor)

            # Draw Arcs
            ax.add_patch(leftArc)
            ax.add_patch(rightArc)

    # Tidy Axes
    plt.axis('off')

    return fig, ax


#TABLE WITH DATA OF THE SELECETED TEAMS

st.title('DATOS POR PARTIDO Y JUGADOR :')

st.write('Home Team:',home_team,'VS','Away Team:',away_team)

st.write(l)


##### TIROS DEL EQUIPO #####

st.header('🥅 Disparos a puerta por Equipo 🥅:')


pitch_length_X = 120
pitch_width_Y = 80

## match id in order to select diferent matches
match_list = [file_name]  # 16131, 16265, 16157, 16289, 15973, 15946, 16056, 16079, 16010, 16136, 16109, 16182, 16029, 16306, 15986, 16248, 16231
teamA = home_team  # <--- adjusted here

for match_id in match_list:

    (fig, ax) = createPitch(pitch_length_X, pitch_width_Y, 'yards', 'gray')  # < moved into for loop
    ## this is the name of our event data file for
    ## our required El Clasico

    ## loading the required event data file

    url = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/' + file_name
    resp = requests.get(url)
    my_data = json.loads(resp.text)

    df = json_normalize(my_data, sep='_').assign(match_id=file_name[:-5])


    teamB = [x for x in list(df['team_name'].unique()) if x != teamA][0]  # <--- get other team name
    ## get the nested structure into a dataframe
    ## store the dataframe in a dictionary with the match id as key

    ## making the list of all column names
    column = list(df.columns)

    ## all the type names we have in our dataframe
    all_type_name = list(df['type_name'].unique())



    ## picking shots from all_type_name
    ## a dataframe of shots
    shots_df = df.loc[df['type_name'] == 'Shot'].set_index('id')



    ## removing the columns having NaN values.
    ## after this we will have a pure shots dataframe
    shots_df.dropna(inplace=True, axis=1)

    for row_num, shot in shots_df.iterrows():
        x_loc = shot['location'][0]  ## shot location x-axis
        y_loc = shot['location'][1]  ## shot location y-axis

        goal = shot['shot_outcome_name'] == 'Goal'
        team_name = shot['team_name']

        ## assigning the circleSize as per xG value
        circleSize = np.sqrt(shot['shot_statsbomb_xg'] * 5)

        if team_name == teamA:
            if goal:
                shot_circle = plt.Circle((x_loc, pitch_width_Y - y_loc), circleSize, color='red')
                player_name = ' '.join(shot['player_name'].split(' ')[:2])
                if player_name == 'Lionel AndrÃ©s':
                    player_name = 'Messi'
                plt.text(x_loc + 2, pitch_width_Y - y_loc, player_name)
            else:
                shot_circle = plt.Circle((x_loc, pitch_width_Y - y_loc), circleSize, color='red')
                shot_circle.set_alpha(alpha=0.2)

        elif team_name == teamB:
            if goal:
                shot_circle = plt.Circle((pitch_length_X - x_loc, y_loc), circleSize, color='blue')
                player_name = ' '.join(shot['player_name'].split(' ')[:2])
                plt.text(pitch_length_X - x_loc + 2, y_loc - 1, player_name)
            else:
                shot_circle = plt.Circle((pitch_length_X - x_loc, y_loc), circleSize, color='blue')
                player_name = ' '.join(shot['player_name'].split(' ')[:2])
                shot_circle.set_alpha(alpha=0.2)

        ax.add_patch(shot_circle)

    plt.text(5, 75, teamB + ' shots')
    plt.text(80, 75, teamA + ' shots')

    fig.set_size_inches(10, 7)
    fig.savefig('{} {} {} shotmap.png'.format(teamA, teamB, match_id), dpi=50)
    plt.show()


tiros_team = st.beta_expander('🥅 Mostrar gráfica tiros y tabla de los tiros')

with tiros_team:
    st.pyplot(plt.show())
    st.write(shots_df)

    #st.write(shots_df.columns)


####  BAR CHART  ####




## TIROS POR EQUIPOS ##
my_exp56 = st.beta_expander('🥅 Comparativa de los dos equipos:')
with my_exp56:
    #FILTRO EQUIPO
    st.subheader('Tiros por jugador del {}'.format(home_team))
    allcolumns = shots_df.columns.tolist()
    typeofplot= 'bar'
    select_column_names = 'team_name'
    if typeofplot == 'bar':
        custdat2 = shots_df[select_column_names]
        st.bar_chart(custdat2)
    elif typeofplot:
        custdat2 = shots_df[select_column_names].plot(kind=type)
        st.write(custdat2)
        st.pyplot()




## TIROS POR JUGADOR HOME TEAM ##
my_exp57 = st.beta_expander('Comparativa de tiros por equipos:')
with my_exp57:
    #FILTRO EQUIPO
    shots_df21 = shots_df.loc[shots_df['team_name'] == home_team, :]
    st.subheader('Tiros por jugador del {}'.format(home_team))
    allcolumns = shots_df21.columns.tolist()
    typeofplot= 'bar'
    select_column_names = 'player_name'
    if typeofplot == 'bar':
        custdat2 = shots_df21[select_column_names]
        st.bar_chart(custdat2)
    elif typeofplot:
        custdat2 = shots_df21[select_column_names].plot(kind=type)
        st.write(custdat2)
        st.pyplot()


## TIROS POR JUGADOR AWAY TEAM ##
    # FILTRO EQUIPO
    shots_df33 = shots_df.loc[shots_df['team_name'] == away_team, :]
    st.subheader('Tiros por jugador del {}'.format(away_team))
    allcolumns = shots_df21.columns.tolist()
    typeofplot= 'bar'
    select_column_names = ['player_name']

    if typeofplot == 'bar':
        custdat2 = shots_df33[select_column_names]
        st.bar_chart(custdat2)
    elif typeofplot:
        custdat2 = shots_df33[select_column_names].plot(kind=type)
        st.write(custdat2)
        st.pyplot()





####OJO#### SIRVE PARA HACER COMPARTIVAS Y GÁFICOS

#col_opt = st.sidebar.selectbox('Columnas a comparar',('index','period','type_name','possession_team_id','possession_team_name'))

#if col_opt == 'index':
    #st.write(shots_df['index'])
#elif col_opt == 'period':
    #st.write(shots_df['period'])
#elif col_opt == 'index':
    #st.write(shots_df['type_name'])
#elif col_opt == 'possession_team_id':
    #st.write(shots_df['possession_team_id'])
#else:
    #st.write(shots_df['possession_team_name'])

kind_shot = shots_df['type_name']








# PASES COMPLETADOS

st.header('Pases de {} :'.format(playername))


## Note Statsbomb data uses yards for their pitch dimensions
pitch_length_X = 120
pitch_width_Y = 80

## match id for our El Clasico
#match_id = 16157  # 16131, 16265, 16157, 16289, 15973, 15946, 16056, 16079, 16010, 16136, 16109, 16182, 16029, 16306, 15986, 16248, 16231
teamA = home_team  # <--- adjusted here

    ## calling the function to create a pitch map
    ## yards is the unit for measurement and
    ## gray will be the line color of the pitch map
(fig, ax) = createPitch(pitch_length_X, pitch_width_Y, 'yards', 'gray')  # < moved into for loop

player_name2 = playername


    ## this is the name of our event data file for
    ## our required El Clasico
#file_name = str(match_id) + '.json'

    ## loading the required event data file

url = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/' + file_name
resp = requests.get(url)
my_data5 = json.loads(resp.text)



    ## get the nested structure into a dataframe
    ## store the dataframe in a dictionary with the match id as key
df5 = json_normalize(my_data5, sep='_').assign(match_id=file_name[:-5])
teamB = [x for x in list(df5['team_name'].unique()) if x != teamA][0]  # <--- get other team name

    ## making the list of all column names
column = list(df5.columns)

    ## all the type names we have in our dataframe
all_type_name = list(df5['type_name'].unique())


    ## creating a data frame for pass
    ## and then removing the null values
    ## only listing the player_name in the dataframe
pass_df = df5.loc[df5['type_name'] == 'Pass', :].copy()
pass_df.dropna(inplace=True, axis=1)
pass_df = pass_df.loc[pass_df['player_name'] == player_name2, :]
    ## creating a data frame for ball receipt
    ## removing all the null values
    ## and only listing Barcelona players in the dataframe
breceipt_df = df5.loc[df5['type_name'] == 'Ball Receipt*', :].copy()
breceipt_df.dropna(inplace=True, axis=1)
breceipt_df = breceipt_df.loc[breceipt_df['team_name'] == 'Barcelona', :]

pass_comp, pass_no = 0, 0

    ## iterating through the pass dataframe
for row_num, passed in pass_df.iterrows():

    if passed['player_name'] == player_name2:
        ## for away side
        x_loc = passed['location'][0]
        y_loc = passed['location'][1]
        pass_id = passed['id']
        pass_team = passed['team_name']


        events_list = [item for sublist in breceipt_df['related_events'] for item in sublist]
        if pass_id in events_list:
            ## if pass made was successful
            color = 'blue'
            label = 'Successful'
            pass_comp += 1
        else:
            ## if pass made was unsuccessful
            color = 'red'
            label = 'Unsuccessful'
            pass_no += 1



        ## plotting circle at the player's position
        shot_circle = plt.Circle((pitch_length_X - x_loc, y_loc), radius=2, color=color, label=label)
        shot_circle.set_alpha(alpha=0.2)
        ax.add_patch(shot_circle)

        ## parameters for making the arrow
        pass_x = 120 - passed['pass_end_location'][0]
        pass_y = passed['pass_end_location'][1]
        dx = ((pitch_length_X - x_loc) - pass_x)
        dy = y_loc - pass_y

        ## making an arrow to display the pass
        pass_arrow = plt.Arrow(pitch_length_X - x_loc, y_loc, -dx, -dy, width=1, color=color)

        ## adding arrow to the plot
        ax.add_patch(pass_arrow)

## computing pass accuracy
pass_acc = (pass_comp / (pass_comp + pass_no)) * 100
pass_acc = str(round(pass_acc, 2))

## adding text to the plot
plt.suptitle('{} pass map vs {}'.format(player_name2, teamB), fontsize=15)  # <-- make dynamic and change to suptitle
plt.title('Pass Accuracy: {}'.format(pass_acc), fontsize=15)  # <-- change to title

    ## handling labels
handles, labels = plt.gca().get_legend_handles_labels()
by_label = dict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys(), loc='best', bbox_to_anchor=(0.9, 1, 0, 0), fontsize=12)

## editing the figure size and saving it
fig.set_size_inches(12, 8)
fig.savefig('{} passmap.png'.format(match_id), dpi=100)  # <-- dynamic file name

## showing the plot
plt.show()
## visualization of pass
my_expander = st.beta_expander('Mostrar pases por jugador')

with my_expander:
    st.write(playername)
    st.write('Acertados:', pass_comp)
    st.write('Fallados:', pass_no)
    st.pyplot(plt.show())




#### TIROS POR JUGADOR

st.subheader('Tiros de {} :'.format(playername))

## Note Statsbomb data uses yards for their pitch dimensions
pitch_length_X = 120
pitch_width_Y = 80

## calling the function to create a pitch map
## yards is the unit for measurement and
## gray will be the line color of the pitch map
(fig, ax) = createPitch(pitch_length_X, pitch_width_Y, 'yards', 'grey')

## match id for our El Clasico
#match_id5 = 69249
#home_team = 'Real Madrid'
#away_team = 'Barcelona'
player_name4 = playername

## this is the name of our event data file for
## our required El Clasico
#file_name5 = str(match_id5) + '.json'

## loading the required event data file
##with open('../Statsbomb/data/events/' + file_name) as event_data:
##my_data = json.load(event_data, encoding='utf-8')

url = 'https://raw.githubusercontent.com/statsbomb/open-data/master/data/events/' + file_name
resp = requests.get(url)
my_data10 = json.loads(resp.text)




## get the nested structure into a dataframe
## store the dataframe in a dictionary with the match id as key

dftj = json_normalize(my_data10, sep='_').assign(match_id=file_name[:-5])

## making the list of all column names
column = list(dftj.columns)

## all the type names we have in our dataframe
all_type_name = list(dftj['type_name'].unique())

## creating the shots dataframe
shots_indv_df = dftj.loc[dftj['type_name'] == 'Shot'].set_index('index')
shots_indv_df.dropna(inplace=True, axis=1)
shots_indv_df = shots_indv_df.loc[shots_indv_df['player_name'] == player_name4, :]
tiros3 = 0
for row_num, shot in shots_indv_df.iterrows():
    x_loc = shot['location'][0]
    y_loc = shot['location'][1]

    if shot['player_name'] == player_name4:
        circleSize = np.sqrt(shot['shot_statsbomb_xg'] * 15)

        touch_circle = plt.Circle((pitch_length_X - x_loc, y_loc), circleSize, color='blue')

        if shot['shot_outcome_name'] != 'Goal':
            ## if shot outcome is not a goal then fade the circle
            touch_circle.set_alpha(0.3)

        ax.add_patch(touch_circle)
        tiros3 = tiros3 + 1

## placing the text on the plot
plt.text(10, 82, '{}\'s Shots vs Real Madrid'.format(player_name4), fontsize=12)
plt.text(80, 85, 'Darker Circles: Shot\'s outcome is a goal', fontsize=12)
plt.text(80, 82, 'Faded Circles: Shot\'s outcome is not a goal', fontsize=12)

## editing and saving the plot
fig.set_size_inches(12, 8)
fig.savefig('{}\'s Shots vs Real Madrid'.format(player_name4))
plt.show()
## displaying the figure

my_expan2 = st.beta_expander('Visualización')
with my_expan2:
    st.write(tiros3)
    st.pyplot(plt.show())











##### - TOUCH MAP -  #####


## Note Statsbomb data uses yards for their pitch dimensions
pitch_length_X = 120
pitch_width_Y = 80

## calling the function to create a pitch map
## yards is the unit for measurement and
## gray will be the line color of the pitch map
(fig, ax) = createPitch(pitch_length_X, pitch_width_Y, 'yards', 'gray')

## match id for our El Clasico
#match_id = 69249
#home_team = hom
#away_team = 'Barcelona'
player_name = playername

## this is the name of our event data file for
## our required El Clasico

## loading the required event data file
##with open('../Statsbomb/data/events/' + file_name) as event_data:
##my_data = json.load(event_data, encoding='utf-8')
#my_data11 = json.load(open(os.path.expanduser('~/Desktop/DATOS/open-data-master/data/events/' + file_name), 'r', encoding='utf-8'))

## get the nested structure into a dataframe
## store the dataframe in a dictionary with the match id as key
dftouch = json_normalize(my_data10, sep='_').assign(match_id=file_name[:-5])

## making the list of all column names
column = list(dftouch.columns)

## all the type names we have in our dataframe
all_type_name = list(dftouch['type_name'].unique())

## creating the dataframe
carry_df = dftouch.loc[dftouch['type_name'] == 'Carry'].set_index('index')
carry_df.dropna(inplace=True, axis=1)
carry_df = carry_df.loc[carry_df['player_name'] == player_name, :]

toques=0
## iterating through each rows
for row_num, carry in carry_df.iterrows():
    x_loc = carry['location'][0]
    y_loc = carry['location'][1]

    if carry['player_name'] == player_name:
        touch_circle = plt.Circle((pitch_length_X - x_loc, y_loc), radius=1.5, color='blue')

        touch_circle.set_alpha(alpha=0.8)
        ax.add_patch(touch_circle)
        toques = toques+1

## adding text to the plot
plt.text(30, 82, '{}\'s Touch Map vs Real Madrid'.format(player_name), fontsize=12)

## editing the figure and saving it
fig.set_size_inches(12, 8)
fig.savefig('{}\'s Touch Map vs Real Madrid'.format(player_name))

## displaying the plot
plt.show()

st.header('Touch Map de {} :'.format(playername))

## displaying the figure
my_expan = st.beta_expander('Visualización')
with my_expan:
    st.write('Intervenciones:',toques)
    st.pyplot(plt.show())
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import plotly.express as px
from streamlit_pandas_profiling import st_profile_report
from imports import *

def visualization_page(data_final,x):
    st.title("GDP of Countries")
    
    st.markdown(
        "This application is a Streamlit dashboard to visualize the GDP of Countries")
    # pr = x.profile_report()
    # st_profile_report(pr)

    st.text('Select the type of plot needed')
    plot_options = ['Scatter Plot', 'Histogram', 'Box Plot', 'Heat Map', 'Pair Plot']
    plot_selection = st.selectbox('Plot Type', plot_options)
    
    if plot_selection == 'Heat Map':
        fig, ax = plt.subplots(figsize=(16,16)) 
        sns.heatmap(data_final.corr(), annot=True, ax=ax, cmap='BrBG').set(title = 'Feature Correlation', xlabel = 'Columns', ylabel = 'Columns')
        st.write(fig)

    if plot_selection == 'Scatter Plot':
        st.text('Select the columns to plot')
        x_axis = st.selectbox('X axis', data_final.columns)
        y_axis = st.selectbox('Y axis', data_final.columns)
        fig = px.scatter(data_final, x=x_axis, y=y_axis, title='Scatter Plot', color='climate')
        st.plotly_chart(fig)

    if plot_selection == 'Histogram':
        st.text('Select the column to plot')
        x_axis = st.selectbox('X axis', data_final.columns)
        y_axis = st.selectbox('Y axis', data_final.columns)
        fig = px.histogram(data_final, x=x_axis, y = y_axis,title='Histogram')
        st.plotly_chart(fig)

    if plot_selection == 'Box Plot':
        st.text('Select the columns to plot')
        x_axis = st.selectbox('X axis', data_final.columns)
        y_axis = st.selectbox('Y axis', data_final.columns)
        fig = px.box(data_final, x=x_axis, y=y_axis, title='Box Plot')
        st.plotly_chart(fig)

    if plot_selection == 'Pair Plot':
        g = sns.pairplot(data_final[['population', 'area', 'net_migration', 'gdp_per_capita', 'climate']], hue='climate')
        g.fig.suptitle('Features Co-relations')
        st.write(g.fig)

    # clist = df['country'].unique().tolist()

    # country = st.sidebar.selectbox("Select a country:", clist)

    # # Data cleaning
    # df.drop(['Country Code'], axis=1, inplace=True)
    # df = df.transpose()

    # years = df.index.to_list()[1:]
    # years = [int(i) for i in years]
    # headers = df.iloc[0]
    # df = pd.DataFrame(df.values[1:], columns=headers)
    # df["years"] = years

    # startyear, endyear = st.sidebar.slider("Select a year:", value=[
    #     years[0], years[-1]], min_value=years[0], max_value=years[-1], step=1)
    # st.write("GDP of", country, "from", startyear, "to", endyear)

    # Data filtering
    #df1 = df.loc[(df[country])]
    # df1 = df[['years', country]]
    # #df1[years] = df.loc[(df["years"] >= startyear) & (df["years"] <= endyear)]
    # df1 = df1.loc[(df1["years"] >= startyear)
    #                 & (df1["years"] <= endyear)]
    # st.write(df1)

    # # Plotting
    # fig = px.line(df1, x="years", y=country, title=country)
    # fig.update_layout(
    #     yaxis_title="GDP in Billions", xaxis_title="Years")
    # st.plotly_chart(fig)

def prediction_page(data_final):
    st.title('Country GDP Estimation:')


    st.write('''
            This app will estimate the GDP per capita for a country, given some 
            attributes for that specific country as input.
            
            Please fill in the attributes below, then hit the GDP Estimate button
            to get the estimate. 
            ''')

    st.header('Input Attributes')
    att_popl = st.number_input(
        'Population (Example: 7000000)', min_value=1e4, max_value=2e9, value=2e7)
    att_area = st.slider('Area (sq. Km)', min_value=2.0,
                        max_value=17e6, value=6e5, step=1e4)
    att_dens = st.slider('Population Density (per sq. mile)',
                        min_value=0, max_value=12000, value=400, step=10)
    att_cost = st.slider('Coastline/Area Ratio', min_value=0,
                        max_value=800, value=30, step=10)
    att_migr = st.slider('Annual Net Migration (migrant(s)/1,000 population)',
                        min_value=-20, max_value=25, value=0, step=2)
    att_mort = st.slider('Infant mortality (per 1000 births)',
                        min_value=0, max_value=195, value=40, step=10)
    att_litr = st.slider('Population literacy Percentage',
                        min_value=0, max_value=100, value=80, step=5)
    att_phon = st.slider('Phones per 1000', min_value=0,
                        max_value=1000, value=250, step=25)
    att_arab = st.slider('Arable Land (%)', min_value=0,
                        max_value=100, value=25, step=2)
    att_crop = st.slider('Crops Land (%)', min_value=0,
                        max_value=100, value=5, step=2)
    att_othr = st.slider('Other Land (%)', min_value=0,
                        max_value=100, value=70, step=2)
    st.text('(Arable, Crops, and Other land should add up to 100%)')
    clim_options = ['Mostly hot(1)', 'Mostly hot and Tropical(1.5)', 'Mostly tropical(2)', 'Mostly cold and Tropical(2.5)', 'Mostly cold(3)']
    clim = st.selectbox('Climate', clim_options)
    if clim == 'Mostly hot(1)':
        att_clim = 1
    elif clim == 'Mostly hot and Tropical(1.5)':
        att_clim = 1.5
    elif clim == 'Mostly tropical(2)':
        att_clim = 2
    elif clim == 'Mostly cold and Tropical(2.5)':
        att_clim = 2.5
    else:
        att_clim = 3
    att_brth = st.slider('Annual Birth Rate (births/1,000)',
                        min_value=7, max_value=50, value=20, step=2)
    att_deth = st.slider('Annual Death Rate (deaths/1,000)',
                        min_value=2, max_value=30, value=10, step=2)
    att_agrc = st.slider('Agricultural Economy', min_value=0.0,
                        max_value=1.0, value=0.15, step=0.05)
    att_inds = st.slider('Industrial Economy', min_value=0.0,
                        max_value=1.0, value=0.25, step=0.05)
    att_serv = st.slider('Services Economy', min_value=0.0,
                        max_value=1.0, value=0.60, step=0.05)
    st.text('(Agricultural, Industrial, and Services Economy should add up to 1)')
    regn_options = ['ASIA (EX. NEAR EAST)', 'BALTICS', 'C.W. OF IND. STATES', 'EASTERN EUROPE', 'LATIN AMER. & CARIB', 'NEAR EAST', 'NORTHERN AFRICA', 'NORTHERN AMERICA', 'OCEANIA', 'SUB-SAHARAN AFRICA', 'WESTERN EUROPE']
    regn = st.selectbox('Region', regn_options)
    if regn == 'ASIA (EX. NEAR EAST)':
        att_regn = 1
    elif regn == 'BALTICS':
        att_regn = 2
    elif regn == 'C.W. OF IND. STATES':
        att_regn = 3
    elif regn == 'EASTERN EUROPE':
        att_regn = 4
    elif regn == 'LATIN AMER. & CARIB':
        att_regn = 5
    elif regn == 'NEAR EAST':
        att_regn = 6
    elif regn == 'NORTHERN AFRICA':
        att_regn = 7
    elif regn == 'NORTHERN AMERICA':
        att_regn = 8
    elif regn == 'OCEANIA':
        att_regn = 9
    elif regn == 'SUB-SAHARAN AFRICA':
        att_regn = 10
    else:
        att_regn = 11
    
    if att_regn == 1:
        att_regn_1 = 1
        att_regn_2 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 2:
        att_regn_2 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 3:
        att_regn_3 = 1
        att_regn_1 = att_regn_2 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 4:
        att_regn_4 = 1
        att_regn_1 = att_regn_3 = att_regn_2 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 5:
        att_regn_5 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_2 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 6:
        att_regn_6 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_2 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 7:
        att_regn_7 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_2 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 8:
        att_regn_8 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_2 = att_regn_9 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 9:
        att_regn_9 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_2 = att_regn_10 = att_regn_11 = 0
    elif att_regn == 10:
        att_regn_10 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_2 = att_regn_11 = 0
    else:
        att_regn_11 = 1
        att_regn_1 = att_regn_3 = att_regn_4 = att_regn_5 = att_regn_6 = att_regn_7 = att_regn_8 = att_regn_9 = att_regn_10 = att_regn_2 = 0

    user_input = np.array([att_popl, att_area, att_dens, att_cost, att_migr,
                        att_mort, att_litr, att_phon, att_arab, att_crop,
                        att_othr, att_clim, att_brth, att_deth, att_agrc,
                        att_inds, att_serv, att_regn_1, att_regn_2, att_regn_3,
                        att_regn_4, att_regn_5, att_regn_6, att_regn_7,
                        att_regn_8, att_regn_9, att_regn_10, att_regn_11]).reshape(1, -1)

    # ------
    # Model
    # ------

    #import dataset


    # def get_dataset():
    #     data = pd.read_csv('countries-of-the-world.csv')
    #     return data


    if st.button('Estimate GDP'):

        
        # Data Split
        y = data_final['gdp_per_capita']
        X = data_final.drop(['gdp_per_capita', 'country'], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=101)

        # model training
        gbm_opt = GradientBoostingRegressor(learning_rate=0.01, n_estimators=500,
                                            max_depth=5, min_samples_split=10,
                                            min_samples_leaf=1, subsample=0.7,
                                            max_features=7, random_state=101)
        gbm_opt.fit(X_train, y_train)

        # making a prediction
        # user_input is taken from input attrebutes
        gbm_predictions = gbm_opt.predict(user_input)
        st.write('The estimated GDP per capita is: ', gbm_predictions)

def main():
    st.sidebar.title("GDP of Countries") 
    page_options = ["Visualization", "Prediction"]
    page_selection = st.sidebar.selectbox("Select Page", page_options)
    data = pd.read_csv('countries-of-the-world.csv')
    x = pd.read_csv('countries-of-the-world.csv')
    # fix column names
    data.columns = (["country", "region", "population", "area", "density",
                    "coastline_area_ratio", "net_migration", "infant_mortality",
                    "gdp_per_capita", "literacy", "phones", "arable", "crops", "other",
                    "climate", "birthrate", "deathrate", "agriculture", "industry",
                    "service"])

    # Fix data types
    data.country = data.country.astype('category')
    data.region = data.region.astype('category')
    data.density = data.density.astype(str)
    data.density = data.density.str.replace(",", ".").astype(float)
    data.coastline_area_ratio = data.coastline_area_ratio.astype(str)
    data.coastline_area_ratio = data.coastline_area_ratio.str.replace(
        ",", ".").astype(float)
    data.net_migration = data.net_migration.astype(str)
    data.net_migration = data.net_migration.str.replace(",", ".").astype(float)
    data.infant_mortality = data.infant_mortality.astype(str)
    data.infant_mortality = data.infant_mortality.str.replace(
        ",", ".").astype(float)
    data.literacy = data.literacy.astype(str)
    data.literacy = data.literacy.str.replace(",", ".").astype(float)
    data.phones = data.phones.astype(str)
    data.phones = data.phones.str.replace(",", ".").astype(float)
    data.arable = data.arable.astype(str)
    data.arable = data.arable.str.replace(",", ".").astype(float)
    data.crops = data.crops.astype(str)
    data.crops = data.crops.str.replace(",", ".").astype(float)
    data.other = data.other.astype(str)
    data.other = data.other.str.replace(",", ".").astype(float)
    data.climate = data.climate.astype(str)
    data.climate = data.climate.str.replace(",", ".").astype(float)
    data.birthrate = data.birthrate.astype(str)
    data.birthrate = data.birthrate.str.replace(",", ".").astype(float)
    data.deathrate = data.deathrate.astype(str)
    data.deathrate = data.deathrate.str.replace(",", ".").astype(float)
    data.agriculture = data.agriculture.astype(str)
    data.agriculture = data.agriculture.str.replace(",", ".").astype(float)
    data.industry = data.industry.astype(str)
    data.industry = data.industry.str.replace(",", ".").astype(float)
    data.service = data.service.astype(str)
    data.service = data.service.str.replace(",", ".").astype(float)

    # fix missing data
    data['net_migration'].fillna(0, inplace=True)
    data['infant_mortality'].fillna(0, inplace=True)
    data['gdp_per_capita'].fillna(2500, inplace=True)
    data['literacy'].fillna(data.groupby(
        'region')['literacy'].transform('mean'), inplace=True)
    data['phones'].fillna(data.groupby('region')[
                        'phones'].transform('mean'), inplace=True)
    data['arable'].fillna(0, inplace=True)
    data['crops'].fillna(0, inplace=True)
    data['other'].fillna(0, inplace=True)
    data['climate'].fillna(0, inplace=True)
    data['birthrate'].fillna(data.groupby(
        'region')['birthrate'].transform('mean'), inplace=True)
    data['deathrate'].fillna(data.groupby(
        'region')['deathrate'].transform('mean'), inplace=True)
    data['agriculture'].fillna(0.17, inplace=True)
    data['service'].fillna(0.8, inplace=True)
    data['industry'].fillna(
        (1 - data['agriculture'] - data['service']), inplace=True)

    # Region Transform
    data_final = pd.concat([data, pd.get_dummies(
        data['region'], prefix='region')], axis=1).drop(['region'], axis=1)

    if page_selection == "Visualization":
        visualization_page(data_final,x)
            
    elif page_selection == "Prediction":
        prediction_page(data_final)
                

if __name__ == "__main__":
    main()

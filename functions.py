import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
pd.set_option('display.max_columns', None)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import scipy.stats as stats
from statsmodels.formula.api import ols

import itertools

def process_date_time(df, date_column):
    # Convert the 'date_column' to datetime data type
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Extract day and week from the date
    df['day'] = df[date_column].dt.day
    df['week'] = df[date_column].dt.week
    
    # Convert time to morning, afternoon, evening, or night
    df['time_period'] = pd.cut(df[date_column].dt.hour, bins=[0, 6, 12, 18, 24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
    
    df['DayOfTheWeek'] = df[date_column].apply(lambda x: x.strftime('%A'))
    
    df['Weekday'] = df['DayOfTheWeek'].apply(lambda x: 0 if x in ['Saturday', 'Sunday'] else 1)
    
    return df

def nullValuePercentageCheck(df):
    null_percentage = df.isna().mean() * 100
    return null_percentage

def dataTypeCheck(df):
    return df.dtypes


def calculateSpeed(df, var1, var2, new_var):
    df[new_var] = df[var1]/df[var2]
    return df

def createNewFeatures(df, predicted_distance_col, predicted_duration_col):
    # Multiply predicted_distance by predicted_duration to get the total travel time
    df['distance_time'] = df[predicted_distance_col] * df[predicted_duration_col]

    # Divide predicted_duration by predicted_distance to get the inverse of speed
    # This represents the time taken to cover a unit distance
    df['time_distance'] = df[predicted_duration_col] / df[predicted_distance_col]

    # Calculate the square of predicted_distance to capture non-linear relationships
    # or exponential effects associated with longer distances
    df['distance_squared'] = df[predicted_distance_col] ** 2

    # Calculate the square of predicted_duration to capture non-linear relationships
    # or time-dependent effects
    df['time_squared'] = df[predicted_duration_col] ** 2

    return df

def count_positive_negative(df, column):
    positive_count = (df[column] > 0).sum()
    negative_count = (df[column] < 0).sum()
    
    print("Positive Count:", positive_count)
    print("Negative Count:", negative_count)


def categoricalValuesCheck(df):
    categorical_columns = df.select_dtypes(include=['object', 'category']).columns
    for column in categorical_columns:
        unique_values = df[column].unique()
        print(f"Unique values in {column}:")
        print(unique_values)
        print()
        
def calculateDifferencesAndPercentage(df, *args):
    for col1, col2 in args:
        df[col1 + '_difference'] = df[col1] - df[col2]
        
        df[col1 + '_difference_percentage'] = abs(df[col1 + '_difference'] / df[col2]) * 100 
    return df

def segregatingDataBasedOnPriceDifference(df, col):
    df_significant_price_difference = df[df[col] > 20]
    df_non_significant_price_difference = df[(df[col] <= 20)]
    df_nan = df[df[col].isna()]
    return df_significant_price_difference, df_non_significant_price_difference, df_nan

def generate_pie_chart(df, df1, df2, df3):
    # Calculate count percentages
    count_percentages = [
        df1.shape[0] / df.shape[0] * 100,
        df2.shape[0] / df.shape[0] * 100,
        df3.shape[0] / df.shape[0] * 100
    ]

    # Define labels and colors
    labels = ['Significant Difference', 'Non-Significant Difference', 'NaN Values']
    colors = ['lightblue', 'lightgreen', 'lightpink']

    # Create a pie chart
    plt.pie(count_percentages, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')  # Equal aspect ratio ensures the pie chart is drawn as a circle
    plt.title('Pie Chart')  # Add a title

    # Display the chart
    plt.show()

    
def columnNormalization(df, cols):
    # Create a MinMaxScaler object
    scaler = MinMaxScaler()

    # Normalize the selected columns
    df[cols] = scaler.fit_transform(df[cols])   
    return df    

def removeOutliers(df, column):
    # Calculate the IQR (Interquartile Range)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    # Define the upper and lower bounds for outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Create a box plot to visualize outliers
    plt.figure()
    plt.boxplot(df[column])
    plt.title('Box Plot - ' + column)
    plt.show()
    
    # Create a new dataframe with outliers
    df_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    
    # Create a new dataframe without outliers
    df_filtered = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    return df_filtered, df_outliers

def plotDensities(df, value_sets):
    # Loop through the value sets
    for idx, value_set in enumerate(value_sets):
        fig, ax = plt.subplots()
        
        # Extract the label for the set
        label = value_set[0]
        
        # Loop through the columns in the set
        for column in value_set[1:]:
            # Plot the distribution of the column
            sns.histplot(df[column], kde=True, label=column)
        
        # Set the plot title and labels
        ax.set_title(f'Distribution of {label}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Density')
        
        # Add a legend
        ax.legend()
        
        # Show the plot
        plt.show()

    
    
def plotSum(df, x_values, y):
    for x in x_values:
        df[x] = df[x].astype(str)  # Convert X values to string type
        sums = df.groupby(x)[y].count().reset_index()
        sns.barplot(data=sums, x=x, y=y)
        plt.xlabel(x)
        plt.ylabel("Count")
        plt.title(f"Count Plot of {x}")
        plt.show()

def regressionAnalysis(X, y):
    # Convert categorical variables into dummy variables
    X = pd.get_dummies(X, drop_first=True)
    
    # Define column names for the results DataFrame
    columns = ['Variable', 'R-squared', 'P-value']
    
    # Create an empty DataFrame to store the results
    results_df = pd.DataFrame(columns=columns)

    # Perform individual variable regression
    for variable in X.columns:
        X_single = sm.add_constant(X[variable])
        model_single = sm.OLS(y, X_single)
        results_single = model_single.fit()
        r_squared_single = results_single.rsquared
        p_value_single = results_single.pvalues.iloc[1]  # Extract p-value for the variable (index 1)
        results_df = pd.concat([results_df, pd.DataFrame({'Variable': [variable], 'R-squared': [r_squared_single], 'P-value': [p_value_single]})])

    # Perform regression with all variables combined
    X_combined = sm.add_constant(X)
    model_combined = sm.OLS(y, X_combined)
    results_combined = model_combined.fit()
    r_squared_combined = results_combined.rsquared
    p_value_combined = results_combined.pvalues.iloc[0]  # Extract p-value for the constant term
    results_df = pd.concat([results_df, pd.DataFrame({'Variable': ['All Combined'], 'R-squared': [r_squared_combined], 'P-value': [p_value_combined]})])

    # Sort the results based on R-squared in descending order
    results_df = results_df.sort_values(by='R-squared', ascending=False).reset_index(drop=True)

    return results_df

    

def performAnova(df, dependent_var, independent_vars):
    # Build the formula for OLS regression
    formula = f"{dependent_var} ~ {' + '.join(independent_vars)}"
    
    # Fit the OLS regression model
    model = ols(formula, data=df).fit()
    
    # Perform ANOVA analysis and generate the table
    anova_table = sm.stats.anova_lm(model, typ=2)
    
    return anova_table


def plot_variable_correlation(df, dependent_var, independent_vars):
    # Select the variables of interest from the DataFrame
    variables = independent_vars + [dependent_var]
    data = df[variables]
    
    # Calculate the correlation matrix
    correlation_matrix = data.corr(method='pearson')
    
    # Sort the correlation values in descending order
    sorted_correlation = correlation_matrix[[dependent_var]].drop([dependent_var]).sort_values(by=dependent_var, ascending=False)
    
    # Plot the correlation between independent variables and the dependent variable in descending order
    plt.figure(figsize=(10, 8))
    sns.heatmap(sorted_correlation, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f")
    plt.title("Correlation between Independent Variables and Dependent Variable")
    plt.show()

    return sorted_correlation

def plot_relationships(df, y, X):
    # Loop through the X variables
    for x in X:
        # Create a new plot for each x variable
        fig, ax = plt.subplots()
        
        # Plot scatter plot with regression line
        sns.regplot(data=df, x=x, y=y)
        
        # Set the x-axis label, y-axis label, and title
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        ax.set_title(f"Scatter Plot: {x} vs {y}")
        
        # Show the plot
        plt.show()

def topAppVersionPlot(df, rider_app_version, CA, CI, metered_price_difference_percentage):
    new_df = df[['rider_app_version','metered_price_difference_percentage']]
    new_df['rider_app_version_sorted'] = new_df[rider_app_version].str.replace('CA.', '').str.replace('CI.', '').astype(float)
    new_df_1 = new_df.groupby('rider_app_version_sorted')['metered_price_difference_percentage'].count().sort_values(ascending = False ).head(20)
    top_values = new_df_1.head(8)
    top_values = pd.DataFrame(top_values).reset_index()
    top_values['rider_app_version_sorted'] = top_values['rider_app_version_sorted'].astype(str)

    # Plot the dataframe on a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_values, x='rider_app_version_sorted', y='metered_price_difference_percentage')
    plt.xlabel('Rider App Version')
    plt.ylabel('Count')
    plt.title('Top 8 Rider App Versions')
    plt.xticks(rotation=90)
    plt.show()
    return top_values.head()


def topDevicesUsed(df, Device_name):
    new_df = df[['order_id_new','device_name','gps_confidence']]
    new_df['Device_company'] = new_df[Device_name].str.split().str[0]
    new_df['Device_company'] = new_df['Device_company'].str.split('_|-').str[0].str.split(',').str[0].str.replace(r'iPhone.*', 'iPhone', regex=True)

    # Filter the DataFrame where GPS is 0
    filtered_df1 = new_df[new_df['gps_confidence'] == 0]

    # Count the occurrences of each phone company
    company_counts1 = filtered_df1['Device_company'].value_counts()
    
    top_values = pd.DataFrame(company_counts1).reset_index()
    
    top_values = top_values.rename(columns={'index': 'company_name', 'Device_company': 'count'})
        # Plot the dataframe on a bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=top_values.head(6), x='company_name', y='count')
    plt.xlabel('Device company')
    plt.ylabel('Count')
    plt.title('Top Devices with gps confidence = 0')
    plt.xticks(rotation=90)
    plt.show()

    return top_values.head(6)
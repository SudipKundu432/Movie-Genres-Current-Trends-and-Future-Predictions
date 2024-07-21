import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Read the data from CSV file
data = pd.read_csv('movie_data.csv')

# Define the specified genres
specified_genres = ['Romance', 'Action', 'Drama', 'Comedy', 'Family', 'Thriller', 
                    'Musical', 'Mystery', 'Biography', 'Horror', 'Crime', 'Sci-Fi', 'War', 
                    'Fantasy', 'History', 'Adventure', 'Epic', 'Period', 'Biographical', 
                    'Sports', 'Spy', 'Dance']

# Function to calculate average rating, box office collection, and count of movies for each genre in a given period
def calculate_averages_and_counts(data, start_year, end_year):
    period_data = data[(data['Year'].dt.year >= start_year) & (data['Year'].dt.year <= end_year)]
    genre_stats = {genre: period_data[period_data['Genre'].str.contains(genre, case=False)] for genre in specified_genres}
    genre_ratings = {genre: stats['Rating'].mean() for genre, stats in genre_stats.items()}
    genre_boxoffice = {genre: stats['BoxOfficeCollection'].mean() for genre, stats in genre_stats.items()}
    genre_counts = {genre: len(stats) for genre, stats in genre_stats.items()}
    return dict(sorted(genre_ratings.items())), dict(sorted(genre_boxoffice.items())), dict(sorted(genre_counts.items()))

# Function to plot bar charts with counts
def plot_bar_chart_with_counts(data, counts, title, ylabel, color):
    labels = [f"{genre} ({count} movies)" for genre, count in counts.items()]
    plt.figure(figsize=(12, 6))
    plt.bar(labels, data.values(), color=color)
    plt.title(title)
    plt.xlabel('Genre')
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

# Function to prepare and forecast data using Holt-Winters method
def forecast_data(genre_yearly_data, steps=4):
    forecasted_data = {}
    for genre, data in genre_yearly_data.items():
        if len(data) > 2:  # Holt-Winters needs at least 3 points to work
            data = data.reindex(range(2000, 2022), fill_value=0)
            model = ExponentialSmoothing(data, trend='add', seasonal=None)
            fit = model.fit()
            forecasted_data[genre] = fit.forecast(steps=steps)
    return forecasted_data

# Function to plot forecasted data
def plot_forecasted_data(forecasted_data, title, ylabel):
    plt.figure(figsize=(14, 7))
    for genre, forecast in forecasted_data.items():
        plt.plot(forecast.index, forecast, label=genre, marker='o')
    plt.title(title)
    plt.xlabel('Year')
    plt.ylabel(ylabel)
    plt.xticks(range(2022, 2026))
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), ncol=2, fontsize='small')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Convert 'Year' column to datetime
data['Year'] = pd.to_datetime(data['Year'], format='%Y')

# Calculate averages and counts for the periods 2000-2010 and 2011-2021
ratings_2000_2010, boxoffice_2000_2010, counts_2000_2010 = calculate_averages_and_counts(data, 2000, 2010)
ratings_2011_2021, boxoffice_2011_2021, counts_2011_2021 = calculate_averages_and_counts(data, 2011, 2021)

# Plotting bar charts for each period
plot_bar_chart_with_counts(ratings_2000_2010, counts_2000_2010, 'Average Rating for Each Genre from 2000 to 2010', 'Average Rating', 'skyblue')
plot_bar_chart_with_counts(boxoffice_2000_2010, counts_2000_2010, 'Average Box Office Collection for Each Genre from 2000 to 2010', 'Average Box Office Collection (INR in Crores)', 'lightgreen')
plot_bar_chart_with_counts(ratings_2011_2021, counts_2011_2021, 'Average Rating for Each Genre from 2011 to 2021', 'Average Rating', 'skyblue')
plot_bar_chart_with_counts(boxoffice_2011_2021, counts_2011_2021, 'Average Box Office Collection for Each Genre from 2011 to 2021', 'Average Box Office Collection (INR in Crores)', 'lightgreen')

# Calculate overall averages and counts for 2000-2021
ratings_2000_2021, boxoffice_2000_2021, counts_2000_2021 = calculate_averages_and_counts(data, 2000, 2021)

# Plotting overall average charts for 2000-2021
plot_bar_chart_with_counts(ratings_2000_2021, counts_2000_2021, 'Average Rating for Each Genre from 2000 to 2021', 'Average Rating', 'skyblue')
plot_bar_chart_with_counts(boxoffice_2000_2021, counts_2000_2021, 'Average Box Office Collection for Each Genre from 2000 to 2021', 'Average Box Office Collection (INR in Crores)', 'lightgreen')

# Filter data from 2000 to 2021
data = data[(data['Year'].dt.year >= 2000) & (data['Year'].dt.year <= 2021)]

# Create dictionaries to store the average ratings and box office collections per year for each genre
genre_yearly_ratings = {genre: data[data['Genre'].str.contains(genre, case=False)].groupby(data['Year'].dt.year)['Rating'].mean() for genre in specified_genres}
genre_yearly_boxoffice = {genre: data[data['Genre'].str.contains(genre, case=False)].groupby(data['Year'].dt.year)['BoxOfficeCollection'].mean() for genre in specified_genres}

# Forecast the average ratings and box office collections for each genre from 2022 to 2025
forecast_ratings = forecast_data(genre_yearly_ratings)
forecast_boxoffice = forecast_data(genre_yearly_boxoffice)

# Plotting the forecasted average ratings
plot_forecasted_data(forecast_ratings, 'Genre Preference Prediction (Average Rating) for the Next 4 Years (2022-2025)', 'Predicted Average Rating')

# Plotting the forecasted average box office collections
plot_forecasted_data(forecast_boxoffice, 'Genre Preference Prediction (Box Office Collections) for the Next 4 Years (2022-2025)', 'Predicted Box Office Collection (INR in Crores)')

# Plotting trend line graph for average rating over years (2000 to 2025)
plt.figure(figsize=(14, 7))
for genre, yearly_ratings in genre_yearly_ratings.items():
    plt.plot(yearly_ratings.index, yearly_ratings.values, label=genre)
for genre, forecast in forecast_ratings.items():
    plt.plot(range(2022, 2026), forecast, label=f"{genre} (forecast)", linestyle='--')
plt.title('Trend Line for Average Rating (2000-2025)')
plt.xlabel('Year')
plt.ylabel('Average Rating')
plt.xticks(range(2000, 2026))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plotting trend line graph for box office collection over years (2000 to 2025)
plt.figure(figsize=(14, 7))
for genre, yearly_boxoffice in genre_yearly_boxoffice.items():
    plt.plot(yearly_boxoffice.index, yearly_boxoffice.values, label=genre)
for genre, forecast in forecast_boxoffice.items():
    plt.plot(range(2022, 2026), forecast, label=f"{genre} (forecast)", linestyle='--')
plt.title('Trend Line for Box Office Collection (2000-2025)')
plt.xlabel('Year')
plt.ylabel('Box Office Collection (INR in Crores)')
plt.xticks(range(2000, 2026))
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=2, fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()

# Determine the genre with the highest predicted average rating and box office collection in the last forecasted year
most_preferred_genre_rating = max(forecast_ratings, key=lambda x: forecast_ratings[x].iloc[-1])
most_preferred_genre_boxoffice = max(forecast_boxoffice, key=lambda x: forecast_boxoffice[x].iloc[-1])

print(f"The most preferred genre based on average rating in the next 4 years is predicted to be: {most_preferred_genre_rating}")
print(f"The most preferred genre based on box office collections in the next 4 years is predicted to be: {most_preferred_genre_boxoffice}")

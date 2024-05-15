import keyword
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import linregress
from tqdm import tqdm
import seaborn as sns

matplotlib.use('TkAgg')

workout_history = pd.read_csv('strong.csv', header=None, sep=';', names=['Date', 'Workout Name', 'Exercise Name',
                                                                         'Set Order', 'Weight', 'Weight Unit', 'Reps',
                                                                         'RPE', 'Distance', 'Distance Unit', 'Seconds',
                                                                         'Notes', 'Workout Notes'])
workout_history = workout_history.drop(columns=['Workout Notes', 'Notes'])

workout_history = workout_history[(workout_history['Date'] > '2021-01-11')]
# Define programs
gslp = workout_history[workout_history['Workout Name'].str.startswith('GSLP')]
gslp.name = 'GreySkull LP'

BBB1Start = '2021-07-12'
BBB1End = '2022-03-10 99:99:99'
BBB1Exclude = ['2021-12-21', '2021-12-22']
BBB1 = workout_history[(workout_history['Date'] >= BBB1Start) & (workout_history['Date'] <= BBB1End)]
BBB1.name = '5/3/1 BBB - 1'

UnityStart = '2022-03-14'
UnityEnd = '2022-04-21 99:99:99'
Unity = workout_history[(workout_history['Date'] >= UnityStart) & (workout_history['Date'] <= UnityEnd)]
Unity.name = 'Unity - John Meadows'

BBB2Start = '2022-04-30'
BBB2End = '2022-06-10 99:99:99'
BBB2 = workout_history[(workout_history['Workout Name'].str.startswith('Bbb')) &
                       (workout_history['Date'] >= BBB2Start) &
                       (workout_history['Date'] <= BBB2End)]
BBB2.name = '5/3/1 BBB - 2'

SBS1Start = '2022-06-23'
SBS1End = '2022-07-18 99:99:99'
SBS1 = workout_history[(workout_history['Date'] >= SBS1Start) & (workout_history['Date'] <= SBS1End)]
SBS1.name = 'Stronger By Science - Hypertrophy'

SS1Start = '2022-07-19'
SS1End = '2022-08-27 99:99:99'
SS1 = workout_history[(workout_history['Date'] >= SS1Start) & (workout_history['Date'] <= SS1End)]
SS1.name = 'Super Squats'

BM1Start = '2022-09-06'
BM1End = '2022-10-09 99:99:99'
BM1 = workout_history[(workout_history['Date'] >= BM1Start) & (workout_history['Date'] <= BM1End)]
BM1.name = 'BullMastiff - 1'

GZCLStart = '2022-11-21'
GZCLEnd = '2023-01-07 99:99:99'
GZCL = workout_history[(workout_history['Date'] >= GZCLStart) & (workout_history['Date'] <= GZCLEnd)]
GZCL.name = 'GZCL - LP'

JNTStart = '2023-01-16'
JNTEnd = '2023-04-06 99:99:99'
JNT = workout_history[(workout_history['Date'] >= JNTStart) & (workout_history['Date'] <= JNTEnd)]
JNT.name = 'Jacked and Tan 2.0'

BM2Start = '2023-04-10'
BM2End = '2023-05-23 99:99:99'
BM2 = workout_history[(workout_history['Date'] >= BM2Start) & (workout_history['Date'] <= BM2End)]
BM2.name = 'BullMastiff - 2'

WarlockStart = '2023-08-28'
WarlockEnd = '2023-10-19 99:99:99'
Warlock = workout_history[(workout_history['Date'] >= WarlockStart) & (workout_history['Date'] <= WarlockEnd)]
Warlock.name = 'Warlock - John Meadows'

BBB3Start = '2023-11-27'
BBB3End = '2023-12-29 99:99:99'
BBB3 = workout_history[(workout_history['Date'] >= BBB3Start) & (workout_history['Date'] <= BBB3End)]
BBB3.name = '5/3/1 BBB - 3'

PBStart = '2024-01-16'
PBEnd = '2024-02-29 99:99:99'
PB = workout_history[(workout_history['Date'] >= PBStart) & (workout_history['Date'] <= PBEnd)]
PB.name = 'PowerBuilding - Bromley'

UHFStart = '2024-03-11'
UHFEnd = '2024-05-11 99:99:99'
UHF = workout_history[(workout_history['Date'] >= UHFStart) & (workout_history['Date'] <= UHFEnd)]
UHF.name = 'GZCL - UHF'

# Group by day and name
exercise_counts = workout_history.groupby(['Date', 'Exercise Name']).size()
# Find the top 5 most frequent exercises
top_5_exercises = exercise_counts.groupby('Exercise Name').size().nlargest(5)
print("Top 5 most frequent exercises:")
print(top_5_exercises)

# Define what is a push and pull exercise
push_keywords = ['bench', 'press', 'squat', 'extension', 'skullcrusher', 'chest', 'raise', 'shoulder']
pull_keywords = ['deadlift', 'row', 'pull', 'shrug', 'chin', 'curl']


# Check if exercise is push or pull
def is_push(name):
    return any(keyword in name.lower() for keyword in push_keywords)
def is_pull(name):
    return any(keyword in name.lower() for keyword in pull_keywords)

# Get total amount of push and pull reps
total_push = workout_history.loc[workout_history['Exercise Name'].apply(is_push)].groupby('Workout Name')['Reps'] \
    .sum().sum()
total_pull = workout_history.loc[workout_history['Exercise Name'].apply(is_pull)].groupby('Workout Name')['Reps'] \
    .sum().sum()
print('Push to Pull Ratio: ', round(total_push / total_pull, 3))\


# Exercise selector
exercise_name = 'Deadlift (Barbell)'
# Calculates an average 1rm based on the best set of the day for a given exercise, over every workout
# works for any data set (in the Strong app format)
exercise_data = workout_history.loc[workout_history['Exercise Name'] == exercise_name]
exercise_data = exercise_data[exercise_data['Reps'] <= 10]
exercise_data['One-Rep Max B'] = exercise_data['Weight'] * (36 / (37 - exercise_data['Reps']))
exercise_data['One-Rep Max E'] = exercise_data['Weight'] * (1 + (.033 * exercise_data['Reps']))
exercise_data['One-Rep Max L'] = exercise_data['Weight'] * pow(exercise_data['Reps'], 0.1)
exercise_data['One-Rep Max O'] = exercise_data['Weight'] * (1 + (.025 * exercise_data['Reps']))
exercise_data['1rm Average'] = (exercise_data['One-Rep Max B'] + exercise_data['One-Rep Max E'] +
                                exercise_data['One-Rep Max L'] + exercise_data['One-Rep Max O']) / 4
max_1rm = exercise_data.groupby('Date')['1rm Average'].max().reset_index()

sns.catplot(x=max_1rm['Date'].str[:10], y='1rm Average', data=max_1rm, kind='point', height=5, aspect=10 / 5)
sns.regplot(data=max_1rm, x=max_1rm.index, y='1rm Average')
plt.title(f'One-Rep Max for {exercise_name} Across Workouts')
plt.xlabel('Index')
plt.ylabel('One-Rep Max')
plt.xticks(rotation=75)
plt.show()

# Slope of the regression line
slope, intercept, r_value, p_value, std_err = linregress(max_1rm.index, max_1rm['1rm Average'])
print("Slope of the regression line:", slope)

# List of programs
programs = [gslp, BBB1, Unity, BBB2, SBS1, SS1, BM1, GZCL, JNT, BM2, Warlock, BBB3, PB, UHF]
results = pd.DataFrame(columns=['Program', 'Date', '1RM'])
# Iterate over each program
for program in programs:
    # Filter workout history for the current program, exercise, and rep count
    program_data = program.loc[program['Exercise Name'] == exercise_name]
    program_data = program_data[program_data['Reps'] <= 10]
    # Calculate the one-rep max for each set using the Brzycki method
    program_data['1RM'] = ((program_data['Weight']) * (36 / (37 - program_data['Reps'])))

    # Group by date
    max_1rm_2 = program_data.groupby('Date')['1RM'].max().reset_index()

    # Add the results to the DataFrame
    max_1rm_2['Program'] = program.name
    results = pd.concat([results, max_1rm_2], ignore_index=True)

# Plot the one-rep max progression for each program on the same chart
sns.catplot(x=results['Date'].str[:10], y='1RM', hue='Program', data=results, kind='point', height=5, aspect=10 / 5)
plt.title(f'One-Rep Max for {exercise_name} Across Programs')
plt.xlabel('Date')
plt.ylabel('One-Rep Max')
plt.xticks(rotation=75)
plt.show()


# todo
# Calculate and plot the 1 rep max increase during a program versus total reps of exercise

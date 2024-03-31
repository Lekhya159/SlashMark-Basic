import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import random
import numpy as np

tasks = pd.read_csv(r'C:\Users\kasth\OneDrive\Documents\AICTE\Basic\Task - 1\basic\tasks.csv')

def save_tasks():
    tasks.to_csv('tasks.csv', index=False)

vectorizer = CountVectorizer()
clf = MultinomialNB()
model = make_pipeline(vectorizer, clf)

def train_model():
    if not tasks.empty:
        model.fit(tasks['description'], tasks['priority'])

def add_task(description, priority):
    new_task = pd.DataFrame({'description': [description], 'priority': [priority]})
    tasks = pd.concat([tasks, new_task], ignore_index=True)
    save_tasks()
    train_model()

def remove_task(description):
    global tasks
    tasks = tasks[tasks['description'] != description]
    save_tasks()
    train_model()

def list_tasks():
    if tasks.empty:
        print("No tasks available.")
    else:
        print(tasks)


def recommend_task():
    if not tasks.empty:
        high_priority_tasks = tasks[tasks['priority'] == 'High']

        if not high_priority_tasks.empty:
            recommendation_index = np.random.choice(high_priority_tasks.index)
            recommended_task = high_priority_tasks.loc[recommendation_index]
            print(f"Recommended task: {recommended_task['description']} - Priority: High")
        else:
            medium_priority_tasks = tasks[tasks['priority'] == 'Medium']
            if not medium_priority_tasks.empty:
                recommendation_index = np.random.choice(medium_priority_tasks.index)
                recommended_task = medium_priority_tasks.loc[recommendation_index]
                print(f"Recommended task: {recommended_task['description']} - Priority: Medium")
            else:
                print("No high or medium-priority tasks available for recommendation.")
    else:
        print("No tasks available for recommendations.")

while True:
    print("\nTask Management App")
    print("1. Add Task")
    print("2. Remove Task")
    print("3. List Tasks")
    print("4. Recommend Task")
    print("5. Exit")

    choice = input("Select an option: ")

    if choice == "1":
        description = input("Enter task description: ")
        priority = input("Enter task priority (Low/Medium/High): ").capitalize()
        add_task(description, priority)
        print("Task added successfully.")

    elif choice == "2":
        description = input("Enter task description to remove: ")
        remove_task(description)
        print("Task removed successfully.")

    elif choice == "3":
        list_tasks()

    elif choice == "4":
        recommend_task()

    elif choice == "5":
        print("Goodbye!")
        break

    else:
        print("Invalid option. Please select a valid option.")

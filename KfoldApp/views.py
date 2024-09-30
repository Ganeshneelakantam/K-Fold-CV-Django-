from django.shortcuts import render, redirect
from django.http import HttpResponseBadRequest
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
import pandas as pd
import traceback
import json
from django.http import HttpResponseRedirect
from urllib.parse import urlencode


# Define the file validation function
def validate_uploaded_file(uploaded_file):
    if not uploaded_file.name.endswith('.csv'):
        return False
    return True

def index(request):
    error_message = request.GET.get('error_message')
    return render(request, 'index.html', context={'error_message': error_message})

def documentation(request):
    return render(request, 'document.html')

def validate_numeric_columns(df):
    """Validate if the DataFrame contains numeric columns."""
    return not df.select_dtypes(include=['number']).empty

def upload(request):
    if request.method != 'POST':
        return HttpResponseBadRequest('Method Not Allowed')

    try:
        uploaded_file = request.FILES['file']
        folds = int(request.POST['folds'])
        kernel = request.POST['kernel']
        random_state = int(request.POST['random_state'])
    except (KeyError, ValueError):
        return HttpResponseRedirect('/?error_message=Invalid form data.')

    if not validate_uploaded_file(uploaded_file):
        return HttpResponseRedirect('/?error_message=Invalid file type. Only CSV files are allowed.')

    try:
        df = pd.read_csv(uploaded_file)
    except Exception as e:
        error_message = f'Error loading dataset: {e}'
        print("Error loading dataset:", traceback.format_exc())
        return HttpResponseRedirect(f'/index?error_message={error_message}')

    # Check if the dataset has numeric columns
    if not validate_numeric_columns(df):
        return HttpResponseRedirect('/?error_message=Error: Dataset contains no numeric columns. Please check your dataset.')

    cross_validator = KFold(n_splits=folds, shuffle=True, random_state=42)

    algorithms = {
        'SVM': SVC(kernel=kernel, random_state=random_state),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'KNN': KNeighborsClassifier(),
        'Random Forest': RandomForestClassifier(random_state=random_state),
        'Logistic Regression': LogisticRegression(random_state=random_state),
        'Naive Bayes': GaussianNB(),
        'Gradient Boosting': GradientBoostingClassifier(random_state=random_state),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Neural Network': MLPClassifier(random_state=random_state)
    }

    X = df.select_dtypes(include=['number']).values
    y = df.iloc[:, -1].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    results = {'individual_fold_accuracies': {}, 'algorithm_accuracies': {}, 'average_accuracy': {},
               'num_folds': folds}

    for algorithm, model in algorithms.items():
        try:
            accuracies = cross_val_score(model, X, y, cv=cross_validator)
            average_accuracy = accuracies.mean()
            results['individual_fold_accuracies'][algorithm] = accuracies.tolist()
            results['algorithm_accuracies'][algorithm] = average_accuracy
            results['average_accuracy'][algorithm] = average_accuracy
        except Exception as e:
            error_message = f'Error during cross-validation for {algorithm}: {e}'
            print(f"Error during cross-validation for {algorithm}:", traceback.format_exc())
            return redirect('index', error_message=error_message)

    # dataset_head = dataset_head = df.head(10).to_dict()
    dataset_head = df.head(5).to_dict(orient='list')  # Ensure columns are lists of values

    # Define how many rows you want to display
    num_rows = 5

    row_range = range(num_rows)  # Create the range

    # Prepare fold range and pass it to the template
    fold_range = range(1, results['num_folds'] + 1)

    dataset_columns = list(dataset_head.keys())  # Extract the column names (keys)
    dataset_items = list(dataset_head.items())  # Convert the items to a list of tuples
    dataset_items = df.head(num_rows).values.tolist()

    # Extract and prepare data for plotting
    algorithm_accuracies = results['average_accuracy']
    algorithm_names = list(algorithm_accuracies.keys())
    accuracies = list(algorithm_accuracies.values())

    return render(request, 'results.html', context={
        'results': results,
        'dataset_head': dataset_head,
        'dataset_columns': dataset_columns,
        'dataset_items': dataset_items,
        'row_range': row_range,
        'fold_range': fold_range,
        'individual_fold_accuracies': results['individual_fold_accuracies'].items(),
        'average_accuracies': results['average_accuracy'].items(),
        'algorithm_accuracies': json.dumps(accuracies),
        'algorithm_names': json.dumps(algorithm_names),
    })


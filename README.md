# evolutionary_algorithm

`evolutionary-algorithm` is a Python library adapted from https://pypi.org/project/geneticalgorithm/ with modifications for streamlining the fine-tuning of predictive models. 

Random-search parameter optimization tends to be extremely sample-inefficient. This library attempts to address this sample efficiency issue by intelligently moving through parameter spaces to discover (locally or globally) optimimal solutions. Noteworthy improvements over existing Python implementations are the ability to pass a dictionary as input to the objective function, as well as a structured dictionary as the final output. The algorithm also handles categorical (Boolean and multilabel) data in a manner similar to Hyperopt (https://github.com/hyperopt/hyperopt).

## Installation
The recommended installation process makes use of `pip` (or `pip3`):
```
pip install evolutionary-algorithm
```

## A minimal example
An ideal use case is passing a list of parameters and parameter bounds directly to a `scikit-learn` model, unpacking the paramters as model arguments within the objective function, and receiving a set of fine-tuned parameters as a result, which can be unpacked directly for downstream usage.
```
from sklearn.ensemble import RandomForestClassifier
from evolutionary_algorithm import EvolutionaryAlgorithm as ea

X, y = make_classification(n_features=2)

parameters = [
  {'name' : 'n_estimators', 'bounds' : [10, 100], 'type' : 'int'}
]

def objective_function(args):
  
  clf = RandomForestClassifier(**args)
  clf.fit(X, y)
  return clf.score(X, y) * -1 # Expects a value to be minimized

model = ea(function=objective_function, parameters=parameters)
model.run()
```

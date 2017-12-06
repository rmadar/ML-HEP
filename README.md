# Machine learning in particle physics

All examples can be run via a web browser using Binder. Do want to try? Just click on [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rmadar/ML-HEP/master?filepath=examples)

## Data inspection

This [notebook](https://github.com/rmadar/ML-HEP/blob/master/examples/1-DatasetExploration.ipynb) explains how to quickly explore distributions for a given dataset with simple selections. It also shows how to group events using a given criteria (such as the number of leptons) in order to inspect the obtained sub-datasets separately in a very easy way.

## Kinematics reconstruction using regression

This [notebook](https://github.com/rmadar/ML-HEP/blob/master/examples/2-Regression.ipynb) shows how to perform the reconstruction of a sophisticated kinematics using machine learning regression. The performance of three algorithm are tested and the separation power between signal and backround is tried on using ROC curves.


## Signal and background classification

This [notebook](https://github.com/rmadar/ML-HEP/blob/master/examples/3-Classification.ipynb) illustrates how an event classification can be performed using machine learning algorithms. The datasets is prepared and three algorithms are compared. The final classifier distributions are displayed and the ROC curves are studied (and compare with ROC curves based on a single variable). The power of the classification can be easily assessed for different selections.

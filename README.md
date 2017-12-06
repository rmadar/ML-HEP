# Machine learning in particle physics

This repository gives few examples of data analysis in the context of particle physics, more specifically on four top quarks production contaminated by top quark pairs produced in association with an electro-weak vector boson. If you have no idea of what I am talking about, [check this out](http://romain-madar.com/research.html)! 

All examples of this page use simulated LHC collisions stored in a csv format, converted from a ROOT file using this [four lines script](https://github.com/rmadar/ML-HEP/blob/master/examples/ConvertRootCsv.py). All the notebooks can be interactively executed simply via a web browser. Do you want to try? Just click on [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rmadar/ML-HEP/master?filepath=examples)

## 1. Data inspection

This [notebook](https://github.com/rmadar/ML-HEP/blob/master/examples/1-DatasetExploration.ipynb) explains how to quickly explore distributions for a given dataset with simple selections. It also shows how to group collisions based on a given criteria (such as the number of leptons) in order to inspect the each sub-datasets separately in a very easy way.

## 2. Kinematics reconstruction using regression

This [notebook](https://github.com/rmadar/ML-HEP/blob/master/examples/2-Regression.ipynb) shows how to perform the reconstruction of a complex kinematic observable, for instance invariant mass of the four top quarks system, using machine learning regression. The performance of three algorithms are tested and the separation power of the reconstructed mass between signal and background is probed using ROC curves.


## 3. Signal and background classification

This [notebook](https://github.com/rmadar/ML-HEP/blob/master/examples/3-Classification.ipynb) illustrates how event categorisation (signal v.s. backgroun) can be performed using machine learning algorithms. The datasets is prepared and three algorithms are compared. The classifier distributions are displayed and the ROC curves are compared classification based on a single variable). The power of the classification can be easily assessed for different selections.

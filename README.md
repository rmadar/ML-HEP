# Machine learning in particle physics

This repository gives few example of data analysis in the context of particle physics, more specifically on four top quark production contaminated by top quark pairs produced in association with an electro-weak vector boson. If you have no idea of I am talking about and if you want to know more, then [check this out](http://romain-madar.com/research.html)! All examples can be interactively executed via a web browser using Binder. Do want to try? Just click on [![Binder](https://mybinder.org/badge.svg)](https://mybinder.org/v2/gh/rmadar/ML-HEP/master?filepath=examples)

## 1. Data inspection

This [notebook](https://github.com/rmadar/ML-HEP/blob/master/examples/1-DatasetExploration.ipynb) explains how to quickly explore distributions for a given dataset with simple selections. It also shows how to group events using a given criteria (such as the number of leptons) in order to inspect the obtained sub-datasets separately in a very easy way.

## 2. Kinematics reconstruction using regression

This [notebook](https://github.com/rmadar/ML-HEP/blob/master/examples/2-Regression.ipynb) shows how to perform the reconstruction of a sophisticated kinematics using machine learning regression. The performance of three algorithm are tested and the separation power between signal and backround is tried on using ROC curves.


## 3. Signal and background classification

This [notebook](https://github.com/rmadar/ML-HEP/blob/master/examples/3-Classification.ipynb) illustrates how an event classification can be performed using machine learning algorithms. The datasets is prepared and three algorithms are compared. The final classifier distributions are displayed and the ROC curves are studied (and compare with ROC curves based on a single variable). The power of the classification can be easily assessed for different selections.

r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers

part2_q1 = r"""
**Your answer:**
On a first hand, when we take a little k (in particular when k=1) we closely fit our dataset. It could probably lead to overfitting and to a big generalization gap.
On the other hand, when k is very high we risk underfitting: at the extreme case, when k=Training_size, our predictor is a simple mean on the training set. It's obviously a very simple and bad model, giving the same predict for each input.
Thanks to these observations we can now answer the question: increasing k allows to increase the generalization capacity of our model, but taking a too big k can vanish our model and lead to bad results.
In practice we get a sweet pot around k = 3-5 (in our implementation we sometimes get 1).


"""

part2_q2 = r"""
**Your answer:**
1. This case is problematic because it exposes us to overfitting: one picking a model giving great results on the train set could pick a very complex model performing very well on the training set (by completely fitting it) but with a poor generalization capacity.
2. This method is satisfying in term of overfitting and does not present the problem evocated in 1. .
The issue with this method is that we do not exploit all our dataset (we dedicate a part of it to be only use for test purpose) to train our model as much as we can.

The k-folder method presents a good trade-off between these two possibilities.

Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q2 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

part3_q3 = r"""
**Your answer:**


Write your answer using **markdown** and $\LaTeX$:
```python
# A code block
a = 2
```
An equation: $e^{i\pi} -1 = 0$

"""

# ==============

# ==============
# Part 4 answers

part4_q1 = r"""
In our case an ideal pattern is an horizontal line on 0. Such a pattern indicate that the correct classification and our prediction are equals.
One can observe that the instances are closer to the line 0 for the trained model after CV than for the top 5-features model. We can also easily check that the trained model adter CV is better by comparising the MSE:
18.1 for the trained model vs 27.2 for the top 5-features model.

"""

part4_q2 = r"""
1. Using logspace we both make less calculations and save memory: we scan less possible for lambda that in using linspace. 
2. We scan every possible combination. There are 20 lambda values (thanks to logspace) and 3 possible degrees.
We totally scan around 60 possibilities.


"""
# ==============

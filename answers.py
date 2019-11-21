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
"""

# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
The value of $\Delta$ was arbitrary since the value of $\Delta$ doesn't matter as much as it matters that it exists.
The goal of having $\Delta$ was to make sure there was some margin that distances the score of the correct class from the other classes. Increasing or decreasing $\Delta$ scales the weights matrix while preserving the ratio between the weights, which is what really matters to the convergence. Therefore, the choice of $\Delta$ can be an arbitrary positive number.
"""

part3_q2 = r"""
Typically, each image in the visualization section corresponds to a digit. Every image would show us what, according to what the model learned, every digit should look like.
That's why, the errors in classification are explained by the fact that if a digit is not clear or is rotated then
the scores will be different and we will get the wrong classification.

This is similar to kNN in the sense that both models, at some point, use L2 distance between the image we want to predict and the training set. In kNN, we calculate the distance of the representative vectors and pick the nearest, while in this model we calculate the distance during training and predict by using the weights matrix.
"""

part3_q3 = r"""
We believe that the learning rate is good, since the loss converges neatly to a very low value. On the other hand, it converged after a very small number of epochs, then it continued to decrease way slower, so the learning rate might have been a bit too high. Typically, we would prefer a steady convergence so that we don't risk stepping over the minimum. If we picked a lower learning rate, the convergence would be a bit slower. A higher learning rate would, as said, have the risk of stepping over the minimum.

The model is slightly overfitted to the training set. According to the accuracy graph, the training set accuracy becomes higher than the validation set accuracy, and it can be seen in the example images that a slight change in the number's rotation would imply a mislabeled image. But since the accuracy keeps improving along the epochs for both sets, the model is only slightly overfitted.
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

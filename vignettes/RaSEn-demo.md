---
output:
  pdf_document: default
  html_document: default
bibliography: reference.bib
---
<a id="top"></a>

---
title: "A demonstration of the RaSEn package"
author: "Ye Tian and Yang Feng"
date: "2020-05-27"
output: rmarkdown::html_vignette
vignette: >
  %\VignetteIndexEntry{RaSEn demo}
  %\VignetteEngine{knitr::rmarkdown}
  %\VignetteEncoding{UTF-8}
---

We provide a detailed demo of the usage for the \verb+RaSEn+ package. 

* [Introduction](#intro)

* [Installation](#install)

* [How to Fit a RaSE Classifier for Prediction](#rase)

* [How to Use RaSE for Feature Ranking](#fk)



## Introduction{#intro}
Suppose we have training data $\{\mathbf{x}_i, y_i\}_{i=1}^n \in \{\mathbb{R}^p, \{0, 1\}\}$, where each $\mathbf{x}_i$ is a $1 \times p$ vector.

Based on training data, RaSE algorithm aims to generate $B_1$ weak learners $\{C_n^{S_j}\}_{j=1}^{B_1}$, each of which is constructed in a feature subspace $S_j \subseteq \{1, ..., p\}$ instead using all $p$ features. To obtain each weak learner, $B_2$ candidates $\{C_n^{S_{jk}}\}_{k=1}^{B_2}$ are trained based in subspaces $\{S_{jk}\}_{k=1}^{B_2}$, respectively. To choose the optimal one among these $B_2$ candidates, some criteria need to be applied, including minimizing ratio information criterion (RIC, @ye2020rase), minimizing extended Bayes information criterion (eBIC, @chen2008extended, @chen2012extended), minimizing the training error, minimizing the validation error (if validation data is available), minimizing the cross-validation error, minimizing leave-one-out error etc. And the type of weak learner can be quite flexible.

To better adapt RaSE into the sparse setting, we can update the distribution of random feature subspaces according to the frequencies of features in $B_1$ subspaces in each round. This can be seen as an adaptive strategy to increase the possibility to cover the signals that contribute to our model, which can improve the performance of RaSE classifiers in sparse settings.

The frequencies of $p$ features in $B_1$ subspaces can be used for feature ranking as well. And we could plot the frequencies to intuitively rank the importance of each feature in a RaSE model.



## Installation{#install}
`RaSEn` can be installed from CRAN. 

```r
install.packages("RaSEn", repos = "http://cran.us.r-project.org")
```
Then we can load the package:

```r
library(RaSEn)
```


<a id="rase"></a>

## How to Fit a RaSE Classifier for Prediction{#rase}
We will show in this section how to fit RaSE classifiers based on different types of base classifiers. First we generate the data from a binary guanssian mixture model (Model 1 in @ye2020rase)
$$
  \mathbf{x} \sim (1-y)N(\mathbf{\mu}^{(0)}, \Sigma) + yN(\mathbf{\mu}^{(1)}, \Sigma),
$$
where $\mathbf{\mu}^{(0)}, \mathbf{\mu}^{(1)}$ are both $1 \times p$ vectors, $\Sigma$ is a $p \times p$ symmetric positive definite matrix. Here $y$ follows a bernoulli distribution:
$$
  y \sim Bernoulli(\pi_1),
$$
where $\pi_1 \in (0,1)$ and we denote $\pi_0 = 1-\pi_1$.

Here we follow from the setting of @mai2012direct, letting $\Sigma  = (0.5^{|i-j|})_{p \times p} , \mathbf{\mu}^{(0)} = \mathbf{0}_{p \times 1}, \mathbf{\mu}^{(1)} = \Sigma^{-1}\times 0.556(3, 1.5, 0, 0, 2, \mathbf{0}_{1 \times p-5})^T$. Let $n = 100, p =50$. According to the definition of minimal discriminative set in Ye Tian and Yang Feng (2020), here the minimal discriminative set $S^* = \{1, 2, 5\}$, which contribute to the classification.

Apply function `RaModel` to generate training data and test data of size 100 with dimension 50.

```r
set.seed(0, kind = "L'Ecuyer-CMRG")
train.data <- RaModel(1, n = 100, p = 50)
test.data <- RaModel(1, n = 100, p = 50)
xtrain <- train.data$x
ytrain <- train.data$y
xtest <- test.data$x
ytest <- test.data$y
```
We can visualize the first two dimensions or feature 1 and 2 as belows:

```r
library(ggplot2)
ggplot(data = data.frame(xtrain, y = factor(ytrain)), mapping = aes(x = X1, 
    y = X2, color = y)) + geom_point()
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png)
Similarly, we can also visualize the feature 6 and 7:

```r
ggplot(data = data.frame(xtrain, y = factor(ytrain)), mapping = aes(x = X6, y = X7, 
    color = y)) + geom_point()
```

![plot of chunk unnamed-chunk-6](figure/unnamed-chunk-6-1.png)
It's obvious to see that in dimension 1 and 2 the data from two classes are more linearly seperate than in dimension 6 and 7. Then we call `Rase` function to fit the RaSE classifier with LDA, QDA and logistic regression base classifiers with criterion of minimizing RIC and RaSE classifier with knn base classifier with criterion of minimizing leave-one-out error. To use different types of base classifier, we set `base` as "lda", "qda", "knn" and "logistic", repectively. `B1` is set to be 100 to generate 100 weak learners and `B2` is set to be 100 as well to generate 100 subspace candidates for each weak learner. Without using iterations, we set `iteration` as 0. `cutoff` is set to be TRUE to apply the empirical optimal threshold for ensemble classification. `criterion` is set to be "ric" for RaSE classifier with LDA, QDA and logistic regression while it is "loo" for RaSE classifier with knn base classifier. Since we want to do feature ranking, we set `ranking` as TRUE to get the frequencies of features in $B_1$ subspaces. To speed up the computation, we apply parallel computing with 2 cores by setting `cores = 2`.

```r
fit.lda <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, cutoff = TRUE, 
    subset.size.max = NULL, base = "lda", cores = 2, criterion = "ric", 
    ranking = TRUE)
fit.qda <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, cutoff = TRUE, 
    subset.size.max = NULL, base = "qda", cores = 2, criterion = "ric", 
    ranking = TRUE)
fit.knn <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, cutoff = TRUE, 
    subset.size.max = NULL, base = "knn", cores = 2, criterion = "loo", 
    ranking = TRUE)
fit.logistic <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, 
    cutoff = TRUE, subset.size.max = NULL, base = "logistic", cores = 2, 
    criterion = "ric", ranking = TRUE)
```
To evaluate the performance of four different models, we calculate the test error on test data:

```r
er.lda <- mean(predict(fit.lda, xtest) != ytest)
er.qda <- mean(predict(fit.qda, xtest) != ytest)
er.knn <- mean(predict(fit.knn, xtest) != ytest)
er.logistic <- mean(predict(fit.logistic, xtest) != ytest)
cat("LDA:", er.lda, "QDA:", er.qda, "knn:", er.knn, "logistic:", er.logistic)
```

```
## LDA: 0.03 QDA: 0.07 knn: 0.05 logistic: 0.05
```
And the output of `Rase` function is an object belonging to S3 class "RaSE". It contains:

* marginal: the marginal probability for each class.

* fit.list: a list of B1 fitted base classifiers.

* B1: the number of weak learners.

* B2: the number of subspace candidates generated for each weak learner.

* base: the type of base classifier.

* cutoff: the empirically optimal threshold.

* subspace: a list of subspaces correponding to B1 weak learners.

* ranking: the frequency of each feature in B1 subspaces.


## How to Use RaSE for Variable Selection{#fk}
The frequencies of features in $B_1$ subspaces for four RaSE classifiers are contained in the output, which can be used for feature ranking. We can plot them by using `ggplot` function:

```r
library(gridExtra)
plot_lda <- ggplot(data = data.frame(frequency = fit.lda$ranking, feature = 1:50), 
    mapping = aes(y = frequency, x = feature)) + geom_point() + ggtitle(expression("RaSE-LDA")) + 
    theme(plot.title = element_text(hjust = 0.5))
plot_qda <- ggplot(data = data.frame(frequency = fit.qda$ranking, feature = 1:50), 
    mapping = aes(y = frequency, x = feature)) + geom_point() + ggtitle(expression("RaSE-QDA")) + 
    theme(plot.title = element_text(hjust = 0.5))
plot_knn <- ggplot(data = data.frame(frequency = fit.knn$ranking, feature = 1:50), 
    mapping = aes(y = frequency, x = feature)) + geom_point() + ggtitle(expression("RaSE-kNN")) + 
    theme(plot.title = element_text(hjust = 0.5))
plot_logistic <- ggplot(data = data.frame(frequency = fit.logistic$ranking, 
    feature = 1:50), mapping = aes(y = frequency, x = feature)) + geom_point() + 
    ggtitle(expression("RaSE-logistic")) + theme(plot.title = element_text(hjust = 0.5))

grid.arrange(plot_lda, plot_qda, plot_knn, plot_logistic, ncol = 2)
```

![plot of chunk unnamed-chunk-9](figure/unnamed-chunk-9-1.png)
From four figures, it can be noticed that feature 1, 2 and 5 obtain the highest frequencies among all $p = 50$ features when the base classifier is taken as LDA, QDA and $k$NN, implying their importance in classification model. We can set a positive iteration number to make their frequencies of appearing in B1 subspaces increase, which may improve the performance.

## Reference





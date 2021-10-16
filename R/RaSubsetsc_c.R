RaSubsetsc_c <- function(xtrain, ytrain, xval, yval, B2, S, model, k, criterion, cv, t0.mle = NULL, t1.mle = NULL, mu0.mle = NULL,  mu1.mle = NULL, Sigma.mle = NULL, Sigma0.mle = NULL, Sigma1.mle = NULL, gam = NULL, kl.k = kl.k,...) {
  list2env(list(...), environment())
  n <- length(ytrain)
  p <- ncol(xtrain)
  p0 <- sum(ytrain == 0)/length(ytrain)
  p1 <- sum(ytrain == 1)/length(ytrain)
  n0 <- sum(ytrain == 0)
  n1 <- sum(ytrain == 1)

  if (model == "logistic") {
    if (criterion == "nric") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        -2*(p0*KL.divergence(xtrain[ytrain == 0, Si, drop = F], xtrain[ytrain == 1, Si, drop = F], k = kl.k[1])[kl.k[1]] + p1*KL.divergence(xtrain[ytrain == 1, Si, drop = F], xtrain[ytrain == 0, Si, drop = F], k = kl.k[2])[kl.k[2]]) + length(Si)*log(log(n))/sqrt(n)
      })
    }

    if (criterion == "ric") {
      subspace.list <- sapply(1:B2, function(i) {
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        score <- predict(glm(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), family = "binomial"), data.frame(x = xtrain.r))
        posterior0 <- 1/(1 + exp(score))
        posterior1 <- 1 - posterior0
        ric("other", xtrain, ytrain, Si, p0 = p0, p1 = p1, posterior0 = posterior0, posterior1 = posterior1, deg = function(i) {
          i
        })
      })
    }

    if (criterion == "ebic") {
      subspace.list <- sapply(1:B2, function(i) {
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        calc_ebic(xtrain, ytrain, Si, gam)
      })
    }

    if (criterion == "bic") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        calc_ebic(xtrain, ytrain, Si, gam = 0)
      })
    }

    if (criterion == "aic") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        calc_aic(xtrain, ytrain, Si)
      })
    }

    if (criterion == "training") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        mean(as.numeric(I(predict(glm(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), family = "binomial"), data.frame(x = xtrain.r)) >
                            0)) != ytrain, na.rm = TRUE)
      })
    }

    if (criterion == "validation") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        xval.r <- xval[, Si, drop = F]
        mean(as.numeric(I(predict(glm(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), family = "binomial"), data.frame(x = xval.r)) >
                            0)) != yval, na.rm = TRUE)
      })
    }

    if (criterion == "cv") {
      folds <- createFolds(ytrain, k = cv)
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        mean(sapply(1:cv, function(j) {
          fit <- glm(y ~ ., data = data.frame(x = xtrain[-folds[[j]], Si, drop = F], y = ytrain[-folds[[j]]]), family = "binomial")
          mean(as.numeric(I(predict(fit, data.frame(x = xtrain[folds[[j]], Si, drop = F]))) > 0) != ytrain[folds[[j]]], na.rm = TRUE)
        }))
      })
    }

    if (criterion == "loo") {
      stop("minimizing leave-one-out error is not available when model = \"logistic\", please choose other criterion")
    }

    i0 <- which.min(subspace.list)
    S <- S[!is.na(S[, i0]), i0]  # final optimal subspace


  }

  if (model == "svm") {
    if (criterion == "nric") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        -2*(p0*KL.divergence(xtrain[ytrain == 0, Si, drop = F], xtrain[ytrain == 1, Si, drop = F], k = kl.k)[kl.k] + p1*KL.divergence(xtrain[ytrain == 1, Si, drop = F], xtrain[ytrain == 0, Si, drop = F], k = kl.k)[kl.k]) +  + length(Si)*log(log(n))/sqrt(n)
      })
    }

    if (!is.character(kernel)) {
      kernel <- "linear"
    }

    if (criterion == "training") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        mean(as.numeric(predict(svm(x = xtrain.r, y = ytrain, kernel = kernel, type = "C-classification"), xtrain.r)) - 1 !=
               ytrain, na.rm = TRUE)
      })
    }

    if (criterion == "validation") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        xval.r <- xval[, Si, drop = F]
        mean(as.numeric(predict(svm(x = xtrain.r, y = ytrain, kernel = kernel, type = "C-classification"), xval.r)) - 1 !=
               yval, na.rm = TRUE)
      })
    }

    if (criterion == "cv") {
      folds <- createFolds(ytrain, k = cv)
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        mean(sapply(1:cv, function(j) {
          fit <- svm(x = xtrain[-folds[[j]], Si, drop = F], y = ytrain[-folds[[j]]], kernel = kernel, type = "C-classification")
          mean(as.numeric(predict(fit, xtrain[folds[[j]], Si, drop = F])) - 1 != ytrain[folds[[j]]], na.rm = TRUE)
        }))
      })
    }

    if (criterion == "ebic") {
      stop("minimizing eBIC is not available when model = \"svm\", please choose other criterion")
    }

    if (criterion == "ric") {
      stop("minimizing RIC is not available when model = \"svm\", please choose other criterion")
    }

    if (criterion == "loo") {
      stop("minimizing leave-one-out error is not available when model = \"svm\", please choose other criterion")
    }

    i0 <- which.min(subspace.list)
    S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
  }


  if (model == "randomforest") {
    if (criterion == "training") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        mean(as.numeric(predict(randomForest(x = xtrain.r, y = factor(ytrain)), xtrain.r)) - 1 != factor(ytrain), na.rm = TRUE)
      })
    }

    if (criterion == "validation") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        xval.r <- xval[, Si, drop = F]
        mean(as.numeric(predict(randomForest(x = xtrain.r, y = factor(ytrain)), xval.r)) - 1 != factor(yval), na.rm = TRUE)
      })
    }

    if (criterion == "cv") {
      folds <- createFolds(ytrain, k = cv)
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        mean(sapply(1:cv, function(j) {
          # fit <- randomForest(x = xtrain[-folds[[j]], Si, drop = F], y = factor(ytrain[-folds[[j]]]))
          # mean((as.numeric(predict(fit, xtrain[folds[[j]], Si, drop = F])) - 1) != ytrain[folds[[j]]], na.rm = TRUE)
          fit <- ranger(y ~ ., data = data.frame(x = xtrain[-folds[[j]], Si, drop = F], y = factor(ytrain[-folds[[j]]])), ...)
          mean((as.numeric(predict(fit, data = data.frame(x = xtrain[folds[[j]], Si, drop = F]))$predictions) - 1) != ytrain[folds[[j]]], na.rm = TRUE)
        }))
      })
    }

    if (criterion == "ebic") {
      stop("minimizing eBIC is not available when model = \"randomforest\", please choose other criterion")
    }

    if (criterion == "ric") {
      stop("minimizing RIC is not available when model = \"randomforest\", please choose other criterion")
    }

    if (criterion == "loo") {
      stop("minimizing leave-one-out error is not available when model = \"randomforest\", please choose other criterion")
    }

    i0 <- which.min(subspace.list)
    S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
  }

  if (model == "knn") {
    if (criterion == "loo") {
      subspace.list <- sapply(1:B2, function(i) {
        d <- length(S[, i][!is.na(S[, i])])  # subspace size
        Si <- matrix(S[, i][!is.na(S[, i])], nrow = d)  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        knn.test <- sapply(k, function(j) {
          mean(knn.cv(xtrain.r, ytrain, j, use.all = FALSE) != ytrain, na.rm = TRUE)
        })
        min(knn.test)
      })

      i0 <- which.min(subspace.list)
      S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
    }

    if (criterion == "validation") {
      subspace.list <- sapply(1:B2, function(i) {
        d <- length(S[, i][!is.na(S[, i])])  # subspace size
        Si <- matrix(S[, i][!is.na(S[, i])], nrow = d)  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        xval.r <- xval[, Si, drop = F]

        knn.test <- sapply(k, function(j) {
          fit <- knn3(x = xtrain.r, y = factor(ytrain), k = j, use.all = FALSE)
          mean(predict(fit, xval.r, type = "class") != yval)
        })
        min(knn.test)
      })

      i0 <- which.min(subspace.list)
      S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
    }

    if (criterion == "cv") {
      folds <- createFolds(ytrain, cv)
      subspace.list <- sapply(1:B2, function(i) {
        Si <- S[!is.na(S[, i]), i]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        knn.test <- sapply(k, function(r) {
          sapply(1:cv, function(j) {
          fit <- knn3(x = xtrain.r[-folds[[j]], ,drop = F], y = factor(ytrain)[-folds[[j]]], k = r, use.all = FALSE)
          mean(predict(fit, xtrain.r[folds[[j]], ,drop = F], type = "class") != ytrain[folds[[j]]])
          })
        })
        mean(knn.test)
      })

      i0 <- which.min(subspace.list)
      S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
    }

    if (criterion == "training") {
      stop("minimizing training error is not available when model = \"knn\", please choose other criterion")
    }

    if (criterion == "ebic") {
      stop("minimizing eBIC is not available when model = \"knn\", please choose other criterion")
    }

    if (criterion == "ric") {
      stop("minimizing RIC is not available when model = \"knn\", please choose other criterion")
    }

  }

  if (model == "tree") {

    if (criterion == "training") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        fit <- rpart(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), method = "class")
        mean((as.numeric(predict(fit, data.frame(x = xtrain.r), type = "class")) - 1) != ytrain, na.rm = TRUE)
      })
    }

    if (criterion == "validation") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        xval.r <- xval[, Si, drop = F]
        fit <- rpart(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), method = "class")
        mean((as.numeric(predict(fit, data.frame(x = xval.r), type = "class")) - 1) != yval, na.rm = TRUE)
      })
    }

    if (criterion == "cv") {
      folds <- createFolds(ytrain, k = cv)
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        mean(sapply(1:cv, function(j) {
          fit <- rpart(y ~ ., data = data.frame(x = xtrain[-folds[[j]], Si, drop = F], y = ytrain[-folds[[j]]]), method = "class")
          mean((as.numeric(predict(fit, data.frame(x = xtrain[folds[[j]], Si, drop = F]), type = "class")) - 1) != ytrain[folds[[j]]],
               na.rm = TRUE)
        }))
      })
    }

    if (criterion == "ric") {
      stop("minimizing RIC is not available when model = \"tree\", please choose other criterion")
    }

    if (criterion == "ebic") {
      stop("minimizing eBIC is not available when model = \"tree\", please choose other criterion")
    }

    if (criterion == "loo") {
      stop("minimizing leave-one-out error is not available when model = \"tree\", please choose other criterion")
    }

    i0 <- which.min(subspace.list)
    S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
  }

  if (model == "lda") {
    if (criterion == "nric") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        # sum(sapply(1:nrow(L), function(j){-2*pb[L[j, 1]+1]*KL.divergence(xtrain[ytrain == L[j, 1], Si, drop = F], xtrain[ytrain == L[j, 2], Si, drop = F], k = kl.k[L[j, 1]+1])[kl.k[L[j, 1]+1]]})) + length(Si)*log(log(n))/sqrt(n)
        -2*(p0*KL.divergence(xtrain[ytrain == 0, Si, drop = F], xtrain[ytrain == 1, Si, drop = F], k = kl.k[1])[kl.k[1]] + p1*KL.divergence(xtrain[ytrain == 1, Si, drop = F], xtrain[ytrain == 0, Si, drop = F], k = kl.k[2])[kl.k[2]]) + length(Si)*log(log(n))/sqrt(n)
      })
    }

    if (criterion == "ric") {
      if (!is.null(Sigma.mle)) {
        subspace.list <- sapply(1:B2, function(i) {
          # the last row is training error for each i in 1:B2
          Si <- S[, i][!is.na(S[, i])]  # current subspace
          ric("lda", xtrain, ytrain, Si, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma.mle = Sigma.mle[Si, Si, drop = F])
        })
      } else {
        subspace.list <- sapply(1:B2, function(i) {
          # the last row is training error for each i in 1:B2
          Si <- S[, i][!is.na(S[, i])]  # current subspace
          Sigma.mle <- (as.matrix(cov(as.matrix(xtrain[ytrain == 0, Si, drop = F])))*(n0-1)+ as.matrix(cov(as.matrix(xtrain[ytrain == 1, Si, drop = F])))*(n1-1))/n

          ric("lda", xtrain, ytrain, Si, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma.mle = Sigma.mle)
        })
      }
    }

    if (criterion == "validation") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        xval.r <- xval[, Si, drop = F]
        mean(predict(lda(x = xtrain.r, grouping = ytrain), xval.r)$class != yval, na.rm = TRUE)
      })
    }


    if (criterion == "training") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        mean(predict(lda(x = xtrain.r, grouping = ytrain), xtrain.r)$class != ytrain, na.rm = TRUE)
      })
    }

    if (criterion == "cv") {
      folds <- createFolds(ytrain, k = cv)
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        mean(sapply(1:cv, function(j) {
          mean(predict(lda(x = xtrain[-folds[[j]], Si, drop = F], grouping = ytrain[-folds[[j]]]), xtrain[folds[[j]], Si, drop = F])$class !=
                 ytrain[folds[[j]]], na.rm = TRUE)
        }))
      })
    }

    if (criterion == "loo") {
      stop("minimizing leave-one-out error is not available when model = \"lda\", please choose other criterion")
    }

    i0 <- which.min(subspace.list)
    S <- S[!is.na(S[, i0]), i0]  # final optimal subspace


  }

  if (model == "qda") {
    if (criterion == "nric") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        -2*(p0*KL.divergence(xtrain[ytrain == 0, Si, drop = F], xtrain[ytrain == 1, Si, drop = F], k = kl.k[1])[kl.k[1]] + p1*KL.divergence(xtrain[ytrain == 1, Si, drop = F], xtrain[ytrain == 0, Si, drop = F], k = kl.k[2])[kl.k[2]]) + length(Si)*(length(Si) + 3)/2*log(log(n))/sqrt(n)
      })
    }

    if (criterion == "ric") {
      if (!is.null(Sigma0.mle)) {
        subspace.list <- sapply(1:B2, function(i) {
          # print(i) the last row is training error for each i in 1:B2
          Si <- S[, i][!is.na(S[, i])]  # current subspace
          ric("qda", xtrain, ytrain, Si, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma0.mle = Sigma0.mle, Sigma1.mle = Sigma1.mle,
              p0 = p0, p1 = p1)
        })
      } else {
        subspace.list <- sapply(1:B2, function(i) {
          # print(i) the last row is training error for each i in 1:B2
          Si <- S[, i][!is.na(S[, i])]  # current subspace
          Sigma0.mle <- as.matrix(cov(xtrain[ytrain == 0, Si, drop = F]))*(n0-1)/n1
          Sigma1.mle <- as.matrix(cov(xtrain[ytrain == 1, Si, drop = F]))*(n0-1)/n1
          ric("qda", xtrain, ytrain, Si, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma0.mle = Sigma0.mle, Sigma1.mle = Sigma1.mle,
              p0 = p0, p1 = p1)
        })
      }
    }

    if (criterion == "validation") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        xval.r <- xval[, Si, drop = F]
        mean(predict(qda(x = xtrain.r, grouping = ytrain), xval.r)$class != yval, na.rm = TRUE)
      })
    }

    if (criterion == "training") {
      subspace.list <- sapply(1:B2, function(i) {
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        mean(predict(qda(x = xtrain.r, grouping = ytrain), xtrain.r)$class != ytrain, na.rm = TRUE)
      })
    }

    if (criterion == "cv") {
      folds <- createFolds(ytrain, k = cv)
      subspace.list <- sapply(1:B2, function(i) {
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        mean(sapply(1:cv, function(j) {
          mean(predict(qda(x = xtrain[-folds[[j]], Si, drop = F], grouping = ytrain[-folds[[j]]]), xtrain[folds[[j]], Si, drop = F])$class !=
                 ytrain[folds[[j]]], na.rm = TRUE)
        }))
      })
    }

    if (criterion == "loo") {
      stop("minimizing leave-one-out error is not available when model = \"qda\", please choose other criterion")
    }

    i0 <- which.min(subspace.list)
    S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
  }

  if (model == "kernelknn") {
    if (criterion == "cv") {
      folds <- createFolds(ytrain, cv)
      ytrain <- ytrain+1
      subspace.list <- sapply(1:B2, function(i) {
        d <- length(S[, i][!is.na(S[, i])])  # subspace size
        Si <- matrix(S[, i][!is.na(S[, i])], nrow = d)  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        knn.test <- sapply(1:cv, function(j) {
          ypred <- KernelKnn(data = xtrain.r[-folds[[j]], , drop = F], TEST_data = xtrain.r[folds[[j]], , drop = F], y = ytrain[-folds[[j]]], k = k, regression = F, Levels = 1:max(ytrain), ...)
          mean(apply(ypred, 1, function(x){which.max(x)}) != ytrain[folds[[j]]])
        })

        mean(knn.test)
      })

      i0 <- which.min(subspace.list)
      S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
    }
  }

    return(list(subset = S))
}

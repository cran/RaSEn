RaSubsetsc_rg <- function(xtrain, ytrain, xval, yval, B2, S, model, k, criterion, cv, t0.mle = NULL, t1.mle = NULL, mu0.mle = NULL,  mu1.mle = NULL, Sigma.mle = NULL, Sigma0.mle = NULL, Sigma1.mle = NULL, gam = NULL, kl.k = kl.k, XX = NULL, XY = NULL, ...) {
  list2env(list(...), environment())
  n <- length(ytrain)
  p <- ncol(xtrain)
  if (model == "lm") {
    D.train <- data.frame(x = xtrain, y = ytrain)
    if (criterion == "mse") {
      subspace.list <- sapply(1:B2, function(i) {
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        sum(ytrain^2) - t(ytrain) %*% xtrain[, Si, drop = F] %*% solve(t(xtrain[, Si, drop = F]) %*% xtrain[, Si, drop = F]) %*% t(xtrain[, Si, drop = F]) %*% ytrain
        # fit <- lm(y ~., data = D.train[, c(Si, ncol(D.train))])
        # mean(residuals(fit)^2)
      })
    } else if (criterion == "bic") {
      subspace.list <- sapply(1:B2, function(i) {
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        sigma.hat2 <- (sum(ytrain^2) - t(XY[Si, ,drop = F]) %*% solve(XX[Si, Si, drop = F]) %*% XY[Si, ,drop = F])/n
        n*log(sigma.hat2) + length(Si)*log(n)
      })
    } else if (criterion == "aic") {
      subspace.list <- sapply(1:B2, function(i) {
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        sigma.hat2 <- (sum(ytrain^2) - t(XY[Si, ,drop = F]) %*% solve(XX[Si, Si, drop = F]) %*% XY[Si, ,drop = F])/n
        n*log(sigma.hat2) + length(Si)*2
      })
    } else if (criterion == "ebic") {
      subspace.list <- sapply(1:B2, function(i) {
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        sigma.hat2 <- (sum(ytrain^2) - t(XY[Si, ,drop = F]) %*% solve(XX[Si, Si, drop = F]) %*% XY[Si, ,drop = F])/n
        # sigma.hat2 <- (sum(ytrain^2) - t(ytrain) %*% xtrain[, Si, drop = F] %*% solve(t(xtrain[, Si, drop = F]) %*% xtrain[, Si, drop = F]) %*% t(xtrain[, Si, drop = F]) %*% ytrain)/n
        n*log(sigma.hat2) + length(Si)*log(n) + 2*gam*length(Si)*log(p)
      })
    }

    i0 <- which.min(subspace.list)
    S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
  }


  if (model == "knn") {
    if (criterion == "cv") {
      folds <- createFolds(ytrain, cv)
      subspace.list <- sapply(1:B2, function(i) {
        d <- length(S[, i][!is.na(S[, i])])  # subspace size
        Si <- matrix(S[, i][!is.na(S[, i])], nrow = d)  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        knn.test <- sapply(1:cv, function(j) {
          fit <- knnreg(x = xtrain.r[-folds[[j]], , drop = F], y = ytrain[-folds[[j]]], k = k, use.all = FALSE)
          mean((predict(fit, xtrain.r[folds[[j]], , drop = F]) - ytrain[folds[[j]]])^2)
          # mean((knn.reg(train = xtrain.r[-folds[[j]], , drop = F], y = ytrain[-folds[[j]]], test = xtrain.r[folds[[j]], , drop = F], k = k)$pred - ytrain[folds[[j]]])^2)
        })
        mean(knn.test)
      })

      i0 <- which.min(subspace.list)
      S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
    }
  }

  if (model == "kernelknn") {
    if (criterion == "cv") {
      folds <- createFolds(ytrain, cv)
      subspace.list <- sapply(1:B2, function(i) {
        d <- length(S[, i][!is.na(S[, i])])  # subspace size
        Si <- matrix(S[, i][!is.na(S[, i])], nrow = d)  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        knn.test <- sapply(1:cv, function(j) {
          ypred <- KernelKnn(data = xtrain.r[-folds[[j]], , drop = F], TEST_data = xtrain.r[folds[[j]], , drop = F], y = ytrain[-folds[[j]]], k = k, regression = T, ...)
          mean((ypred  - ytrain[folds[[j]]])^2)
          # mean((knn.reg(train = xtrain.r[-folds[[j]], , drop = F], y = ytrain[-folds[[j]]], test = xtrain.r[folds[[j]], , drop = F], k = k)$pred - ytrain[folds[[j]]])^2)
        })

        mean(knn.test)
      })

      i0 <- which.min(subspace.list)
      S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
    }
  }

  if (model == "svm") {
    if (!is.character(kernel)) {
      kernel <- "radial"
    }

    if (criterion == "training") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        mean(as.numeric(predict(svm(x = xtrain.r, y = ytrain, kernel = kernel, type = "eps-regression"), xtrain.r)) - 1 !=
               ytrain, na.rm = TRUE)
      })
    }

    if (criterion == "validation") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        xval.r <- xval[, Si, drop = F]
        mean(as.numeric(predict(svm(x = xtrain.r, y = ytrain, kernel = kernel, type = "eps-regression"), xval.r)) - 1 !=
               yval, na.rm = TRUE)
      })
    }

    if (criterion == "cv") {
      folds <- createFolds(ytrain, k = cv)
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        mean(sapply(1:cv, function(j) {
          fit <- svm(x = xtrain[-folds[[j]], Si, drop = F], y = ytrain[-folds[[j]]], kernel = kernel, type = "eps-regression")
          mean((predict(fit, xtrain[folds[[j]], Si, drop = F]) - ytrain[folds[[j]]])^2, na.rm = TRUE)
        }))
      })
    }

    if (criterion == "ebic") {
      stop("minimizing eBIC is not available when model = \"svm\", please choose other criteria")
    }

    if (criterion == "ric") {
      stop("minimizing RIC is not available when model = \"svm\", please choose other criteria")
    }

    if (criterion == "loo") {
      stop("minimizing leave-one-out error is not available when model = \"svm\", please choose other criteria")
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
          # fit <- randomForest(x = xtrain[-folds[[j]], Si, drop = F], y = ytrain[-folds[[j]]], ...)
          # mean((as.numeric(predict(fit, xtrain[folds[[j]], Si, drop = F])) - ytrain[folds[[j]]])^2)
          fit <- ranger(y ~ ., data = data.frame(x = xtrain[-folds[[j]], Si, drop = F], y = ytrain[-folds[[j]]]), ...)
          mean((as.numeric(predict(fit, data = data.frame(x = xtrain[folds[[j]], Si, drop = F]))$predictions) - ytrain[folds[[j]]])^2)
        }))
      })
    }

    if (criterion == "ebic") {
      stop("minimizing eBIC is not available when model = \"randomforest\", please choose other criteria")
    }

    if (criterion == "ric") {
      stop("minimizing RIC is not available when model = \"randomforest\", please choose other criteria")
    }

    if (criterion == "loo") {
      stop("minimizing leave-one-out error is not available when model = \"randomforest\", please choose other criteria")
    }

    i0 <- which.min(subspace.list)
    S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
  }

  if (model == "tree") {

    if (criterion == "training") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        fit <- rpart(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), method = "class")
        mean((as.numeric(predict(fit, data = data.frame(x = xtrain.r), type = "class")) - 1) != ytrain, na.rm = TRUE)
      })
    }

    if (criterion == "validation") {
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        xtrain.r <- xtrain[, Si, drop = F]
        xval.r <- xval[, Si, drop = F]
        fit <- rpart(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), method = "class")
        mean((as.numeric(predict(fit, data = data.frame(x = xval.r), type = "class")) - 1) != yval, na.rm = TRUE)
      })
    }

    if (criterion == "cv") {
      folds <- createFolds(ytrain, k = cv)
      subspace.list <- sapply(1:B2, function(i) {
        # the last row is training error for each i in 1:B2
        Si <- S[, i][!is.na(S[, i])]  # current subspace
        mean(sapply(1:cv, function(j) {
          fit <- rpart(y ~ ., data = data.frame(x = xtrain[-folds[[j]], Si, drop = F], y = ytrain[-folds[[j]]]), method = "anova", ...)
          mean((as.numeric(predict(fit, data = data.frame(x = xtrain[folds[[j]], Si, drop = F]), type = "vector")) - ytrain[folds[[j]]])^2)
        }))
      })
    }

    if (criterion == "ric") {
      stop("minimizing RIC is not available when model = \"tree\", please choose other criteria")
    }

    if (criterion == "ebic") {
      stop("minimizing eBIC is not available when model = \"tree\", please choose other criteria")
    }

    if (criterion == "loo") {
      stop("minimizing leave-one-out error is not available when model = \"tree\", please choose other criteria")
    }

    i0 <- which.min(subspace.list)
    S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
  }

  return(list(subset = S))
}

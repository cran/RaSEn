RaSubset <- function(xtrain, ytrain, xval, yval, B2, S, base, k, criterion, cv, t0.mle = NULL, t1.mle = NULL, mu0.mle = NULL,  mu1.mle = NULL, Sigma.mle = NULL, Sigma0.mle = NULL, Sigma1.mle = NULL, gam = NULL, kl.k = kl.k, ...) {
    list2env(list(...), environment())
    n <- length(ytrain)
    p <- ncol(xtrain)
    p0 <- sum(ytrain == 0)/length(ytrain)
    p1 <- sum(ytrain == 1)/length(ytrain)

    if (base == "gamma") {
        if (criterion == "nric") {
            subspace.list <- sapply(1:B2, function(i) {
                # the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                -2*(p0*KL.divergence(xtrain[ytrain == 0, Si, drop = F], xtrain[ytrain == 1, Si, drop = F], k = kl.k[1])[kl.k[1]] + p1*KL.divergence(xtrain[ytrain == 1, Si, drop = F], xtrain[ytrain == 0, Si, drop = F], k = kl.k[2])[kl.k[2]]) + length(Si)*2*log(log(n))/sqrt(n)
            })
        }
        if (criterion == "ric") {
            subspace.list <- sapply(1:B2, function(i) {
                # the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                ric("gamma", xtrain, ytrain, Si, t0.mle = t0.mle, t1.mle = t1.mle, ...)
            })
        }

        if (criterion == "training") {
            subspace.list <- sapply(1:B2, function(i) {
                # the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                mean(gamma_classifier(t0.mle, t1.mle, p0, p1, newx = xtrain, Si) != ytrain)
            })
        }

        if (criterion == "validation") {
            subspace.list <- sapply(1:B2, function(i) {
                # the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                mean(gamma_classifier(t0.mle, t1.mle, p0, p1, newx = xval, Si) != yval)
            })
        }

        if (criterion == "cv") {
            stop("minimizing cross-validation error is not available when base = \"gamma\", please choose other criteria")
        }

        if (criterion == "ebic") {
            stop("minimizing eBIC is not available when base = \"gamma\", please choose other criteria")
        }

        if (criterion == "loo") {
            stop("minimizing leave-one-out error is not available when base = \"gamma\", please choose other criteria")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
        fit <- list(t0.mle, t1.mle, p0, p1)
        ytrain.pred <- factor(gamma_classifier(t0.mle, t1.mle, p0, p1, newx = xtrain, S))
    }

    if (base == "logistic") {
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


        if (criterion == "aic") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                calc_aic(xtrain, ytrain, Si)
            })
        }

        if (criterion == "ebic") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                # calc_BIC(xtrain, ytrain, Si, D = 0, K = 0, debug = F, gam = gam)
                calc_ebic(xtrain, ytrain, Si, gam)
            })
        }

        if (criterion == "bic") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                calc_ebic(xtrain, ytrain, Si, gam = 0)
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
            stop("minimizing leave-one-out error is not available when base = \"logistic\", please choose other criteria")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        fit <- glm(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), family = "binomial")
        ytrain.pred <- as.numeric(I(predict(fit, data.frame(x = xtrain.r)) > 0))
    }

    if (base == "svm") {
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
            stop("minimizing eBIC is not available when base = \"svm\", please choose other criteria")
        }

        if (criterion == "ric") {
            stop("minimizing RIC is not available when base = \"svm\", please choose other criteria")
        }

        if (criterion == "loo") {
            stop("minimizing leave-one-out error is not available when base = \"svm\", please choose other criteria")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        fit <- svm(x = xtrain.r, y = ytrain, kernel = kernel, type = "C-classification")
        ytrain.pred <- as.numeric(predict(fit, xtrain.r)) - 1
    }


    if (base == "randomforest") {
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
                  fit <- randomForest(x = xtrain[-folds[[j]], Si, drop = F], y = factor(ytrain[-folds[[j]]]))
                  mean((as.numeric(predict(fit, xtrain[folds[[j]], Si, drop = F])) - 1) != ytrain[folds[[j]]], na.rm = TRUE)
                }))
            })
        }

        if (criterion == "ebic") {
            stop("minimizing eBIC is not available when base = \"randomforest\", please choose other criteria")
        }

        if (criterion == "ric") {
            stop("minimizing RIC is not available when base = \"randomforest\", please choose other criteria")
        }

        if (criterion == "loo") {
            stop("minimizing leave-one-out error is not available when base = \"randomforest\", please choose other criteria")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        fit <- randomForest(x = xtrain.r, y = factor(ytrain), ...)
        ytrain.pred <- as.numeric(predict(fit, xtrain.r)) - 1
    }

    if (base == "knn") {
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

            xtrain.r <- xtrain[, S, drop = F]

            knn.test <- sapply(k, function(j) {
                mean(knn.cv(xtrain.r, ytrain, j, use.all = FALSE) != ytrain, na.rm = TRUE)
            })
            k.op <- k[which.min(knn.test)]
            fit <- knn3(x = xtrain.r, y = factor(ytrain), k = k.op, use.all = FALSE)
            ytrain.pred <- predict(fit, xtrain.r, type = "class")
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

            xtrain.r <- xtrain[, S, drop = F]
            xval.r <- xval[, S, drop = F]
            knn.test <- sapply(k, function(j) {
                fit <- knn3(x = xtrain.r, y = factor(ytrain), k = j, use.all = FALSE)
                mean(as.numeric(predict(fit, xval.r, type = "class")) - 1 != yval)
            })
            k.op <- k[which.min(knn.test)]
            fit <- knn3(x = xtrain.r, y = factor(ytrain), k = k.op, use.all = FALSE)
            ytrain.pred <- predict(fit, xtrain.r, type = "class")
        }

        if (criterion == "cv") {
            folds <- createFolds(ytrain, k = cv)
            subspace.list <- sapply(1:B2, function(i) {
                # the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                knn.test <- sapply(k, function(l) {
                    mean(sapply(1:cv, function(j) {
                        mean(predict(knn3(x = xtrain[-folds[[j]], Si, drop = F], y = factor(ytrain[-folds[[j]]]), k = l, use.all = FALSE), xtrain[folds[[j]], Si, drop = F], type = "class") != ytrain[folds[[j]]], na.rm = TRUE)
                    }))
                })
                min(knn.test)
            })

            i0 <- which.min(subspace.list)
            S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

            xtrain.r <- xtrain[, S, drop = F]

            knn.test <- sapply(k, function(j) {
                mean(knn.cv(xtrain.r, ytrain, j, use.all = FALSE) != ytrain, na.rm = TRUE)
            })
            k.op <- k[which.min(knn.test)]
            fit <- knn3(x = xtrain.r, y = factor(ytrain), k = k.op, use.all = FALSE)
            ytrain.pred <- predict(fit, xtrain.r, type = "class")
        }

        if (criterion == "training") {
            stop("minimizing training error is not available when base = \"knn\", please choose other criteria")
        }

        if (criterion == "ebic") {
            stop("minimizing eBIC is not available when base = \"knn\", please choose other criteria")
        }

        if (criterion == "ric") {
            stop("minimizing RIC is not available when base = \"knn\", please choose other criteria")
        }

    }

    if (base == "tree") {

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
                  fit <- rpart(y ~ ., data = data.frame(x = xtrain[-folds[[j]], Si, drop = F], y = ytrain[-folds[[j]]]), method = "class")
                  mean((as.numeric(predict(fit, data = data.frame(x = xtrain[folds[[j]], Si, drop = F]), type = "class")) - 1) != ytrain[folds[[j]]],
                    na.rm = TRUE)
                }))
            })
        }

        if (criterion == "ric") {
            stop("minimizing RIC is not available when base = \"tree\", please choose other criteria")
        }

        if (criterion == "ebic") {
            stop("minimizing eBIC is not available when base = \"tree\", please choose other criteria")
        }

        if (criterion == "loo") {
            stop("minimizing leave-one-out error is not available when base = \"tree\", please choose other criteria")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        fit <- rpart(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), method = "class")
        ytrain.pred <- as.numeric(predict(fit, data.frame(x = xtrain.r), class = "class")) - 1
    }

    if (base == "lda") {
        if (criterion == "nric") {
            subspace.list <- sapply(1:B2, function(i) {
                # the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                -2*(p0*KL.divergence(xtrain[ytrain == 0, Si, drop = F], xtrain[ytrain == 1, Si, drop = F], k = kl.k[1])[kl.k[1]] + p1*KL.divergence(xtrain[ytrain == 1, Si, drop = F], xtrain[ytrain == 0, Si, drop = F], k = kl.k[2])[kl.k[2]]) + length(Si)*log(log(n))/sqrt(n)
            })
        }

        if (criterion == "ric") {
            subspace.list <- sapply(1:B2, function(i) {
                # the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                ric("lda", xtrain, ytrain, Si, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma.mle = Sigma.mle)
            })
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

        if (criterion == "ebic") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                # calc_BIC(xtrain, ytrain, Si, D = 0, K = 0, debug = F, gam = gam)
                calc_ebic(xtrain, ytrain, Si, gam)
            })
        }

        if (criterion == "bic") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                calc_ebic(xtrain, ytrain, Si, gam = 0)
            })
        }

        if (criterion == "aic") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                calc_aic(xtrain, ytrain, Si)
            })
        }

        if (criterion == "training") {
            subspace.list <- sapply(1:B2, function(i) {
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
            stop("minimizing leave-one-out error is not available when base = \"lda\", please choose other criteria")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        fit <- lda(x = as.matrix(xtrain.r), grouping = ytrain)
        ytrain.pred <- predict(fit, as.matrix(xtrain.r))$class

    }

    if (base == "qda") {
        if (criterion == "nric") {
            subspace.list <- sapply(1:B2, function(i) {
                # the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                -2*(p0*KL.divergence(xtrain[ytrain == 0, Si, drop = F], xtrain[ytrain == 1, Si, drop = F], k = kl.k[1])[kl.k[1]] + p1*KL.divergence(xtrain[ytrain == 1, Si, drop = F], xtrain[ytrain == 0, Si, drop = F], k = kl.k[2])[kl.k[2]]) + length(Si)*(length(Si) + 3)/2*log(log(n))/sqrt(n)
            })
        }

        if (criterion == "ric") {
            subspace.list <- sapply(1:B2, function(i) {
                # print(i) the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                ric("qda", xtrain, ytrain, Si, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma0.mle = Sigma0.mle, Sigma1.mle = Sigma1.mle,
                  p0 = p0, p1 = p1)
            })
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
            stop("minimizing leave-one-out error is not available when base = \"qda\", please choose other criteria")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        fit <- qda(x = xtrain.r, grouping = ytrain)
        ytrain.pred <- predict(fit, xtrain.r)$class
    }

    return(list(fit = fit, ytrain.pred = as.numeric(ytrain.pred) - 1, subset = S))

}

RaSubset <- function(xtrain, ytrain, xval, yval, B2, S, base, k, criterion, cv, t0.mle = NULL, t1.mle = NULL, mu0.mle = NULL,  mu1.mle = NULL, Sigma.mle = NULL, Sigma0.mle = NULL, Sigma1.mle = NULL, gam = NULL, kl.k = NULL, lower.limits = NULL, upper.limits = NULL, weights = NULL, ...) {
    list2env(list(...), environment())
    n <- length(ytrain)
    p <- ncol(xtrain)
    p0 <- sum(ytrain == 0)/length(ytrain)
    p1 <- sum(ytrain == 1)/length(ytrain)

    if (all(base == "gamma")) {
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
            stop("minimizing cross-validation error is not available when base = \"gamma\", please choose other criterion")
        }

        if (criterion == "ebic") {
            stop("minimizing eBIC is not available when base = \"gamma\", please choose other criterion")
        }

        if (criterion == "loo") {
            stop("minimizing leave-one-out error is not available when base = \"gamma\", please choose other criterion")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
        fit <- list(t0.mle, t1.mle, p0, p1)
        ytrain.pred <- factor(gamma_classifier(t0.mle, t1.mle, p0, p1, newx = xtrain, S))
    }

    if (all(base == "logistic")) {
        if (criterion == "auc") {
            if (all(is.null(lower.limits)) && all(is.null(upper.limits))) {
                subspace.list <- sapply(1:B2, function(i) {
                    Si <- S[, i][!is.na(S[, i])]  # current subspace
                    xtrain.r <- xtrain[, Si, drop = F]
                    score <- predict(glm(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), family = "binomial", weights = weights), data.frame(x = xtrain.r))
                    -auc(ytrain, score)
                })
            } else {
                if (all(is.null(lower.limits))) {
                    lower.limits <- rep(-Inf, p)
                }

                if (all(is.null(upper.limits))) {
                    upper.limits <- rep(Inf, p)
                }

                subspace.list <- sapply(1:B2, function(i) {
                    Si <- S[, i][!is.na(S[, i])]  # current subspace
                    xtrain.r <- xtrain[, Si, drop = F]
                    score <- predict(glmnet(x = xtrain.r, y = ytrain, family = "binomial", alpha = 1, lambda = 0, weights = weights, upper.limits = upper.limits, lower.limits = lower.limits), xtrain.r)
                    -auc(ytrain, score)
                })
            }
        }


        if (criterion == "nric") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                -2*(p0*KL.divergence(xtrain[ytrain == 0, Si, drop = F], xtrain[ytrain == 1, Si, drop = F], k = kl.k[1])[kl.k[1]] + p1*KL.divergence(xtrain[ytrain == 1, Si, drop = F], xtrain[ytrain == 0, Si, drop = F], k = kl.k[2])[kl.k[2]]) + length(Si)*log(log(n))/sqrt(n)
            })
        }

        if (is.null(weights)) {
            weights <- rep(1, n)/n
        }

        if (criterion == "ric") {
            if (all(is.null(lower.limits)) && all(is.null(upper.limits))) {
                subspace.list <- sapply(1:B2, function(i) {
                    Si <- S[, i][!is.na(S[, i])]  # current subspace
                    xtrain.r <- xtrain[, Si, drop = F]
                    score <- predict(glm(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), family = "binomial"), data.frame(x = xtrain.r))
                    posterior0 <- 1/(1 + exp(score))
                    posterior1 <- 1 - posterior0
                    ric("other", xtrain, ytrain, Si, p0 = p0, p1 = p1, posterior0 = posterior0, posterior1 = posterior1, weights = weights, deg = function(i) {
                        i
                    })
                })
            } else {
                if (all(is.null(lower.limits))) {
                    lower.limits <- rep(-Inf, p)
                }

                if (all(is.null(upper.limits))) {
                    upper.limits <- rep(Inf, p)
                }
                subspace.list <- sapply(1:B2, function(i) {
                    Si <- S[, i][!is.na(S[, i])]  # current subspace
                    xtrain.r <- xtrain[, Si, drop = F]
                    score <- predict(glmnet(x = xtrain.r, y = ytrain, family = "binomial", intercept = FALSE, alpha = 1, lambda = 0, weights = weights, lower.limits = lower.limits, upper.limits = upper.limits), xtrain.r)
                    posterior0 <- 1/(1 + exp(score))
                    posterior1 <- 1 - posterior0
                    ric("other", xtrain, ytrain, Si, p0 = p0, p1 = p1, posterior0 = posterior0, posterior1 = posterior1, weights = weights, deg = function(i) {
                        i
                    })
                })
            }

        }


        if (criterion == "aic") {
            if (all(is.null(lower.limits)) && all(is.null(upper.limits))) {
                subspace.list <- sapply(1:B2, function(i) {
                    Si <- S[, i][!is.na(S[, i])]  # current subspace
                    calc_aic(xtrain, ytrain, Si, weights = weights)
                })
            } else {
                if (all(is.null(lower.limits))) {
                    lower.limits <- rep(-Inf, p)
                }

                if (all(is.null(upper.limits))) {
                    upper.limits <- rep(Inf, p)
                }

                subspace.list <- sapply(1:B2, function(i) {
                    Si <- S[, i][!is.na(S[, i])]  # current subspace
                    calc_aic_glmnet(x = xtrain, y = ytrain, S = Si, weights = weights, upper.limits = upper.limits[Si], lower.limits = lower.limits[Si])
                })
            }
        }

        if (criterion == "ebic" || criterion == "bic") {
            if (criterion == "bic") {
                gam <- 0
            }

            if (all(is.null(lower.limits)) && all(is.null(upper.limits))) {
                subspace.list <- sapply(1:B2, function(i) {
                    Si <- S[, i][!is.na(S[, i])]  # current subspace
                    calc_ebic(xtrain, ytrain, Si, gam, weights = weights)
                })
            } else {
                if (all(is.null(lower.limits))) {
                    lower.limits <- rep(-Inf, p)
                }

                if (all(is.null(upper.limits))) {
                    upper.limits <- rep(Inf, p)
                }

                subspace.list <- sapply(1:B2, function(i) {
                    Si <- S[, i][!is.na(S[, i])]  # current subspace
                    calc_ebic_glmnet(x = xtrain, y = ytrain, S = Si, gam = gam, weights = weights, upper.limits = upper.limits[Si], lower.limits = lower.limits[Si])
                })
            }

        }


        if (criterion == "training") {
            if (all(is.null(lower.limits)) && all(is.null(upper.limits))) {
                subspace.list <- sapply(1:B2, function(i) {
                    # the last row is training error for each i in 1:B2
                    Si <- S[, i][!is.na(S[, i])]  # current subspace
                    xtrain.r <- xtrain[, Si, drop = F]
                    mean(as.numeric(I(predict(glm(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), family = "binomial", weights = weights), data.frame(x = xtrain.r)) >
                      0)) != ytrain, na.rm = TRUE)
                })
            } else {
                if (all(is.null(lower.limits))) {
                    lower.limits <- rep(-Inf, p)
                }

                if (all(is.null(upper.limits))) {
                    upper.limits <- rep(Inf, p)
                }

                subspace.list <- sapply(1:B2, function(i) {
                    # the last row is training error for each i in 1:B2
                    Si <- S[, i][!is.na(S[, i])]  # current subspace
                    xtrain.r <- xtrain[, Si, drop = F]
                    mean(as.numeric(I(predict(glmnet(x = xtrain.r, y = ytrain, alpha = 1, lambda = 0, intercept = FALSE, family = "binomial", weights = weights, upper.limits = upper.limits, lower.limits = lower.limits), xtrain.r) >
                                          0)) != ytrain, na.rm = TRUE)
                })
            }
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
            stop("minimizing leave-one-out error is not available when base = \"logistic\", please choose other criterion")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        if (all(is.null(lower.limits)) && all(is.null(upper.limits)) || criterion == "nric") {
            fit <- glm(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), family = "binomial", weights = weights)
            ytrain.pred <- as.numeric(I(predict(fit, data.frame(x = xtrain.r)) > 0))
        } else {
            fit <- glmnet(x = xtrain.r, y = ytrain, family = "binomial", alpha = 1, lambda = 0, weights = weights, upper.limits =  upper.limits[S], lower.limits = lower.limits[S])
            ytrain.pred <- as.numeric(I(predict(fit, xtrain.r) > 0))
        }


    }

    if (all(base == "svm")) {
        if (!is.character(kernel)) {
            kernel <- "linear"
        }

        if (criterion == "auc") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                xtrain.r <- xtrain[, Si, drop = F]
                score <- as.numeric(attr(predict(svm(x = xtrain.r, y = ytrain, kernel = kernel, type = "C-classification", ...), xtrain.r),"decision.values"))
                -auc(ytrain, score)
            })
        }

        if (criterion == "training") {
            subspace.list <- sapply(1:B2, function(i) {
                # the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                xtrain.r <- xtrain[, Si, drop = F]
                mean(as.numeric(predict(svm(x = xtrain.r, y = ytrain, kernel = kernel, type = "C-classification", ...), xtrain.r)) - 1 !=
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
            stop("minimizing eBIC is not available when base = \"svm\", please choose other criterion")
        }

        if (criterion == "ric") {
            stop("minimizing RIC is not available when base = \"svm\", please choose other criterion")
        }

        if (criterion == "loo") {
            stop("minimizing leave-one-out error is not available when base = \"svm\", please choose other criterion")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        fit <- svm(x = xtrain.r, y = ytrain, kernel = kernel, type = "C-classification", ...)
        ytrain.pred <- as.numeric(predict(fit, xtrain.r)) - 1
    }


    if (all(base == "randomforest")) {
        if (criterion == "auc") {
            subspace.list <- sapply(1:B2, function(i) {
                # the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                xtrain.r <- xtrain[, Si, drop = F]
                score <- as.numeric(predict(randomForest(x = xtrain.r, y = factor(ytrain), ...), xtrain.r, type = "prob")[, 1])
                -auc(ytrain, score)
            })
        }


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
            stop("minimizing eBIC is not available when base = \"randomforest\", please choose other criterion")
        }

        if (criterion == "ric") {
            stop("minimizing RIC is not available when base = \"randomforest\", please choose other criterion")
        }

        if (criterion == "loo") {
            stop("minimizing leave-one-out error is not available when base = \"randomforest\", please choose other criterion")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        fit <- randomForest(x = xtrain.r, y = factor(ytrain), ...)
        ytrain.pred <- as.numeric(predict(fit, xtrain.r)) - 1
    }

    if (all(base == "knn")) {
        if (criterion == "auc") {
            subspace.list <- sapply(1:B2, function(i) {
                d <- length(S[, i][!is.na(S[, i])])  # subspace size
                Si <- matrix(S[, i][!is.na(S[, i])], nrow = d)  # current subspace
                xtrain.r <- xtrain[, Si, drop = F]
                knn.test <- sapply(k, function(j) {
                    rs <- knn.cv(xtrain.r, ytrain, j, use.all = FALSE, prob = TRUE)
                    - auc(ytrain, attr(rs,"prob"))
                })
                min(knn.test)
            })

            i0 <- which.min(subspace.list)
            S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

            xtrain.r <- xtrain[, S, drop = F]

            knn.test <- sapply(k, function(j) {
                rs <- knn.cv(xtrain.r, ytrain, j, use.all = FALSE, prob = TRUE)
                - auc(rs, attr(rs,"prob"))
            })
            k.op <- k[which.min(knn.test)]
            fit <- knn3(x = xtrain.r, y = factor(ytrain), k = k.op, use.all = FALSE)
            ytrain.pred <- predict(fit, xtrain.r, type = "class")
        }

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

            knn.test <- sapply(k, function(l) {
                mean(sapply(1:cv, function(j) {
                    mean(predict(knn3(x = xtrain.r[-folds[[j]], ,drop = F], y = factor(ytrain[-folds[[j]]]), k = l, use.all = FALSE), xtrain.r[folds[[j]], , drop = F], type = "class") != ytrain[folds[[j]]], na.rm = TRUE)
                }))

            })
            k.op <- k[which.min(knn.test)]
            fit <- knn3(x = xtrain.r, y = factor(ytrain), k = k.op, use.all = FALSE)
            ytrain.pred <- predict(fit, xtrain.r, type = "class")
        }

        if (criterion == "training") {
            stop("minimizing training error is not available when base = \"knn\", please choose other criterion")
        }

        if (criterion == "ebic") {
            stop("minimizing eBIC is not available when base = \"knn\", please choose other criterion")
        }

        if (criterion == "ric") {
            stop("minimizing RIC is not available when base = \"knn\", please choose other criterion")
        }

    }

    if (all(base == "tree")) {
        ytrain <- factor(ytrain)
        if (criterion == "auc") {
            subspace.list <- sapply(1:B2, function(i) {
                # the last row is training error for each i in 1:B2
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                xtrain.r <- xtrain[, Si, drop = F]
                fit <- rpart(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), method = "class")
                score <- as.numeric(predict(fit, data.frame(x = xtrain.r), type = "prob")[, 2])
                -auc(ytrain, score)
            })
        }

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
            stop("minimizing RIC is not available when base = \"tree\", please choose other criterion")
        }

        if (criterion == "ebic") {
            stop("minimizing eBIC is not available when base = \"tree\", please choose other criterion")
        }

        if (criterion == "loo") {
            stop("minimizing leave-one-out error is not available when base = \"tree\", please choose other criterion")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        fit <- rpart(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), method = "class")
        ytrain.pred <- predict(fit, data.frame(x = xtrain.r), type = "class")
    }

    if (all(base == "lda")) {
        if (criterion == "auc") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                xtrain.r <- xtrain[, Si, drop = F]
                score <- as.numeric(predict(lda(x = xtrain.r, grouping = ytrain), xtrain.r)$x)
                -auc(ytrain, score)
            })
        }

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
            stop("minimizing leave-one-out error is not available when base = \"lda\", please choose other criterion")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        fit <- lda(x = as.matrix(xtrain.r), grouping = ytrain, ...)
        ytrain.pred <- predict(fit, as.matrix(xtrain.r))$class

    }

    if (all(base == "qda")) {
        if (criterion == "auc") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                xtrain.r <- xtrain[, Si, drop = F]
                score <- predict(qda(x = xtrain.r, grouping = ytrain), xtrain.r)$posterior[, 2]
                -auc(ytrain, score)
            })
        }

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
            stop("minimizing leave-one-out error is not available when base = \"qda\", please choose other criterion")
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace

        xtrain.r <- xtrain[, S, drop = F]
        fit <- qda(x = xtrain.r, grouping = ytrain, ...)
        ytrain.pred <- predict(fit, xtrain.r)$class
    }

    if (length(unique(base)) == 1) {
        return(list(fit = fit, ytrain.pred = as.numeric(ytrain.pred) - 1, subset = S, base.list = base[i0]))
    }

    # super RaSE
    # ---------------------------------------------------

    if (length(unique(base)) > 1) {
        if (criterion == "training") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                xtrain.r <- xtrain[, Si, drop = F]
                if (base[i] == "qda"){
                    mean(predict(qda(x = xtrain.r, grouping = ytrain), xtrain.r)$class != ytrain, na.rm = TRUE)
                } else if (base[i] == "lda"){
                    mean(predict(lda(x = xtrain.r, grouping = ytrain), xtrain.r)$class != ytrain, na.rm = TRUE)
                } else if (base[i] == "svm"){
                    mean(as.numeric(predict(svm(x = xtrain.r, y = ytrain, kernel = kernel, type = "C-classification", ...), xtrain.r)) - 1 != ytrain, na.rm = TRUE)
                } else if (base[i] == "tree"){
                    ytrain <- factor(ytrain)
                    fit <- rpart(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), method = "class")
                    score <- as.numeric(predict(fit, data.frame(x = xtrain.r), type = "prob")[, 2])
                    -auc(ytrain, score)
                } else if (base[i] == "randomforest"){
                    mean(as.numeric(predict(randomForest(x = xtrain.r, y = factor(ytrain)), xtrain.r)) - 1 != factor(ytrain), na.rm = TRUE)
                } else if (base[i] == "logistic"){
                    mean(as.numeric(I(predict(glm(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), family = "binomial", weights = weights), data.frame(x = xtrain.r)) > 0)) != ytrain, na.rm = TRUE)
                } else if (base[i] == "knn"){
                    stop("'criterion' cannot be 'training' when base classifiers include 'knn'! Please check your input.")
                }
            })
        } else if (criterion == "cv") {
            if (!is.character(kernel)) {
                kernel <- "linear"
            }
            folds <- createFolds(ytrain, k = cv)
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                xtrain.r <- xtrain[, Si, drop = F]
                if (base[i] == "qda"){
                    mean(sapply(1:cv, function(j) {
                        mean(predict(qda(x = xtrain.r[-folds[[j]], , drop = F], grouping = ytrain[-folds[[j]]]), xtrain.r[folds[[j]], , drop = F])$class !=
                                 ytrain[folds[[j]]], na.rm = TRUE)
                    }))
                } else if (base[i] == "lda"){
                    mean(sapply(1:cv, function(j) {
                        mean(predict(lda(x = xtrain.r[-folds[[j]], , drop = F], grouping = ytrain[-folds[[j]]]), xtrain.r[folds[[j]], , drop = F])$class !=
                                 ytrain[folds[[j]]], na.rm = TRUE)
                    }))
                } else if (base[i] == "svm"){
                    mean(sapply(1:cv, function(j) {
                        mean(as.numeric(predict(svm(x = xtrain.r[-folds[[j]], , drop = F], y = ytrain[-folds[[j]]], kernel = kernel, type = "C-classification", ...), xtrain.r[folds[[j]], , drop = F])) - 1 != ytrain[folds[[j]]], na.rm = TRUE)
                    }))
                } else if (base[i] == "tree"){
                    ytrain <- factor(ytrain)
                    mean(sapply(1:cv, function(j) {
                        fit <- rpart(y ~ ., data = data.frame(x = xtrain.r[-folds[[j]], , drop = F], y = ytrain[-folds[[j]]]), method = "class")
                        mean((as.numeric(predict(fit, data.frame(x = xtrain.r[folds[[j]], , drop = F]), type = "class")) - 1) != ytrain[folds[[j]]], na.rm = TRUE)
                    }))
                } else if (base[i] == "randomforest"){
                    mean(sapply(1:cv, function(j) {
                        mean(as.numeric(predict(randomForest(x = xtrain.r[-folds[[j]], , drop = F], y = factor(ytrain)[-folds[[j]]]), xtrain.r[folds[[j]], , drop = F])) - 1 != factor(ytrain)[folds[[j]]], na.rm = TRUE)
                    }))
                } else if (base[i] == "logistic"){
                    mean(sapply(1:cv, function(j) {
                        mean(as.numeric(I(predict(glm(y ~ ., data = data.frame(x = xtrain.r[-folds[[j]], , drop = F], y = ytrain[-folds[[j]]]), family = "binomial", weights = weights), data.frame(x = xtrain.r[folds[[j]], , drop = F])) > 0)) != ytrain[folds[[j]]], na.rm = TRUE)
                    }))
                } else if (base[i] == "knn") {
                    knn.test <- sapply(k, function(l) {
                        mean(sapply(1:cv, function(j) {
                            mean(predict(knn3(x = xtrain.r[-folds[[j]], , drop = F], y = factor(ytrain[-folds[[j]]]), k = l, use.all = FALSE), xtrain.r[folds[[j]], , drop = F], type = "class") != ytrain[folds[[j]]], na.rm = TRUE)
                        }))
                    })
                    min(knn.test)
                }
            })
        } else if (criterion == "auc") {
            subspace.list <- sapply(1:B2, function(i) {
                Si <- S[, i][!is.na(S[, i])]  # current subspace
                xtrain.r <- xtrain[, Si, drop = F]
                if (base[i] == "qda"){
                    score <- as.numeric(predict(qda(x = xtrain.r, grouping = ytrain), xtrain.r)$x)
                    -auc(ytrain, score)
                } else if (base[i] == "lda"){
                    score <- as.numeric(predict(lda(x = xtrain.r, grouping = ytrain), xtrain.r)$x)
                    -auc(ytrain, score)
                } else if (base[i] == "svm"){
                    stop("'criterion' cannot be 'auc' when base classifiers include 'svm'! Please check your input.")
                } else if (base[i] == "tree"){
                    ytrain <- factor(ytrain)
                    fit <- rpart(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), method = "class")
                    score <- as.numeric(predict(fit, data.frame(x = xtrain.r), type = "prob")[, 2])
                    -auc(ytrain, score)
                } else if (base[i] == "randomforest"){
                    score <- as.numeric(predict(randomForest(x = xtrain.r, y = factor(ytrain), ...), xtrain.r, type = "prob")[, 1])
                    -auc(ytrain, score)
                } else if (base[i] == "logistic"){
                    score <- predict(glm(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), family = "binomial", weights = weights), data.frame(x = xtrain.r))
                    -auc(ytrain, score)
                } else if (base[i] == "knn") {
                    knn.test <- sapply(k, function(j) {
                        rs <- knn.cv(xtrain.r, ytrain, j, use.all = FALSE, prob = TRUE)
                        - auc(ytrain, attr(rs,"prob"))
                    })
                    min(knn.test)
                }
            })
        }

        i0 <- which.min(subspace.list)
        S <- S[!is.na(S[, i0]), i0]  # final optimal subspace
        xtrain.r <- xtrain[, S, drop = F]
        if (base[i0] == "qda"){
            fit <- qda(x = xtrain.r, grouping = ytrain, ...)
            ytrain.pred <- as.numeric(predict(fit, xtrain.r)$class) - 1
        }
        if (base[i0] == "lda"){
            fit <- lda(x = xtrain.r, grouping = ytrain, ...)
            ytrain.pred <- as.numeric(predict(fit, xtrain.r)$class) - 1
        }
        if (base[i0] == "svm"){
            fit <- svm(x = xtrain.r, y = ytrain, kernel = kernel, type = "C-classification", ...)
            ytrain.pred <- as.numeric(predict(fit, xtrain.r)) - 1
        }
        if (base[i0] == "tree"){
            fit <- rpart(y ~ ., data = data.frame(x = xtrain.r, y = factor(ytrain)), method = "class")
            ytrain.pred <- as.numeric(predict(fit, data.frame(x = xtrain.r), type = "class")) - 1
        }
        if (base[i0] == "randomforest"){
            fit <- randomForest(x = xtrain.r, y = factor(ytrain))
            ytrain.pred <- as.numeric(predict(fit, xtrain.r)) - 1
        }
        if (base[i0] == "logistic"){
            fit <- glm(y ~ ., data = data.frame(x = xtrain.r, y = ytrain), family = "binomial", weights = weights)
            ytrain.pred <- as.numeric(I(predict(fit, data.frame(x = xtrain.r)) > 0))
        }
        if (base[i0] == "knn") {
            if (criterion == "loo") {
                knn.test <- sapply(k, function(j) {
                    mean(knn.cv(xtrain.r, ytrain, j, use.all = FALSE) != ytrain, na.rm = TRUE)
                })
            } else if (criterion == "cv") {
                knn.test <- sapply(k, function(l) {
                    mean(sapply(1:cv, function(j) {
                        mean(predict(knn3(x = xtrain.r[-folds[[j]], , drop = F], y = factor(ytrain[-folds[[j]]]), k = l, use.all = FALSE), xtrain.r[folds[[j]], , drop = F], type = "class") != ytrain[folds[[j]]], na.rm = TRUE)
                    }))
                })
            } else if (criterion == "auc") {
                knn.test <- sapply(k, function(j) {
                    rs <- knn.cv(xtrain.r, ytrain, j, use.all = FALSE, prob = TRUE)
                    - auc(rs, attr(rs,"prob"))
                })
            } else if (criterion == "validation") {
                xval.r <- xval[, S, drop = F]
                knn.test <- sapply(k, function(j) {
                    fit <- knn3(x = xtrain.r, y = factor(ytrain), k = j, use.all = FALSE)
                    mean(as.numeric(predict(fit, xval.r, type = "class")) - 1 != yval)
                })
            }

            k.op <- k[which.min(knn.test)]
            fit <- knn3(x = xtrain.r, y = factor(ytrain), k = k.op, use.all = FALSE)
            ytrain.pred <- as.numeric(predict(fit, xtrain.r, type = "class")) - 1
        }



        return(list(fit = fit, ytrain.pred = ytrain.pred, subset = S, base.list = base[i0]))
    }





}

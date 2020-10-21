#' Construct the random subspace ensemble classifier.
#'
#' \code{Rase} is a novel model-free ensemble classification framework to solve the sparse classification problem. In RaSE algorithm, for each of the B1 weak learners, B2 random subspaces are generated and the optimal one is chosen to train the model on the basis of some criterion.
#' @export
#' @importFrom MASS lda
#' @importFrom MASS qda
#' @importFrom MASS mvrnorm
#' @importFrom class knn
#' @importFrom class knn.cv
#' @importFrom caret knn3
#' @importFrom caret createFolds
#' @importFrom caret findLinearCombos
#' @importFrom doParallel registerDoParallel
#' @importFrom doParallel stopImplicitCluster
#' @importFrom foreach foreach
#' @importFrom foreach %dopar%
#' @importFrom parallel detectCores
#' @importFrom rpart rpart
#' @importFrom nnet multinom
#' @importFrom randomForest randomForest
#' @importFrom e1071 svm
#' @importFrom stats glm
#' @importFrom stats predict
#' @importFrom stats nlm
#' @importFrom stats rmultinom
#' @importFrom stats rgamma
#' @importFrom stats dgamma
#' @importFrom stats ecdf
#' @importFrom stats optimise
#' @importFrom stats cov
#' @importFrom stats cor
#' @importFrom stats var
#' @importFrom ggplot2 ggplot
#' @importFrom ggplot2 aes
#' @importFrom ggplot2 aes_string
#' @importFrom ggplot2 geom_point
#' @importFrom ggplot2 ggtitle
#' @importFrom ggplot2 theme
#' @importFrom ggplot2 element_text
#' @importFrom ggplot2 expr
#' @importFrom ggplot2 labs
#' @importFrom gridExtra grid.arrange
#' @importFrom formatR tidy_eval
#' @importFrom FNN KL.divergence
#' @param xtrain n * p observation matrix. n observations, p features.
#' @param ytrain n 0/1 observatons.
#' @param xval observation matrix for validation. Default = \code{NULL}. Useful only when \code{criterion} = 'validation'.
#' @param yval 0/1 observation for validation. Default = \code{NULL}. Useful only when \code{criterion} = 'validation'.
#' @param B1 the number of weak learners. Default = 200.
#' @param B2 the number of subspace candidates generated for each weak learner. Default = 500.
#' @param D the maximal subspace size when generating random subspaces from the uniform distribution. Default = \code{NULL}, which is \eqn{min(\sqrt n0, \sqrt n1, p)} when \code{base} = 'lda' and is \eqn{min(\sqrt n, p)} otherwise.
#' @param dist the distribution for features when generating random subspaces. Default = \code{NULL}, which represents the uniform distribution. First generate an integer \eqn{d} from \eqn{1,...,D} uniformly, then uniformly generate a subset with cardinality \eqn{d}.
#' @param base the type of base classifier. Default = 'lda'.
#' \itemize{
#' \item lda: linear discriminant analysis. \code{\link[MASS]{lda}} in \code{MASS} package.
#' \item qda: quadratic discriminant analysis. \code{\link[MASS]{qda}} in \code{MASS} package.
#' \item knn: k-nearest neighbor. \code{\link[class]{knn}}, \code{\link[class]{knn.cv}} in \code{class} package and \code{\link[caret]{knn3}} in \code{caret} package.
#' \item logistic: logistic regression. \code{\link[glmnet]{glmnet}} in \code{glmnet} package.
#' \item tree: decision tree. \code{\link[rpart]{rpart}} in \code{rpart} package.
#' \item svm: support vector machine. \code{\link[e1071]{svm}} in \code{e1071} package.
#' \item randomforest: random forest. \code{\link[randomForest]{randomForest}} in \code{randomForest} package.
#' \item gamma: Bayesian classifier for multivariate gamma distribution with independent marginals.
#' }
#' @param criterion the criterion to choose the best subspace for each weak learner. Default = 'ric' when \code{base} = 'lda', 'qda', 'gamma'; default = 'ebic' and set \code{gam} = 0 when \code{base} = 'logistic'; default = 'loo' when \code{base} = 'knn'; default = 'training' when \code{base} = 'tree', 'svm', 'randomforest'.
#' \itemize{
#' \item ric: minimizing ratio information criterion with parametric estimation (Tian, Y. and Feng, Y., 2020). Available when \code{base} = 'lda', 'qda', 'gamma' or 'logistic'.
#' \item nric: minimizing ratio information criterion with non-parametric estimation (Tian, Y. and Feng, Y., 2020; Wang, Q., Kulkarni, S.R. and Verdú, S., 2009). Available when \code{base} = 'lda', 'qda', 'gamma' or 'logistic'.
#' \item training: minimizing training error. Not available when \code{base} = 'knn'.
#' \item loo: minimizing leave-one-out error. Only available when  \code{base} = 'knn'.
#' \item validation: minimizing validation error based on the validation data. Available for all base classifiers.
#' \item cv: minimizing k-fold cross-validation error. k equals to the value of \code{cv}. Default = 10. Not available when \code{base} = 'gamma'.
#' \item ebic: minimizing extended Bayesian information criterion (Chen, J. and Chen, Z., 2008; 2012). Need to assign value for \code{gam}. When \code{gam} = 0, it denotes the classical BIC. Available when \code{base} = 'lda', 'qda' or 'logistic'.
#'
#' EBIC = -2 * log-likelihood + |S| * log(n) + 2 * |S| * gam * log(p).
#' }
#' @param ranking whether the function outputs the selected percentage of each feature in B1 subspaces. Logistic, default = TRUE.
#' @param k the number of nearest neightbors considered when \code{base} = 'knn'. Only useful when \code{base} = 'knn'.
#' @param cores the number of cores used for parallel computing. Default = 1.
#' @param seed the random seed assigned at the start of the algorithm, which can be a real number or \code{NULL}. Default = \code{NULL}, in which case no random seed will be set.
#' @param iteration the number of iterations. Default = 0.
#' @param cutoff whether to use the empirically optimal threshold. Logistic, default = TRUE. If it is FALSE, the threshold will be set as 0.5.
#' @param cv the number of cross-validations used. Default = 10. Only useful when \code{criterion} = 'cv'.
#' @param scale whether to normalize the data. Logistic, default = FALSE.
#' @param C0 the threshold used to adjust the sampling probabilities of features when \code{iteration} > 0. Default = 0.1.
#' @param kl.k the number of nearest neighbors used to estimate KL divergences when \code{criterion} = 'nric'. 2-dimensional vector. Default = \code{NULL}, in which case it will be set as \eqn{\sqrt n0, \sqrt n1}.
#' @param ... additional arguments.
#' @return An object with S3 class \code{'RaSE'}.
#' \item{marginal}{the marginal probability for each class.}
#' \item{base}{the type of base classifier.}
#' \item{criterion}{the criterion to choose the best subspace for each weak learner.}
#' \item{B1}{the number of weak learners.}
#' \item{B2}{the number of subspace candidates generated for each weak learner.}
#' \item{iteration}{the number of iterations.}
#' \item{fit.list}{sequence of B1 fitted base classifiers.}
#' \item{cutoff}{the empirically optimal threshold.}
#' \item{subspace}{sequence of subspaces correponding to B1 weak learners.}
#' \item{ranking}{the selected percentage of each feature in B1 subspaces.}
#' \item{scale}{a list of scaling parameters, including the scaling center and the scale parameter for each feature. Equals to \code{NULL} when the data is not scaled in \code{RaSE} model fitting.}
#' \item{C0}{the threshold used to adjust the sampling probabilities of features when \code{iteration} > 0.}
#' @seealso \code{\link{predict.RaSE}}, \code{\link{RaModel}}, \code{\link{print.RaSE}}, \code{\link{RaPlot}}.
#' @references
#' Tian, Y. and Feng, Y., 2020. RaSE: Random subspace ensemble classification. arXiv preprint arXiv:2006.08855.
#'
#' Chen, J. and Chen, Z., 2008. Extended Bayesian information criteria for model selection with large model spaces. Biometrika, 95(3), pp.759-771.
#'
#' Chen, J. and Chen, Z., 2012. Extended BIC for small-n-large-P sparse GLM. Statistica Sinica, pp.555-574.
#'
#' Wang, Q., Kulkarni, S.R. and Verdú, S., 2009. Divergence estimation for multidimensional densities via $ k $-nearest-neighbor distances. IEEE Transactions on Information Theory, 55(5), pp.2392-2405.#' @examples
#'
#' @examples
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' train.data <- RaModel(1, n = 100, p = 50)
#' test.data <- RaModel(1, n = 100, p = 50)
#' xtrain <- train.data$x
#' ytrain <- train.data$y
#' xtest <- test.data$x
#' ytest <- test.data$y
#'
#' # test RaSE classifier with LDA base classifier
#' fit <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, base = 'lda',
#' cores = 2, criterion = 'ric')
#' mean(predict(fit, xtest) != ytest)
#'
#' \dontrun{
#' # test RaSE classifier with LDA base classifier and 1 iteration round
#' fit <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 1, base = 'lda',
#' cores = 2, criterion = 'ric')
#' mean(predict(fit, xtest) != ytest)
#'
#' # test RaSE classifier with QDA base classifier and 1 iteration round
#' fit <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 1, base = 'qda',
#' cores = 2, criterion = 'ric')
#' mean(predict(fit, xtest) != ytest)
#'
#' # test RaSE classifier with knn base classifier
#' fit <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, base = 'knn',
#' cores = 2, criterion = 'loo')
#' mean(predict(fit, xtest) != ytest)
#'
#' # test RaSE classifier with logistic regression base classifier
#' fit <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, base = 'logistic',
#' cores = 2, criterion = 'ebic', gam = 0)
#' mean(predict(fit, xtest) != ytest)
#'
#' # test RaSE classifier with svm base classifier
#' fit <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, base = 'svm',
#' cores = 2, criterion = 'training')
#' mean(predict(fit, xtest) != ytest)
#'
#' # test RaSE classifier with random forest base classifier
#' fit <- Rase(xtrain, ytrain, B1 = 20, B2 = 10, iteration = 0, base = 'randomforest',
#' cores = 2, criterion = 'cv', cv = 3)
#' mean(predict(fit, xtest) != ytest)
#' }

Rase <- function(xtrain, ytrain, xval = NULL, yval = NULL, B1 = 200, B2 = 500, D = NULL, dist = NULL, base = c("lda",
    "qda", "knn", "logistic", "tree", "svm", "randomforest", "gamma"), criterion = NULL, ranking = TRUE, k = c(3, 5, 7, 9, 11), cores = 1,
    seed = NULL, iteration = 0, cutoff = TRUE, cv = 10, scale = FALSE, C0 = 0.1, kl.k = NULL, ...) {


    if (!is.null(seed)) {
        set.seed(seed, kind = "L'Ecuyer-CMRG")
    }

    if (is.null(criterion)) {
        if (base == "lda" || base == "qda" || base == "gamma") {
            criterion <- "ric"
        } else if (base == "logistic") {
            criterion <- "ebic"
            gam <- 0
        } else if (base == "knn") {
            criterion <- "loo"
        } else {
            criterion <- "training"
        }
    }


    xtrain <- as.matrix(xtrain)
    base <- match.arg(base)
    p <- ncol(xtrain)
    n <- length(ytrain)
    n0 <- sum(ytrain == 0)
    n1 <- sum(ytrain == 1)


    if(is.null(kl.k)) {
        kl.k <- floor(c(sqrt(n0), sqrt(n1)))
    }

    if (scale == TRUE) {
        L <- scale_Rase(xtrain)
        xtrain <- L$data
        scale.center <- L$center
        scale.scale <- L$scale
    }

    registerDoParallel(cores)

    # remove redundant features
    if (base == "lda") {
        # clean data
        a <- suppressWarnings(cor(xtrain))
        b <- a - diag(diag(a))
        b0 <- which(abs(b) > 0.9999, arr.ind = T)
        b0 <- matrix(b0[b0[, 1] > b0[, 2], ], ncol = 2)
        a <- diag(cov(xtrain))

        delete.ind <- unique(c(which(a == 0), b0[, 1]))
        sig.ind <- setdiff(1:p, delete.ind)

        # estimate parameters
        if (is.null(D)) {
            D <- floor(min(sqrt(n), length(sig.ind)))
        }
        Sigma.mle <- ((n0 - 1) * cov(xtrain[ytrain == 0, , drop = F]) + (n1 - 1) * cov(xtrain[ytrain == 1, , drop = F]))/n
        mu0.mle <- colMeans(xtrain[ytrain == 0, , drop = F])
        mu1.mle <- colMeans(xtrain[ytrain == 1, , drop = F])

        # start loops
        dist <- rep(1, p)
        dist[delete.ind] <- 0
        for (t in 1:(iteration + 1)) {
            output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
                S <- sapply(1:B2, function(j) {
                  S.size <- sample(1:D, 1)
                  c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=
                    0]))))
                })
                S <- sapply(1:B2, function(j) {
                  snew <- S[!is.na(S[, j]), j]
                  if (length(snew) > 2) {
                    ind0 <- findLinearCombos(Sigma.mle[snew, snew, drop = F])$remove
                    if (!is.null(ind0)) {
                      snew <- snew[-ind0]
                    }
                  }
                  c(snew, rep(NA, D - length(snew)))
                })

                 RaSubset(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, base = base, k = k,
                  criterion = criterion, cv = cv, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma.mle = Sigma.mle, kl.k = kl.k, ...)
            }

            subspace <- output[, 3]

            s <- rep(0, p)
            for (i in 1:length(subspace)) {
                s[subspace[[i]]] <- s[subspace[[i]]] + 1
            }


            dist <- s/sum(s)
            dist[dist < C0/log(p)] <- C0/p
            dist[delete.ind] <- 0

            ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))

        }

        fit.list <- output[, 1]

    }

    if (base == "qda") {
        # clean data
        a <- suppressWarnings(cor(xtrain[ytrain == 0, ]))
        b <- a - diag(diag(a))

        b0 <- which(abs(b) > 0.9999, arr.ind = T)
        b0 <- matrix(b0[b0[, 1] > b0[, 2], ], ncol = 2)
        a0 <- diag(cov(xtrain[ytrain == 0, ]))

        a <- suppressWarnings(cor(xtrain[ytrain == 1, ]))
        b <- a - diag(diag(a))

        b1 <- which(abs(b) > 0.9999, arr.ind = T)
        b1 <- matrix(b1[b1[, 1] > b1[, 2], ], ncol = 2)
        a1 <- diag(cov(xtrain[ytrain == 1, ]))

        delete.ind <- unique(c(b0[, 1], b1[, 1], which(a0 == 0), which(a1 == 0)))
        sig.ind <- setdiff(1:p, delete.ind)

        # estimate parameters
        if (is.null(D)) {
            D <- floor(min(sqrt(n0), sqrt(n1), length(sig.ind)))
        }
        Sigma0.mle <- (n0 - 1)/n0 * cov(xtrain[ytrain == 0, , drop = F])
        Sigma1.mle <- (n1 - 1)/n1 * cov(xtrain[ytrain == 1, , drop = F])
        mu0.mle <- colMeans(xtrain[ytrain == 0, , drop = F])
        mu1.mle <- colMeans(xtrain[ytrain == 1, , drop = F])

        # start loops
        dist <- rep(1, p)
        dist[delete.ind] <- 0

        for (t in 1:(iteration + 1)) {
            output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
                S <- sapply(1:B2, function(j) {
                  S.size <- sample(1:D, 1)
                  c(sample(1:p, size = min(S.size, sum(dist != 0)), prob = dist), rep(NA, D - min(S.size, sum(dist != 0))))
                })
                S <- sapply(1:B2, function(j) {
                  snew <- S[!is.na(S[, j]), j]
                  if (length(snew) > 2) {
                    ind0 <- findLinearCombos(Sigma0.mle[snew, snew, drop = F])$remove
                    ind1 <- findLinearCombos(Sigma1.mle[snew, snew, drop = F])$remove
                    if (!all(is.null(c(ind0, ind1)))) {
                      snew <- snew[-c(ind0, ind1)]
                    }
                  }
                  c(snew, rep(NA, D - length(snew)))
                })
                RaSubset(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, base = base, k = k,
                  criterion = criterion, cv = cv, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma0.mle = Sigma0.mle, Sigma1.mle = Sigma1.mle, kl.k = kl.k,
                  ...)
            }

            subspace <- output[, 3]

            s <- rep(0, p)
            for (i in 1:length(subspace)) {
                s[subspace[[i]]] <- s[subspace[[i]]] + 1
            }

            dist <- s/sum(s)
            dist[dist < C0/log(p)] <- C0/p
            dist[delete.ind] <- 0

            ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))

        }

        fit.list <- output[, 1]

    }

    if (base == "knn") {
        # estimate parameters
        if (is.null(D)) {
            D <- floor(min(sqrt(n), p))
        }

        # start loops
        dist <- rep(1, p)
        for (t in 1:(iteration + 1)) {
            output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
                S <- sapply(1:B2, function(j) {
                  S.size <- sample(1:D, 1)
                  c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=
                    0]))))
                })
                RaSubset(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, base = base, k = k, kl.k = kl.k,
                  criterion = criterion, cv = cv, ...)
            }

            subspace <- output[, 3]

            s <- rep(0, p)
            for (i in 1:length(subspace)) {
                s[subspace[[i]]] <- s[subspace[[i]]] + 1
            }

            dist <- s/sum(s)
            dist[dist < C0/log(p)] <- C0/p
            ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))
        }

        fit.list <- output[, 1]
    }

    if (base == "tree") {
        # estimate parameters
        if (is.null(D)) {
            D <- floor(min(sqrt(n), p))
        }

        # start loops
        dist <- rep(1, p)
        for (t in 1:(iteration + 1)) {
            output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
                S <- sapply(1:B2, function(j) {
                  S.size <- sample(1:D, 1)
                  c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=
                    0]))))
                })
                RaSubset(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, base = base, k = k,
                  criterion = criterion, cv = cv, ...)
            }

            subspace <- output[, 3]

            s <- rep(0, p)
            for (i in 1:length(subspace)) {
                s[subspace[[i]]] <- s[subspace[[i]]] + 1
            }

            dist <- s/sum(s)
            dist[dist < C0/log(p)] <- C0/p
            ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))
        }

        fit.list <- output[, 1]
    }

    if (base == "logistic" || base == "svm" || base == "randomforest") {
        # estimate parameters
        if (is.null(D)) {
            D <- floor(min(sqrt(n), p))
        }

        # start loops
        dist <- rep(1, p)
        for (t in 1:(iteration + 1)) {
            output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
                S <- sapply(1:B2, function(j) {
                  S.size <- sample(1:D, 1)
                  c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=
                    0]))))
                })
                RaSubset(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, base = base, k = k, kl.k = kl.k,
                  criterion = criterion, cv = cv, ...)
            }

            subspace <- output[, 3]

            s <- rep(0, p)
            for (i in 1:length(subspace)) {
                s[subspace[[i]]] <- s[subspace[[i]]] + 1
            }

            dist <- s/sum(s)
            dist[dist < C0/log(p)] <- C0/p
            ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))
        }

        fit.list <- output[, 1]
    }


    if (base == "gamma") {
        # estimate parameters
        if (is.null(D)) {
            D <- floor(min(sqrt(n), p))
        }

        lfun <- function(t, v) {
            -sum(dgamma(v, shape = t[1], scale = t[2], log = TRUE))
        }

        t0.mle <- t(sapply(1:p, function(i) {
            ai <- mean(xtrain[ytrain == 0, i])^2/var(xtrain[ytrain == 0, i])
            bi <- var(xtrain[ytrain == 0, i])/mean(xtrain[ytrain == 0, i])
            suppressWarnings(nlm(lfun, p = c(ai, bi), v = xtrain[ytrain == 0, i])$estimate)
        }))

        t1.mle <- t(sapply(1:p, function(i) {
            ai <- mean(xtrain[ytrain == 1, i])^2/var(xtrain[ytrain == 1, i])
            bi <- var(xtrain[ytrain == 1, i])/mean(xtrain[ytrain == 1, i])
            suppressWarnings(nlm(lfun, p = c(ai, bi), v = xtrain[ytrain == 1, i])$estimate)
        }))

        # start loops
        dist <- rep(1, p)
        for (t in 1:(iteration + 1)) {
            output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
                S <- sapply(1:B2, function(j) {
                  S.size <- sample(1:D, 1)
                  c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=
                    0]))))
                })
                RaSubset(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, base = base, k = k,
                  criterion = criterion, cv = cv, t0.mle = t0.mle, t1.mle = t1.mle, kl.k = kl.k, ...)
            }

            subspace <- output[, 3]

            s <- rep(0, p)
            for (i in 1:length(subspace)) {
                s[subspace[[i]]] <- s[subspace[[i]]] + 1
            }

            dist <- s/sum(s)
            dist[dist < C0/log(p)] <- C0/p
            ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))
        }

        fit.list <- output[, 1]
    }


    # -------------------------------


    p0 <- sum(ytrain == 0)/nrow(xtrain)
    if (cutoff == TRUE) {
        cutoff <- RaCutoff(ytrain.pred, ytrain, p0)
    } else {
        cutoff <- 0.5
    }

    if (ranking == TRUE) {
        rk <- s/B1*100
        names(rk) <- 1:length(rk)
    } else {
        rk <- NULL
    }

    if (scale == TRUE) {
        scale.parameters <- list(center = scale.center, scale = scale.scale)
    } else {
        scale.parameters <- NULL
    }

    stopImplicitCluster()
    obj <- list(marginal = c(`class 0` = p0, `class 1` = 1 - p0), base = base, criterion = criterion, B1 = B1, B2 = B2,
                iteration = iteration, fit.list = fit.list, cutoff = cutoff, subspace = subspace, ranking = rk, scale = scale.parameters, C0 = C0)
    class(obj) <- "RaSE"

    return(obj)
}

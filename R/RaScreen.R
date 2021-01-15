#' Variable screening via RaSE.
#'
#' \code{RaSE} is a general framework for variable screening. In RaSE screening, to select each of the B1 subspaces, B2 random subspaces are generated and the optimal one is chosen according to some criterion. Then the selected proportions (equivalently, percentages) of variables in the B1 subspaces are used as importance measure to rank these variables.
#' @export
#' @param xtrain n * p observation matrix. n observations, p features.
#' @param ytrain n 0/1 observatons.
#' @param xval observation matrix for validation. Default = \code{NULL}. Useful only when \code{criterion} = 'validation'.
#' @param yval 0/1 observation for validation. Default = \code{NULL}. Useful only when \code{criterion} = 'validation'.
#' @param B1 the number of weak learners. Default = 200.
#' @param B2 the number of subspace candidates generated for each weak learner. Default = \eqn{floor(p^{1.1})}.
#' @param D the maximal subspace size when generating random subspaces. Default = \code{NULL}. It means that \code{D} = \eqn{min(\sqrt n0, \sqrt n1, p)} when \code{model} = 'qda', \code{D} = \eqn{min(n^{0.49}, p)} when \code{model} = 'lm', and \code{D} = \eqn{min(\sqrt n, p)} otherwise.
#' @param dist the distribution for features when generating random subspaces. Default = \code{NULL}, which represents the hierarchical uniform distribution. First generate an integer \eqn{d} from \eqn{1,...,D} uniformly, then uniformly generate a subset with cardinality \eqn{d}.
#' @param model the model to use. Default = 'lda' when \code{classification} = TRUE and 'lm' when \code{classification} = FALSE.
#' \itemize{
#' \item lm: linear regression. Only available for regression.
#' \item lda: linear discriminant analysis. \code{\link[MASS]{lda}} in \code{MASS} package. Only available for classification.
#' \item qda: quadratic discriminant analysis. \code{\link[MASS]{qda}} in \code{MASS} package. Only available for classification.
#' \item knn: k-nearest neighbor. \code{\link[class]{knn}}, \code{\link[class]{knn.cv}} in \code{class} package, \code{\link[caret]{knn3}} in \code{caret} package and \code{\link[caret]{knnreg}} in \code{caret} package.
#' \item logistic: logistic regression. \code{\link[glmnet]{glmnet}} in \code{glmnet} package. Only available for classification.
#' \item tree: decision tree. \code{\link[rpart]{rpart}} in \code{rpart} package. Only available for classification.
#' \item svm: support vector machine. If kernel is not identified by user, it will use RBF kernel. \code{\link[e1071]{svm}} in \code{e1071} package.
#' \item randomforest: random forest. \code{\link[randomForest]{randomForest}} in \code{randomForest} package and \code{\link[ranger]{ranger}} in \code{ranger} package.
#' }
#' @param criterion the criterion to choose the best subspace. Default = 'ric' when \code{model} = 'lda', 'qda'; default = 'bic' when \code{model} = 'lm' or 'logistic'; default = 'loo' when \code{model} = 'knn'; default = 'cv' and set \code{cv} = 5 when \code{model} = 'tree', 'svm', 'randomforest'.
#' \itemize{
#' \item ric: minimizing ratio information criterion (RIC) with parametric estimation (Tian, Y. and Feng, Y., 2020). Available for binary classification and \code{model} = 'lda', 'qda', or 'logistic'.
#' \item nric: minimizing ratio information criterion (RIC) with non-parametric estimation (Tian, Y. and Feng, Y., 2020; ). Available for binary classification and \code{model} = 'lda', 'qda', or 'logistic'.
#' \item training: minimizing training error/MSE. Not available when \code{model} = 'knn'.
#' \item loo: minimizing leave-one-out error/MSE. Only available when  \code{model} = 'knn'.
#' \item validation: minimizing validation error/MSE based on the validation data.
#' \item cv: minimizing k-fold cross-validation error/MSE. k equals to the value of \code{cv}. Default = 5.
#' \item aic: minimizing Akaike information criterion (Akaike, H., 1973). Available when \code{base} = 'lm' or 'logistic'.
#'
#' AIC = -2 * log-likelihood + |S| * 2.
#'
#' \item bic: minimizing Bayesian information criterion (Schwarz, G., 1978). Available when \code{model} = 'lm' or 'logistic'.
#'
#' BIC = -2 * log-likelihood + |S| * log(n).
#'
#' \item ebic: minimizing extended Bayesian information criterion (Chen, J. and Chen, Z., 2008; 2012). \code{gam} value is needed. When \code{gam} = 0, it represents BIC. Available when \code{model} = 'lm' or 'logistic'.
#'
#' eBIC = -2 * log-likelihood + |S| * log(n) + 2 * |S| * gam * log(p).
#' }
#' @param k the number of nearest neightbors considered when \code{model} = 'knn'. Only useful when \code{model} = 'knn'. Default = 5.
#' @param cores the number of cores used for parallel computing. Default = 1.
#' @param seed the random seed assigned at the start of the algorithm, which can be a real number or \code{NULL}. Default = \code{NULL}, in which case no random seed will be set.
#' @param iteration the number of iterations. Default = 0.
#' @param cv the number of cross-validations used. Default = 5. Only useful when \code{criterion} = 'cv'.
#' @param scale whether to normalize the data. Logistic, default = FALSE.
#' @param C0 a positive constant used when \code{iteration} > 1. See Tian, Y. and Feng, Y., 2021 for details. Default = 0.1.
#' @param kl.k the number of nearest neighbors used to estimate RIC in a non-parametric way. Default = \code{NULL}, which means that \eqn{k0 = floor(\sqrt n0)} and \eqn{k1 = floor(\sqrt n1)}. See Tian, Y. and Feng, Y., 2020 for details. Only available when \code{criterion} = 'nric'.
#' @param classification the indicator of the problem type, which can be TRUE, FALSE or \code{NULL}. Default = \code{NULL}, which will automatically set \code{classification} = TRUE if the number of unique response value \eqn{\le} 10. Otherwise, it will be set as FALSE.
#' @param ... additional arguments.
#' @return A list including the following items.
#' \item{model}{the model used in RaSE screening.}
#' \item{criterion}{the criterion to choose the best subspace for each weak learner.}
#' \item{B1}{the number of selected subspaces.}
#' \item{B2}{the number of subspace candidates generated for each of B1 subspaces.}
#' \item{n}{the sample size.}
#' \item{p}{the dimension of data.}
#' \item{D}{the maximal subspace size when generating random subspaces.}
#' \item{iteration}{the number of iterations.}
#' \item{selected.perc}{A list of length (\code{iteration}+1) recording the selected percentages of each feature in B1 subspaces. When it is of length 1, the result will be automatically transformed to a vector.}
#' \item{scale}{a list of scaling parameters, including the scaling center and the scale parameter for each feature. Equals to \code{NULL} when the data is not scaled by \code{RaScreen}.}
#' @seealso \code{\link{Rase}}, \code{\link{RaRank}}.
#' @references
#' Tian, Y. and Feng, Y., 2021. RaSE: A Variable Screening Framework via Random Subspace Ensembles.
#'
#' Tian, Y. and Feng, Y., 2021. RaSE: Random subspace ensemble classification. Journal of Machine Learning Research, 22, to appear.
#'
#' Chen, J. and Chen, Z., 2008. Extended Bayesian information criteria for model selection with large model spaces. Biometrika, 95(3), pp.759-771.
#'
#' Chen, J. and Chen, Z., 2012. Extended BIC for small-n-large-P sparse GLM. Statistica Sinica, pp.555-574.
#'
#' Schwarz, G., 1978. Estimating the dimension of a model. The annals of statistics, 6(2), pp.461-464.
#'
#' @examples
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' train.data <- RaModel("screening", 1, n = 100, p = 100)
#' xtrain <- train.data$x
#' ytrain <- train.data$y
#'
#' # test RaSE screening with linear regression model and BIC
#' fit <- RaScreen(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, model = 'lm',
#' cores = 2, criterion = 'bic')
#'
#' # Select D variables
#' RaRank(fit, selected.num = "D")
#'
#'
#' \dontrun{
#' # test RaSE screening with knn model and 5-fold cross-validation MSE
#' fit <- RaScreen(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, model = 'knn',
#' cores = 2, criterion = 'cv', cv = 5)
#'
#' # Select n/logn variables
#' RaRank(fit, selected.num = "n/logn")
#'
#'
#' # test RaSE screening with SVM and 5-fold cross-validation MSE
#' fit <- RaScreen(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, model = 'svm',
#' cores = 2, criterion = 'cv', cv = 5)
#'
#' # Select n/logn variables
#' RaRank(fit, selected.num = "n/logn")
#'
#'
#' # test RaSE screening with logistic regression model and eBIC (gam = 0.5). Set iteration number = 1
#' train.data <- RaModel("screening", 6, n = 100, p = 100)
#' xtrain <- train.data$x
#' ytrain <- train.data$y
#'
#' fit <- RaScreen(xtrain, ytrain, B1 = 100, B2 = 100, iteration = 1, model = 'logistic',
#' cores = 2, criterion = 'ebic', gam = 0.5)
#'
#' # Select n/logn variables from the selected percentage after one iteration round
#' RaRank(fit, selected.num = "n/logn", iteration = 1)
#' }


RaScreen <- function(xtrain, ytrain, xval = NULL, yval = NULL, B1 = 200, B2 = floor(ncol(xtrain)^1.1), D = NULL, dist = NULL, model = NULL, criterion = NULL, k = 5, cores = 1,
                         seed = NULL, iteration = 0, cv = 5, scale = FALSE, C0 = 0.1, kl.k = NULL, classification = NULL, ...) {
  if (!is.null(seed)) {
    set.seed(seed, kind = "L'Ecuyer-CMRG")
  }

  if (is.null(classification)) {
    classification <- ifelse(length(unique(ytrain)) > 10, FALSE, TRUE)
  }

  if(is.null(model)){
    model <- ifelse(classification, "lda", "lm")
  }

  if (is.null(criterion)) {
    if (model == "lda" || model == "qda" || model == "gamma") {
      criterion <- "ric"
    } else if (model == "logistic") {
      criterion <- "bic"
    } else if (model == "knn") {
      criterion <- "loo"
    } else if  (model == "lm"){
      criterion <- "bic"
    } else {
      criterion <- "cv"
      cv <- 5
    }
  }


  if (classification) {
    n0 <- sum(ytrain == 0)
    n1 <- sum(ytrain == 1)
  }

  if(classification && is.null(kl.k)) {
    kl.k <- floor(sqrt(as.vector(table(ytrain))))
  }

  xtrain <- as.matrix(xtrain)
  p <- ncol(xtrain)
  n <- length(ytrain)


  if (scale == TRUE) {
    L <- scale_Rase(xtrain)
    xtrain <- L$data
    scale.center <- L$center
    scale.scale <- L$scale
  }

  selected.perc <- numeric(p)
  selected.perc <- rep(list(selected.perc), iteration+1)

  registerDoParallel(cores)

  if (model == "lm") {
    if (is.null(D)) {
      D <- floor(n^(0.49))
    }
    XX <- t(xtrain) %*% xtrain
    XY <- t(xtrain) %*% ytrain
    if (is.null(dist)) {
      dist <- rep(1, p)
    }
    for (t in 1:(iteration + 1)) {
      output <- foreach(i = 1:B1, .combine = "rbind") %dopar% {
          S <- sapply(1:B2, function(j) {
            S.size <- sample(1:D, size = 1)
            c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=0]))))
          })


        RaSubsetsc_rg(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, model = model, k = k,
                 criterion = criterion, cv = cv, XX = XX, XY = XY, ...)
      }

      subspace <- output

      s <- rep(0, p)
      for (i in 1:length(subspace)) {
        s[subspace[[i]]] <- s[subspace[[i]]] + 1
      }

      dist <- s/B1
      selected.perc[[t]] <- dist
      dist[dist < C0/log(p)] <- C0/p

    }

  }



  if (model == "lda") {
    class.no <- length(unique(ytrain))
    nc <- as.numeric(table(ytrain))
    # remove redundant features
    if (p <= 5000) {
      # clean data
      a <- suppressWarnings(cor(xtrain))
      b <- a - diag(diag(a))
      b0 <- which(abs(b) > 0.9999, arr.ind = T)
      b0 <- matrix(b0[b0[, 1] > b0[, 2], ], ncol = 2)
      a <- diag(cov(xtrain))

      delete.ind <- unique(c(which(a == 0), b0[, 1]))
      sig.ind <- setdiff(1:p, delete.ind)
    } else {
      sig.ind <- 1:p
      delete.ind <- NULL
    }

    if (is.null(D)) {
      D <- floor(min(n^(0.49), length(sig.ind)))
    }

    # estimate parameters
    if (criterion == "ric" && p <= 5000 || criterion != "ric"){
      # Sigma.mle <- sapply(1:class.no, function(i){
      #   cov(as.matrix(xtrain[ytrain == i-1, , drop = F]))*(nc[i]-1)/n
      # }, simplify = F)
      # Sigma.mle <- Reduce("+", Sigma.mle)
      # mu.mle <- sapply(1:class.no, function(i){
      #   colMeans(as.matrix(xtrain[ytrain == i-1, , drop = F]), na.rm = TRUE)
      # }, simplify = F)

      Sigma.mle <- ((n0 - 1) * cov(xtrain[ytrain == 0, , drop = F]) + (n1 - 1) * cov(xtrain[ytrain == 1, , drop = F]))/n
      mu0.mle <- colMeans(xtrain[ytrain == 0, , drop = F])
      mu1.mle <- colMeans(xtrain[ytrain == 1, , drop = F])
    } else {
      mu.mle <- sapply(1:class.no, function(i){
        colMeans(as.matrix(xtrain[ytrain == i-1, , drop = F]), na.rm = TRUE)
      }, simplify = F)
      Sigma.mle <- NULL
    }

    # start loops
    if (is.null(dist)) {
      dist <- rep(1, p)
    }
    dist[delete.ind] <- 0
      for (t in 1:(iteration + 1)) {
        if (p <= 5000){
          output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
            S <- sapply(1:B2, function(j) {
              S.size <- sample(1:D, 1)
              c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist != 0]))))
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
            RaSubsetsc_c(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, model = model, k = k,
                         criterion = criterion, cv = cv, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma.mle = Sigma.mle, kl.k = kl.k,...)
          }

        } else {
          output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
            S <- sapply(1:B2, function(j) {
              S.size <- sample(1:D, 1)
              c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist != 0]))))
            })
            S <- sapply(1:B2, function(j) {
              snew <- S[!is.na(S[, j]), j]
              if (length(snew) >= 2) {
                ind0 <- findLinearCombos(xtrain[, snew, drop = F])$remove
                if (!is.null(ind0)) {
                  snew <- snew[-ind0]
                }
              }
              c(snew, rep(NA, D - length(snew)))
            })
            RaSubsetsc_c(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, model = model, k = k,
                         criterion = criterion, cv = cv, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma.mle = Sigma.mle, kl.k = kl.k,...)
          }
        }

        subspace <- output

        s <- rep(0, p)
        for (i in 1:length(subspace)) {
          s[subspace[[i]]] <- s[subspace[[i]]] + 1
        }

        dist <- s/B1
        selected.perc[[t]] <- dist
        dist[dist < C0/log(p)] <- C0/p
      }
      fit.list <- output[, 1]

  }

  if (model == "qda") {
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

    if (is.null(D)) {
      D <- floor(min(sqrt(n0), sqrt(n1), length(sig.ind)))
    }

    if (criterion == "ric" && p <= 7000) {
      Sigma0.mle <- (n0 - 1)/n0 * cov(xtrain[ytrain == 0, , drop = F])
      Sigma1.mle <- (n1 - 1)/n1 * cov(xtrain[ytrain == 1, , drop = F])
      mu0.mle <- colMeans(xtrain[ytrain == 0, , drop = F])
      mu1.mle <- colMeans(xtrain[ytrain == 1, , drop = F])
    } else {
      Sigma0.mle <- NULL
      Sigma1.mle <- NULL
      mu0.mle <- colMeans(xtrain[ytrain == 0, , drop = F])
      mu1.mle <- colMeans(xtrain[ytrain == 1, , drop = F])
    }
    # estimate parameters


    # start loops
    if (is.null(dist)) {
      dist <- rep(1, p)
    }
    dist[delete.ind] <- 0
    if (p <= 7000) {
      for (t in 1:(iteration + 1)) {
        output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
          S <- sapply(1:B2, function(j) {
            S.size <- sample(1:D, 1)
            c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist != 0]))))
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
          RaSubsetsc_c(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, model = model, k = k,
                       criterion = criterion, cv = cv, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma0.mle = Sigma0.mle, Sigma1.mle = Sigma1.mle,
                       ...)
        }

        subspace <- output

        s <- rep(0, p)
        for (i in 1:length(subspace)) {
          s[subspace[[i]]] <- s[subspace[[i]]] + 1
        }

        dist <- s/B1
        selected.perc[[t]] <- dist
        dist[dist < C0/log(p)] <- C0/p

      }
    } else {
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
              ind0 <- findLinearCombos(xtrain[ytrain == 0, snew, drop = F])$remove
              ind1 <- findLinearCombos(xtrain[ytrain == 1, snew, drop = F])$remove
              if (!all(is.null(c(ind0, ind1)))) {
                snew <- snew[-c(ind0, ind1)]
              }
            }
            c(snew, rep(NA, D - length(snew)))
          })
          RaSubsetsc_c(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, model = model, k = k,
                       criterion = criterion, cv = cv, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma0.mle = Sigma0.mle, Sigma1.mle = Sigma1.mle,
                       ...)
        }

        subspace <- output

        s <- rep(0, p)
        for (i in 1:length(subspace)) {
          s[subspace[[i]]] <- s[subspace[[i]]] + 1
        }

        dist <- s/B1
        selected.perc[[t]] <- dist
        dist[dist < C0/log(p)] <- C0/p

      }
    }



  }

  if (model == "knn") {
    # estimate parameters
    if (is.null(D)) {
      D <- floor(min(floor(n^(0.49)), p))
    }

    # start loops
    if (is.null(dist)) {
      dist <- rep(1, p)
    }

    if (classification) {
      for (t in 1:(iteration + 1)) {
        output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
          S <- sapply(1:B2, function(j) {
            S.size <- sample(1:D, 1)
            c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=
                                                                                                                              0]))))
          })
          RaSubsetsc_c(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, model = model, k = k,
                       criterion = criterion, cv = cv, ...)
        }

        subspace <- output

        s <- rep(0, p)
        for (i in 1:length(subspace)) {
          s[subspace[[i]]] <- s[subspace[[i]]] + 1
        }

        dist <- s/B1
        selected.perc[[t]] <- dist
        dist[dist < C0/log(p)] <- C0/p
      }
    } else {
      for (t in 1:(iteration + 1)) {
        output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
          S <- sapply(1:B2, function(j) {
            S.size <- sample(1:D, 1)
            c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=
                                                                                                                              0]))))
          })
          RaSubsetsc_rg(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, model = model, k = k,
                       criterion = criterion, cv = cv, ...)
        }

        subspace <- output

        s <- rep(0, p)
        for (i in 1:length(subspace)) {
          s[subspace[[i]]] <- s[subspace[[i]]] + 1
        }

        dist <- s/B1
        selected.perc[[t]] <- dist
        dist[dist < C0/log(p)] <- C0/p
      }
    }

  }

  if (model == "tree") {
    # estimate parameters
    if (is.null(D)) {
      D <- floor(min(sqrt(n), p))
    }

    # start loops
    if (is.null(dist)) {
      dist <- rep(1, p)
    }
    for (t in 1:(iteration + 1)) {
      output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
        S <- sapply(1:B2, function(j) {
          S.size <- sample(1:D, 1)
          c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=
                                                                                                                            0]))))
        })
        RaSubsetsc_c(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, model = model, k = k,
                 criterion = criterion, cv = cv, ...)
      }

      subspace <- output
      s <- rep(0, p)
      for (i in 1:length(subspace)) {
        s[subspace[[i]]] <- s[subspace[[i]]] + 1
      }
      dist <- s/B1
      selected.perc[[t]] <- dist
      dist[dist < C0/log(p)] <- C0/p
    }

    fit.list <- output[, 1]
  }

  if (model == "logistic" || model == "svm" || model == "randomforest") {
    # estimate parameters
    if (is.null(D)) {
      D <- floor(min(floor(n^(0.49)), p))
    }

    # start loops
    if(classification) {
      if (is.null(dist)) {
        dist <- rep(1, p)
      }
      for (t in 1:(iteration + 1)) {
        output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
          S <- sapply(1:B2, function(j) {
            S.size <- sample(1:D, 1)
            c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=
                                                                                                                              0]))))
          })
          RaSubsetsc_c(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, model = model, k = k,
                       criterion = criterion, cv = cv, ...)
        }

        subspace <- output

        s <- rep(0, p)
        for (i in 1:length(subspace)) {
          s[subspace[[i]]] <- s[subspace[[i]]] + 1
        }

        # dist <- s/sum(s)
        dist <- s/B1
        selected.perc[[t]] <- dist
        dist[dist < C0/log(p)] <- C0/p
      }
    } else {
      if (is.null(dist)) {
        dist <- rep(1, p)
      }
      for (t in 1:(iteration + 1)) {
        output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
          S <- sapply(1:B2, function(j) {
            S.size <- sample(1:D, 1)
            c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=
                                                                                                                              0]))))
          })
          RaSubsetsc_rg(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, model = model, k = k,
                       criterion = criterion, cv = cv, ...)
        }

        subspace <- output

        s <- rep(0, p)
        for (i in 1:length(subspace)) {
          s[subspace[[i]]] <- s[subspace[[i]]] + 1
        }

        # dist <- s/sum(s)
        dist <- s/B1
        selected.perc[[t]] <- dist
        dist[dist < C0/log(p)] <- C0/p
      }
    }


  }



  # -------------------------------


  # rk <- s/B1*100
  # names(rk) <- 1:length(rk)

  for (t in 1:(iteration+1)) {
    names(selected.perc[[t]]) <- 1:p
    selected.perc[[t]] <- selected.perc[[t]]*100
  }

  stopImplicitCluster()
  if (iteration == 0) {
    selected.perc <- selected.perc[[1]]
  }

  if (scale == TRUE) {
    scale.parameters <- list(center = scale.center, scale = scale.scale)
  } else {
    scale.parameters <- NULL
  }

  obj <- list(model = model, criterion = criterion, B1 = B1, B2 = B2, n = n, p = p, D = D, iteration = iteration, selected.perc = selected.perc, scale = scale.parameters)


  return(obj)

}

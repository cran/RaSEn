#' Construct the random subspace ensemble classifier.
#'
#' \code{RaSE} is a general ensemble classification framework to solve the sparse classification problem. In RaSE algorithm, for each of the B1 weak learners, B2 random subspaces are generated and the optimal one is chosen to train the model on the basis of some criterion.
#' @export
#' @importFrom MASS lda
#' @importFrom MASS qda
#' @importFrom MASS mvrnorm
#' @importFrom class knn
#' @importFrom class knn.cv
#' @importFrom caret knn3
#' @importFrom caret createFolds
#' @importFrom caret findLinearCombos
#' @importFrom caret knnreg
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
#' @importFrom stats rnorm
#' @importFrom stats rt
#' @importFrom stats runif
#' @importFrom stats deviance
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
#' @importFrom ranger ranger
#' @importFrom KernelKnn KernelKnn
#' @importFrom utils data
#' @importFrom glmnet glmnet
#' @importFrom glmnet predict.glmnet
#' @importFrom ModelMetrics auc
#' @param xtrain n * p observation matrix. n observations, p features.
#' @param ytrain n 0/1 observatons.
#' @param xval observation matrix for validation. Default = \code{NULL}. Useful only when \code{criterion} = 'validation'.
#' @param yval 0/1 observation for validation. Default = \code{NULL}. Useful only when \code{criterion} = 'validation'.
#' @param B1 the number of weak learners. Default = 200.
#' @param B2 the number of subspace candidates generated for each weak learner. Default = 500.
#' @param D the maximal subspace size when generating random subspaces. Default = \code{NULL}, which is \eqn{min(\sqrt n0, \sqrt n1, p)} when \code{base} = 'qda' and is \eqn{min(\sqrt n, p)} otherwise. For classical RaSE with a single classifier type, \code{D} is a positive integer. For super RaSE with multiple classifier types, \code{D} is a vector indicating different D values used for each base classifier type (the corresponding classifier types should be noted in the names of the vector).
#' @param dist the distribution for features when generating random subspaces. Default = \code{NULL}, which represents the uniform distribution. First generate an integer \eqn{d} from \eqn{1,...,D} uniformly, then uniformly generate a subset with cardinality \eqn{d}.
#' @param base the type of base classifier. Default = 'lda'. Can be either a single string chosen from the following options or a string/probability vector. When it indicates a single type of base classifiers, the classical RaSE model (Tian, Y. and Feng, Y., 2021(b)) will be fitted. When it is a string vector which includes multiple base classifier types, a super RaSE model (Zhu, J. and Feng, Y., 2021) will be fitted, by samling base classifiers with equal probabilty. It can also be a probability vector with row names corresponding to the specific classifier type, in which case a super RaSE model will be trained by sampling base classifiers in the given sampling probability.
#' \itemize{
#' \item lda: linear discriminant analysis. \code{\link[MASS]{lda}} in \code{MASS} package.
#' \item qda: quadratic discriminant analysis. \code{\link[MASS]{qda}} in \code{MASS} package.
#' \item knn: k-nearest neighbor. \code{\link[class]{knn}}, \code{\link[class]{knn.cv}} in \code{class} package and \code{\link[caret]{knn3}} in \code{caret} package.
#' \item logistic: logistic regression. \code{\link[stats]{glm}} in \code{stats} package and \code{\link[glmnet]{glmnet}} in \code{glmnet} package.
#' \item tree: decision tree. \code{\link[rpart]{rpart}} in \code{rpart} package.
#' \item svm: support vector machine. \code{\link[e1071]{svm}} in \code{e1071} package.
#' \item randomforest: random forest. \code{\link[randomForest]{randomForest}} in \code{randomForest} package.
#' \item gamma: Bayesian classifier for multivariate gamma distribution with independent marginals.
#' }
#' @param super a list of control parameters for super RaSE (Zhu, J. and Feng, Y., 2021). Not used when base equals to a single string. Should be a list object with the following components:
#' \itemize{
#' \item type: the type of super RaSE. Currently the only option is 'separate', meaning that subspace distributions are different for each type of base classifiers.
#' \item base.update: indicates whether the sampling probability of base classifiers should be updated during iterations or not. Logistic, default = TRUE.
#' }
#' @param criterion the criterion to choose the best subspace for each weak learner. For the classical RaSE (when \code{base} includes a single classifier type), default = 'ric' when \code{base} = 'lda', 'qda', 'gamma'; default = 'ebic' and set \code{gam} = 0 when \code{base} = 'logistic'; default = 'loo' when \code{base} = 'knn'; default = 'training' when \code{base} = 'tree', 'svm', 'randomforest'. For the super RaSE (when \code{base} indicates multiple classifiers or the sampling probability of multiple classifiers), default = 'cv' with the number of folds \code{cv} = 5, and it can only be 'cv', 'training' or 'auc'.
#' \itemize{
#' \item ric: minimizing ratio information criterion with parametric estimation (Tian, Y. and Feng, Y., 2021(b)). Available when \code{base} = 'lda', 'qda', 'gamma' or 'logistic'.
#' \item nric: minimizing ratio information criterion with non-parametric estimation (Tian, Y. and Feng, Y., 2021(b)). Available when \code{base} = 'lda', 'qda', 'gamma' or 'logistic'.
#' \item training: minimizing training error. Not available when \code{base} = 'knn'.
#' \item loo: minimizing leave-one-out error. Only available when  \code{base} = 'knn'.
#' \item validation: minimizing validation error based on the validation data. Available for all base classifiers.
#' \item auc: minimizing negative area under the ROC curve (AUC). Currently it is estimated on training data via function \code{\link[ModelMetrics]{auc}} from package \code{ModelMetrics}. It is available for all classier choices.
#' \item cv: minimizing k-fold cross-validation error. k equals to the value of \code{cv}. Default = 5. Not available when \code{base} = 'gamma'.
#' \item aic: minimizing Akaike information criterion (Akaike, H., 1973). Available when \code{base} = 'lda' or 'logistic'.
#'
#' AIC = -2 * log-likelihood + |S| * 2.
#'
#' \item bic: minimizing Bayesian information criterion (Schwarz, G., 1978). Available when \code{base} = 'lda' or 'logistic'.
#'
#' BIC = -2 * log-likelihood + |S| * log(n).
#'
#' \item ebic: minimizing extended Bayesian information criterion (Chen, J. and Chen, Z., 2008; 2012). Need to assign value for \code{gam}. When \code{gam} = 0, it denotes the classical BIC. Available when \code{base} = 'lda' or 'logistic'.
#'
#' EBIC = -2 * log-likelihood + |S| * log(n) + 2 * |S| * gam * log(p).
#'
#' }
#' @param ranking whether the function outputs the selected percentage of each feature in B1 subspaces. Logistic, default = TRUE.
#' @param k the number of nearest neightbors considered when \code{base} = 'knn'. Only useful when \code{base} = 'knn'. Default = (3, 5, 7, 9, 11).
#' @param cores the number of cores used for parallel computing. Default = 1.
#' @param seed the random seed assigned at the start of the algorithm, which can be a real number or \code{NULL}. Default = \code{NULL}, in which case no random seed will be set.
#' @param iteration the number of iterations. Default = 0.
#' @param cutoff whether to use the empirically optimal threshold. Logistic, default = TRUE. If it is FALSE, the threshold will be set as 0.5.
#' @param cv the number of cross-validations used. Default = 5. Only useful when \code{criterion} = 'cv'.
#' @param scale whether to normalize the data. Logistic, default = FALSE.
#' @param C0 a positive constant used when \code{iteration} > 1. Default = 0.1. See Tian, Y. and Feng, Y., 2021(b) for details.
#' @param kl.k the number of nearest neighbors used to estimate RIC in a non-parametric way. Default = \code{NULL}, which means that \eqn{k0 = floor(\sqrt n0)} and \eqn{k1 = floor(\sqrt n1)}. See Tian, Y. and Feng, Y., 2021(b) for details. Only available when \code{criterion} = 'nric'.
#' @param lower.limits the vector of lower limits for each coefficient in logistic regression. Should be a vector of length equal to the number of variables (the column number of \code{xtrain}). Each of these must be non-positive. Default = \code{NULL}, meaning that lower limits are \code{-Inf} for all coefficients. Only available when \code{base} = 'logistic'. When it's activated, function \code{\link[glmnet]{glmnet}} will be used to fit logistic regression models, in which case the minimum subspace size is required to be larger than 1. The default subspace size distribution will be changed to uniform distribution on (2, ..., D).
#' @param upper.limits the vector of upper limits for each coefficient in logistic regression. Should be a vector of length equal to the number of variables (the column number of \code{xtrain}). Each of these must be non-negative. Default = \code{NULL}, meaning that upper limits are \code{Inf} for all coefficients. Only available when \code{base} = 'logistic'. When it's activated, function \code{\link[glmnet]{glmnet}} will be used to fit logistic regression models, in which case the minimum subspace size is required to be larger than 1. The default subspace size distribution will be changed to uniform distribution on (2, ..., D).
#' @param weights observation weights. Should be a vector of length equal to training sample size (the length of \code{ytrain}). It will be normailized inside the algorithm. Each component of weights must be non-negative. Default is \code{NULL}, representing equal weight for each observation. Only available when \code{base} = 'logistic'. When it's activated, function \code{\link[glmnet]{glmnet}} will be used to fit logistic regression models, in which case the minimum subspace size is required to be larger than 1. The default subspace size distribution will be changed to uniform distribution on (2, ..., D).
#' @param ... additional arguments.
#' @return An object with S3 class \code{'RaSE'} if \code{base} indicates a single base classifier.
#' \item{marginal}{the marginal probability for each class.}
#' \item{base}{the type of base classifier.}
#' \item{criterion}{the criterion to choose the best subspace for each weak learner.}
#' \item{B1}{the number of weak learners.}
#' \item{B2}{the number of subspace candidates generated for each weak learner.}
#' \item{D}{the maximal subspace size when generating random subspaces.}
#' \item{iteration}{the number of iterations.}
#' \item{fit.list}{sequence of B1 fitted base classifiers.}
#' \item{cutoff}{the empirically optimal threshold.}
#' \item{subspace}{sequence of subspaces correponding to B1 weak learners.}
#' \item{ranking}{the selected percentage of each feature in B1 subspaces.}
#' \item{scale}{a list of scaling parameters, including the scaling center and the scale parameter for each feature. Equals to \code{NULL} when the data is not scaled in \code{RaSE} model fitting.}
#' An object with S3 class \code{'super_RaSE'} if \code{base} includes multiple base classifiers or the sampling probability of multiple classifiers.
#' \item{marginal}{the marginal probability for each class.}
#' \item{base}{the list of B1 base classifier types.}
#' \item{criterion}{the criterion to choose the best subspace for each weak learner.}
#' \item{B1}{the number of weak learners.}
#' \item{B2}{the number of subspace candidates generated for each weak learner.}
#' \item{D}{the maximal subspace size when generating random subspaces.}
#' \item{iteration}{the number of iterations.}
#' \item{fit.list}{sequence of B1 fitted base classifiers.}
#' \item{cutoff}{the empirically optimal threshold.}
#' \item{subspace}{sequence of subspaces correponding to B1 weak learners.}
#' \item{ranking.feature}{the selected percentage of each feature corresponding to each type of classifier.}
#' \item{ranking.base}{the selected percentage of each classifier type in the selected B1 learners.}
#' \item{scale}{a list of scaling parameters, including the scaling center and the scale parameter for each feature. Equals to \code{NULL} when the data is not scaled in \code{RaSE} model fitting.}
#' @author Ye Tian (maintainer, \email{ye.t@@columbia.edu}) and Yang Feng. The authors thank Yu Cao (Exeter Finance) and his team for many helpful suggestions and discussions.
#' @seealso \code{\link{predict.RaSE}}, \code{\link{RaModel}}, \code{\link{print.RaSE}}, \code{\link{print.super_RaSE}}, \code{\link{RaPlot}}, \code{\link{RaScreen}}.
#' @references
#' Tian, Y. and Feng, Y., 2021(a). RaSE: A variable screening framework via random subspace ensembles. Journal of the American Statistical Association, (just-accepted), pp.1-30.
#'
#' Tian, Y. and Feng, Y., 2021(b). RaSE: Random subspace ensemble classification. Journal of Machine Learning Research, 22(45), pp.1-93.
#'
#' Zhu, J. and Feng, Y., 2021. Super RaSE: Super Random Subspace Ensemble Classification. https://www.preprints.org/manuscript/202110.0042
#'
#' Chen, J. and Chen, Z., 2008. Extended Bayesian information criteria for model selection with large model spaces. Biometrika, 95(3), pp.759-771.
#'
#' Chen, J. and Chen, Z., 2012. Extended BIC for small-n-large-P sparse GLM. Statistica Sinica, pp.555-574.
#'
#' Akaike, H., 1973. Information theory and an extension of the maximum likelihood principle. In 2nd International Symposium on Information Theory, 1973 (pp. 267-281). Akademiai Kaido.
#'
#' Schwarz, G., 1978. Estimating the dimension of a model. The annals of statistics, 6(2), pp.461-464.
#'
#' @examples
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' train.data <- RaModel("classification", 1, n = 100, p = 50)
#' test.data <- RaModel("classification", 1, n = 100, p = 50)
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
#' # test RaSE classifier with kNN base classifier
#' fit <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, base = 'knn',
#' cores = 2, criterion = 'loo')
#' mean(predict(fit, xtest) != ytest)
#'
#' # test RaSE classifier with logistic regression base classifier
#' fit <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, base = 'logistic',
#' cores = 2, criterion = 'bic')
#' mean(predict(fit, xtest) != ytest)
#'
#' # test RaSE classifier with SVM base classifier
#' fit <- Rase(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, base = 'svm',
#' cores = 2, criterion = 'training')
#' mean(predict(fit, xtest) != ytest)
#'
#' # test RaSE classifier with random forest base classifier
#' fit <- Rase(xtrain, ytrain, B1 = 20, B2 = 10, iteration = 0, base = 'randomforest',
#' cores = 2, criterion = 'cv', cv = 3)
#' mean(predict(fit, xtest) != ytest)
#'
#' # fit a super RaSE classifier by sampling base learner from kNN, LDA and logistic
#' # regression in equal probability
#' fit <- Rase(xtrain = xtrain, ytrain = ytrain, B1 = 100, B2 = 100,
#' base = c("knn", "lda", "logistic"), super = list(type = "separate", base.update = T),
#' criterion = "cv", cv = 5, iteration = 1, cores = 2)
#' mean(predict(fit, xtest) != ytest)
#'
#' # fit a super RaSE classifier by sampling base learner from random forest, LDA and
#' # SVM with probability 0.2, 0.5 and 0.3
#' fit <- Rase(xtrain = xtrain, ytrain = ytrain, B1 = 100, B2 = 100,
#' base = c(randomforest = 0.2, lda = 0.5, svm = 0.3),
#' super = list(type = "separate", base.update = F),
#' criterion = "cv", cv = 5, iteration = 0, cores = 2)
#' mean(predict(fit, xtest) != ytest)
#' }

Rase <- function(xtrain, ytrain, xval = NULL, yval = NULL, B1 = 200, B2 = 500, D = NULL, dist = NULL, base = NULL, super = list(type = c("separate"), base.update = TRUE), criterion = NULL, ranking = TRUE, k = c(3, 5, 7, 9, 11), cores = 1,
    seed = NULL, iteration = 0, cutoff = TRUE, cv = 5, scale = FALSE, C0 = 0.1, kl.k = NULL, lower.limits = NULL, upper.limits = NULL, weights = NULL, ...) {

    if (!is.null(seed)) {
        set.seed(seed, kind = "L'Ecuyer-CMRG")
    }

    if (is.null(base)) {
        base <- "lda"
    }

    xtrain <- as.matrix(xtrain)
    base.dist <- NULL
    super$type <- super$type[1]
    if (length(base) > 1) { # super RaSE
        if (is.character(base)) {
            if (!all(base %in% c("lda", "qda", "knn", "logistic", "tree", "svm", "randomforest", "gamma"))) {stop("'base' can only be chosen from 'lda', 'qda', 'knn', 'logistic', 'tree', 'svm', 'randomforest' and 'gamma'!")}
            base.dist <- rep(1/length(base), length(base))
            names(base.dist) <- base
        } else {
            base.dist <- base
            base <- names(base)
        }
    }

    p <- ncol(xtrain)
    n <- length(ytrain)


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
    if (is.null(base.dist) && base == "lda") {
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
                  flag <- TRUE
                  while (flag) {
                      snew <- S[!is.na(S[, j]), j]
                      if (length(snew) > 2) {
                          ind0 <- findLinearCombos(Sigma.mle[snew, snew, drop = F])$remove
                          if (!is.null(ind0)) {
                              snew <- snew[-ind0]
                          }
                      }
                      snew1 <- c(snew, rep(NA, D - length(snew)))
                      if (any(abs(mu1.mle[snew1] - mu0.mle[snew1]) > 1e-10)) {
                          flag <- FALSE
                      }
                  }
                  snew1
                })

                 RaSubset(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, base = base, k = k,
                  criterion = criterion, cv = cv, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma.mle = Sigma.mle, kl.k = kl.k, ...)
            }

            if (is.matrix(output)) {
                subspace <- output[, 3]
            } else {
                subspace <- output[3]
            }

            s <- rep(0, p)
            for (i in 1:length(subspace)) {
                s[subspace[[i]]] <- s[subspace[[i]]] + 1
            }


            dist <- s/B1
            dist[dist < C0/log(p)] <- C0/p
            dist[delete.ind] <- 0

        }

        if (is.matrix(output)) {
            ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))
            fit.list <- output[, 1]
        } else {
            ytrain.pred <- data.frame(matrix(unlist(output[2]), ncol = B1))
            fit.list <- output[1]
        }

    }

    if (is.null(base.dist) && base == "qda") {
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

        # select.v <- NULL

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

            if (is.matrix(output)) {
                subspace <- output[, 3]
            } else {
                subspace <- output[3]
            }

            s <- rep(0, p)
            for (i in 1:length(subspace)) {
                s[subspace[[i]]] <- s[subspace[[i]]] + 1
            }

            dist <- s/B1
            dist[dist < C0/log(p)] <- C0/p
            dist[delete.ind] <- 0

            # ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))

        }

        # fit.list <- output[, 1]
        if (is.matrix(output)) {
            ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))
            fit.list <- output[, 1]
        } else {
            ytrain.pred <- data.frame(matrix(unlist(output[2]), ncol = B1))
            fit.list <- output[1]
        }
    }

    if (is.null(base.dist) && (base == "knn" || base == "tree" || base == "logistic" || base == "svm" || base == "randomforest")) {

        # estimate parameters
        if (is.null(D)) {
            D <- floor(min(sqrt(n), p))
        }

        if (all(is.null(lower.limits)) && all(is.null(upper.limits)) && base == "logistic" || criterion == "nric") {
            use.glmnet <- FALSE
        } else {
            use.glmnet <- TRUE
        }

        # start loops
        dist <- rep(1, p)
        for (t in 1:(iteration + 1)) {
            output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {

                S <- sapply(1:B2, function(j) {
                    if (use.glmnet) {
                        S.size <- sample(2:D, 1) # glmnet cannot fit the model with a single variable
                        if (length(dist[dist != 0]) == 1) {
                            stop ("Only one feature has positive sampling weights! 'glmnet' cannot be applied in this case! ")
                        }
                    } else {
                        S.size <- sample(1:D, 1)
                    }
                    c(sample(1:p, size = min(S.size, length(dist[dist != 0])), prob = dist), rep(NA, D - min(S.size, length(dist[dist !=
                                                                                                                                     0]))))
                })

                RaSubset(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, base = base, k = k, kl.k = kl.k,
                  criterion = criterion, cv = cv, lower.limits = lower.limits, upper.limits = upper.limits, weights = weights, gam = gam, ...)
            }

            if (is.matrix(output)) {
                subspace <- output[, 3]
            } else {
                subspace <- output[3]
            }

            s <- rep(0, p)
            for (i in 1:length(subspace)) {
                s[subspace[[i]]] <- s[subspace[[i]]] + 1
            }

            dist <- s/B1
            dist[dist < C0/log(p)] <- C0/p
        }

        if (is.matrix(output)) {
            ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))
            fit.list <- output[, 1]
        } else {
            ytrain.pred <- data.frame(matrix(unlist(output[2]), ncol = B1))
            fit.list <- output[1]
        }
    }


    if (is.null(base.dist) && base == "gamma") {
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
                  criterion = criterion, cv = cv, t0.mle = t0.mle, t1.mle = t1.mle, kl.k = kl.k, lower.limits = lower.limits,
                  upper.limits = upper.limits, weights = weights, ...)
            }

            if (is.matrix(output)) {
                subspace <- output[, 3]
            } else {
                subspace <- output[3]
            }

            s <- rep(0, p)
            for (i in 1:length(subspace)) {
                s[subspace[[i]]] <- s[subspace[[i]]] + 1
            }

            dist <- s/B1
            dist[dist < C0/log(p)] <- C0/p
        }

        if (is.matrix(output)) {
            ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))
            fit.list <- output[, 1]
        } else {
            ytrain.pred <- data.frame(matrix(unlist(output[2]), ncol = B1))
            fit.list <- output[1]
        }
    }

    # super RaSE
    # -------------------------------

    if (!is.null(base.dist) && super$type == "separate") {
        dist <- matrix(1, nrow = length(base), ncol = p)
        rownames(dist) <- base
        is.null.D <- is.null(D)
        is.na.D <- is.na(D)
        if (is.null.D) {
            D <- rep(floor(min(sqrt(n), p)), length(base))
            names(D) <- base
        }
        if ("lda" %in% names(base.dist)) {
            # clean data
            a <- suppressWarnings(cor(xtrain))
            b <- a - diag(diag(a))
            b0 <- which(abs(b) > 0.9999, arr.ind = T)
            b0 <- matrix(b0[b0[, 1] > b0[, 2], ], ncol = 2)
            a <- diag(cov(xtrain))

            delete.ind.lda <- unique(c(which(a == 0), b0[, 1]))
            sig.ind <- setdiff(1:p, delete.ind.lda)

            # estimate parameters
            if (is.null.D || is.na.D["lda"]) {
                D["lda"] <- floor(min(sqrt(n), length(sig.ind)))
            }
            Sigma.mle <- ((n0 - 1) * cov(xtrain[ytrain == 0, , drop = F]) + (n1 - 1) * cov(xtrain[ytrain == 1, , drop = F]))/n
            mu0.mle <- colMeans(xtrain[ytrain == 0, , drop = F])
            mu1.mle <- colMeans(xtrain[ytrain == 1, , drop = F])

            # start loops
            dist["lda", ] <- rep(1, p)
            dist["lda", delete.ind.lda] <- 0
        }

        if ("qda" %in% names(base.dist)) {
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

            delete.ind.qda <- unique(c(b0[, 1], b1[, 1], which(a0 == 0), which(a1 == 0)))
            sig.ind <- setdiff(1:p, delete.ind.qda)

            # estimate parameters
            if (is.null.D || is.na.D["qda"]) {
                D["qda"] <- floor(min(sqrt(n0), sqrt(n1), length(sig.ind)))
            }

            Sigma0.mle <- (n0 - 1)/n0 * cov(xtrain[ytrain == 0, , drop = F])
            Sigma1.mle <- (n1 - 1)/n1 * cov(xtrain[ytrain == 1, , drop = F])
            mu0.mle <- colMeans(xtrain[ytrain == 0, , drop = F])
            mu1.mle <- colMeans(xtrain[ytrain == 1, , drop = F])

            # start loops
            dist["qda",] <- rep(1, p)
            dist["qda", delete.ind.qda] <- 0
        }

        for (t in 1:(iteration + 1)) {
            output <- foreach(i = 1:B1, .combine = "rbind", .packages = "MASS") %dopar% {
                base.list <- sample(base, size = B2, prob = base.dist, replace = TRUE)
                S <- sapply(1:B2, function(j) {
                    S.size <- sample(1:D[base.list[j]], 1)
                    snew <- sample(1:p, size = min(S.size, sum(dist[base.list[j], ] != 0)), prob = dist[base.list[j], ])
                    if (base.list[j] == "lda") {
                        flag <- TRUE
                        while (flag) {
                            if (length(snew) > 2) {
                                ind0 <- findLinearCombos(Sigma.mle[snew, snew, drop = F])$remove
                                if (!is.null(ind0)) {
                                    snew <- snew[-ind0]
                                }
                            }
                            snew1 <- c(snew, rep(NA, max(D) - length(snew)))
                            if (any(abs(mu1.mle[snew1] - mu0.mle[snew1]) > 1e-10)) {
                                flag <- FALSE
                            }
                        }
                        snew1
                    } else if (base.list[j] == "qda") {
                        if (length(snew) > 2) {
                            ind0 <- findLinearCombos(Sigma0.mle[snew, snew, drop = F])$remove
                            ind1 <- findLinearCombos(Sigma1.mle[snew, snew, drop = F])$remove
                            if (!all(is.null(c(ind0, ind1)))) {
                                snew <- snew[-c(ind0, ind1)]
                            }
                        }
                        c(snew, rep(NA, max(D) - length(snew)))
                    } else {
                        c(snew, rep(NA, max(D) - length(snew)))
                    }
                })

                RaSubset(xtrain = xtrain, ytrain = ytrain, xval = xval, yval = yval, B2 = B2, S = S, base = base.list, k = k,
                         criterion = criterion, cv = cv, mu0.mle = mu0.mle, mu1.mle = mu1.mle, Sigma0.mle = Sigma0.mle,
                         Sigma1.mle = Sigma1.mle,  Sigma.mle = Sigma.mle, kl.k = kl.k, gam = gam, ...)
            }

            if (is.matrix(output)) {
                subspace <- output[, 3]
                base.list <- output[, 4]
            } else {
                subspace <- output[3]
                base.list <- output[, 4]
            }


            s <- matrix(rep(0, p*length(base)), ncol = p)
            colnames(s) <- 1:p
            rownames(s) <- base
            for (i in 1:length(subspace)) {
                s[base.list[[i]], subspace[[i]]] <- s[base.list[[i]], subspace[[i]]] + 1
            }
            base.count <- sapply(1:length(base), function(i){
                sum(Reduce("c", base.list) == base[i])
            })

            if (super$base.update) { # update the base classifier distribution
                base.dist[1:length(base.dist)] <- base.count/B1
            }

            dist <- s/base.count
            dist[dist < C0/log(p)] <- C0/p
            if (any(base.count == 0) && (!super$base.update) && t != (iteration + 1)) {
                dist[base.count == 0, ] <- 1/p
                warning("Some base classifiers have zero selecting frequency, and the feature sampling distribution cannot be calculated. Use uniform distribution instead in the next interation round.")
            }
            if ("lda" %in% base) {
                dist["lda", delete.ind.lda] <- 0
            } else if ("qda" %in% base) {
                dist["qda", delete.ind.qda] <- 0
            }



        }

        if (is.matrix(output)) {
            ytrain.pred <- data.frame(matrix(unlist(output[, 2]), ncol = B1))
            fit.list <- output[, 1]
        } else {
            ytrain.pred <- data.frame(matrix(unlist(output[2]), ncol = B1))
            fit.list <- output[1]
        }
    }



    # output
    # -------------------------------

    if (is.null(base.dist)) { # original RaSE
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
        obj <- list(marginal = c(`class 0` = p0, `class 1` = 1 - p0), base = base, criterion = criterion, B1 = B1, B2 = B2, D = D,
                    iteration = iteration, fit.list = fit.list, cutoff = cutoff, subspace = subspace, ranking = rk, scale = scale.parameters)
        class(obj) <- "RaSE"
    } else if (!is.null(base.dist)) { # super RaSE
        p0 <- sum(ytrain == 0)/nrow(xtrain)
        if (cutoff == TRUE) {
            cutoff <- RaCutoff(ytrain.pred, ytrain, p0)
        } else {
            cutoff <- 0.5
        }

        if (ranking == TRUE) {
            rk.feature <- s/base.count*100
        } else {
            rk.feature <- NULL
        }

        if (ranking == TRUE) {
            rk.base <- base.count/B1*100
            names(rk.base) <- base
        } else {
            rk.base <- NULL
        }

        if (scale == TRUE) {
            scale.parameters <- list(center = scale.center, scale = scale.scale)
        } else {
            scale.parameters <- NULL
        }

        stopImplicitCluster()
        obj <- list(marginal = c(`class 0` = p0, `class 1` = 1 - p0), base = Reduce("c", base.list), criterion = criterion, B1 = B1, B2 = B2, D = D,
                    iteration = iteration, fit.list = fit.list, cutoff = cutoff, subspace = subspace, ranking.feature = rk.feature, ranking.base = rk.base, scale = scale.parameters)
        class(obj) <- "super_RaSE"
    }





    return(obj)
}

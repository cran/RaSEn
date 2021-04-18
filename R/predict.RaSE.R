#' Predict the outcome of new observations based on the estimated RaSE classifier.
#'
#' @export
#' @param object fitted \code{'RaSE'} object using \code{Rase}.
#' @param newx a set of new observations. Each row of \code{newx} is a new observation.
#' @param ... additional arguments.
#' @return The predicted labels for new observations.
#' @seealso \code{\link{Rase}}.
#' @references
#' Tian, Y. and Feng, Y., 2021. RaSE: Random subspace ensemble classification. Journal of Machine Learning Research, 22(45), pp.1-93.
#'
#' @examples
#' \dontrun{
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' train.data <- RaModel(1, n = 100, p = 50)
#' test.data <- RaModel(1, n = 100, p = 50)
#' xtrain <- train.data$x
#' ytrain <- train.data$y
#' xtest <- test.data$x
#' ytest <- test.data$y
#'
#' model.fit <- Rase(xtrain, ytrain, B1 = 100, B2 = 100, iteration = 0, base = 'lda',
#' cores = 2, criterion = 'ric', ranking = TRUE)
#' ypred <- predict(model.fit, xtest)
#' }
#'

predict.RaSE <- function(object, newx, ...) {
    if (!is.null(object$scale)) {
        newx <- scale(newx, center = object$scale$center, scale = object$scale$scale)
    }

    if (object$base == "lda" || object$base == "qda") {
        ytest.pred <- sapply(1:object$B1, function(i) {
            as.numeric(predict(object$fit.list[[i]], newx[, object$subspace[[i]], drop = F])$class) - 1
        })
    }

    if (object$base == "knn") {
        ytest.pred <- sapply(1:object$B1, function(i) {
            as.numeric(predict(object$fit.list[[i]], newx[, object$subspace[[i]], drop = F], type = "class")) - 1
        })
    }

    if (object$base == "tree") {
        ytest.pred <- sapply(1:object$B1, function(i) {
            as.numeric(predict(object$fit.list[[i]], data.frame(x = newx[, object$subspace[[i]], drop = F]), type = "class")) - 1
        })
    }

    if (object$base == "logistic") {
        ytest.pred <- sapply(1:object$B1, function(i) {
            as.numeric(predict(object$fit.list[[i]], data.frame(x = newx[, object$subspace[[i]], drop = F])) > 0)
        })
    }

    if (object$base == "svm") {
        ytest.pred <- sapply(1:object$B1, function(i) {
            as.numeric(predict(object$fit.list[[i]], newx[, object$subspace[[i]], drop = F])) - 1
        })
    }

    if (object$base == "randomforest") {
        ytest.pred <- sapply(1:object$B1, function(i) {
            as.numeric(predict(object$fit.list[[i]], newx[, object$subspace[[i]], drop = F])) - 1
        })
    }


    if (object$base == "gamma") {
        ytest.pred <- sapply(1:object$B1, function(i) {
            gamma_classifier(t0.mle = object$fit.list[[i]][[1]], t1.mle = object$fit.list[[i]][[2]], p0 = object$fit.list[[i]][[3]],
                p1 = object$fit.list[[i]][[4]], newx, object$subspace[[i]])
        })
    }
    if (nrow(newx) == 1) {
        vote <- mean(ytest.pred, na.rm = TRUE)
    }
    if (nrow(newx) > 1) {
        vote <- rowMeans(ytest.pred, na.rm = TRUE)
    }
    Class <- as.numeric(vote > object$cutoff)
    return(Class)
}

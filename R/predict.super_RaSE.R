#' Predict the outcome of new observations based on the estimated super RaSE classifier (Zhu, J. and Feng, Y., 2021).
#'
#' @export
#' @param object fitted \code{'super_RaSE'} object using \code{Rase}.
#' @param newx a set of new observations. Each row of \code{newx} is a new observation.
#' @param type the type of prediction output. Can be 'vote', 'prob', 'raw-vote' or 'raw-prob'. Default = 'vote'.
#' \itemize{
#' \item vote: output the predicted class (by voting and cut-off) of new observations. Avalilable for all base learner types.
#' \item prob: output the predicted probabilities (posterior probability of each observation to be class 1) of new observations. It is the average probability over all base learners.
#' \item raw-vote: output the predicted class of new observations for all base learners. It is a \code{n} by \code{B1} matrix. \code{n} is the test sample size and \code{B1} is the number of base learners used in RaSE. Avalilable for all base learner types.
#' \item raw-prob: output the predicted probabilities (posterior probability of each observation to be class 1) of new observations for all base learners. It is a \code{n} by \code{B1} matrix.
#' }
#' @param ... additional arguments.
#' @return depends on the parameter \code{type}. See the list above.
#' @seealso \code{\link{Rase}}.
#' @references
#' Zhu, J. and Feng, Y., 2021. Super RaSE: Super Random Subspace Ensemble Classification. https://www.preprints.org/manuscript/202110.0042
#'
#' @examples
#' \dontrun{
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' train.data <- RaModel("classification", 1, n = 100, p = 50)
#' test.data <- RaModel("classification", 1, n = 100, p = 50)
#' xtrain <- train.data$x
#' ytrain <- train.data$y
#' xtest <- test.data$x
#' ytest <- test.data$y
#'
#' # fit a super RaSE classifier by sampling base learner from kNN, LDA and
#' # logistic regression in equal probability
#' fit <- Rase(xtrain = xtrain, ytrain = ytrain, B1 = 100, B2 = 100,
#' base = c("knn", "lda", "logistic"), super = list(type = "separate", base.update = T),
#' criterion = "cv", cv = 5, iteration = 1, cores = 2)
#' ypred <- predict(fit, xtest)
#' mean(ypred != ytest)
#' }
#'

predict.super_RaSE <- function(object, newx, type = c("vote", "prob", "raw-vote", "raw-prob"), ...) {
  type <- match.arg(type)

  if (!is.null(object$scale)) {
    newx <- scale(newx, center = object$scale$center, scale = object$scale$scale)
  }


  ytest.pred <- sapply(1:object$B1, function(i) {
    if (object$base[i] == "lda" || object$base[i] == "qda") {
      if (type == "vote" || type == "raw-vote") {
        rs <- as.numeric(predict(object$fit.list[[i]], newx[, object$subspace[[i]], drop = F])$class) - 1
      } else if (type == "prob" || type == "raw-prob") {
        rs <- as.numeric(predict(object$fit.list[[i]], newx[, object$subspace[[i]], drop = F])$posterior[, 2])
      }
    }

    if (object$base[i] == "knn") {
      if (type == "vote" || type == "raw-vote") {
        rs <- as.numeric(predict(object$fit.list[[i]], newx[, object$subspace[[i]], drop = F], type = "class")) - 1
      } else if (type == "prob" || type == "raw-prob") {
        rs <- as.numeric(predict(object$fit.list[[i]], newx[, object$subspace[[i]], drop = F], type = "prob")[, 2])
      }
    }

    if (object$base[i] == "tree") {
      if (type == "vote" || type == "raw-vote") {
        rs <- as.numeric(predict(object$fit.list[[i]], data.frame(x = newx[, object$subspace[[i]], drop = F]), type = "class")) - 1
      } else if (type == "prob" || type == "raw-prob") {
        rs <- as.numeric(predict(object$fit.list[[i]], data.frame(x = newx[, object$subspace[[i]], drop = F]), type = "prob")[, 2])
      }
    }

    if (object$base[i] == "logistic") {
      if (type == "vote" || type == "raw-vote") {
        rs <- as.numeric(predict(object$fit.list[[i]], data.frame(x = newx[, object$subspace[[i]], drop = F])) > 0)
      } else if (type == "prob" || type == "raw-prob") {
        rs <- as.numeric(predict(object$fit.list[[i]], data.frame(x = newx[, object$subspace[[i]], drop = F]), type = "response"))
      }
    }

    if (object$base[i] == "svm") {
      rs <- as.numeric(predict(object$fit.list[[i]], newx[, object$subspace[[i]], drop = F])) - 1
    }


    if (object$base[i] == "randomforest") {
      if (type == "vote" || type == "raw-vote") {
        rs <- as.numeric(predict(object$fit.list[[i]], newx[, object$subspace[[i]], drop = F])) - 1
      } else if (type == "prob" || type == "raw-prob") {
        rs <- as.numeric(predict(object$fit.list[[i]], newx[, object$subspace[[i]], drop = F], type = "prob")[, 2])
      }

    }

    if (object$base[i] == "gamma") {
      rs <- gamma_classifier(t0.mle = object$fit.list[[i]][[1]], t1.mle = object$fit.list[[i]][[2]], p0 = object$fit.list[[i]][[3]],
                         p1 = object$fit.list[[i]][[4]], newx, object$subspace[[i]])
    }

    rs
  })






  # final output
  if (type == "vote") {
    if (nrow(newx) == 1) {
      vote <- mean(ytest.pred, na.rm = TRUE)
    }
    if (nrow(newx) > 1) {
      vote <- rowMeans(ytest.pred, na.rm = TRUE)
    }
    Class <- as.numeric(vote > object$cutoff)
    return(Class)
  } else if (type == "prob") {
    if (nrow(newx) == 1) {
      vote <- mean(ytest.pred, na.rm = TRUE)
    }
    if (nrow(newx) > 1) {
      vote <- rowMeans(ytest.pred, na.rm = TRUE)
    }
    return(as.numeric(vote))
  } else if (type == "raw-vote" || type == "raw-prob") {
    return(ytest.pred)
  }

}

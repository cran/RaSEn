#' Visualize the feature ranking results of a fitted RaSE object.
#'
#' This function plots the feature ranking results from a fitted \code{'RaSE'} object via \code{ggplot2}. In the figure, x-axis represents the feature number and y-axis represents the selected percentage of each feature in B1 subspaces.
#' @export
#' @param object fitted \code{'RaSE'} model object.
#' @param main title of the plot. Default = \code{NULL}, which makes the title following the orm 'RaSE-base' with subscript i (rounds of iterations), where base represents the type of base classifier. i is omitted when it is zero.
#' @param xlab the label of x-axis. Default = 'feature'.
#' @param ylab the label of y-axis. Default = 'selected percentage'.
#' @param ... additional arguments.
#' @return a \code{'ggplot'} object.
#' @seealso \code{\link{Rase}}.
#' @references
#' Tian, Y. and Feng, Y., 2021. RaSE: Random subspace ensemble classification. Journal of Machine Learning Research, 22(45), pp.1-93.
#'
#' @examples
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' train.data <- RaModel("classification", 1, n = 100, p = 50)
#' xtrain <- train.data$x
#' ytrain <- train.data$y
#'
#' # fit RaSE classifier with QDA base classifier
#' fit <- Rase(xtrain, ytrain, B1 = 50, B2 = 50, iteration = 1, base = 'qda',
#' cores = 2, criterion = 'ric')
#'
#' # plot the selected percentage of each feature appearing in B1 subspaces
#' RaPlot(fit)
#'

RaPlot <- function(object, main = NULL, xlab = "feature", ylab = "selected percentage", ...) {
  if (is.null(object$ranking)) {
    stop("RaSE object has no feature ranking results to plot!")
  }

  if (is.null(main)) {
    if (object$iteration > 0) {
      ggplot(data = data.frame('percentage' = object$ranking, 'feature' = 1:length(object$ranking)), mapping = aes_string(y = 'percentage', x = 'feature')) +
        geom_point() + ggtitle(expr(paste("RaSE-", !!object$base, sep = "")[!!object$iteration])) + theme(plot.title = element_text(hjust = 0.5)) +
        labs(x = xlab, y = ylab)
    } else {
      ggplot(data = data.frame('percentage' = object$ranking, 'feature' = 1:length(object$ranking)), mapping = aes_string(y = 'percentage', x = 'feature')) +
        geom_point() + ggtitle(expr(paste("RaSE-", !!object$base, sep = ""))) + theme(plot.title = element_text(hjust = 0.5)) +
        labs(x = xlab, y = ylab)
    }
  } else {
    ggplot(data = data.frame('percentage' = object$ranking, 'feature' = 1:length(object$ranking)), mapping = aes_string(y = 'percentage', x = 'feature')) +
      geom_point() + ggtitle(as.character(main)) + theme(plot.title = element_text(hjust = 0.5)) + labs(x = xlab, y = ylab)
  }



}

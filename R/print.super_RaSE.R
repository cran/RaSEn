#' Print a fitted super_RaSE object.
#'
#' Similar to the usual print methods, this function summarizes results.
#' from a fitted \code{'super_RaSE'} object.
#' @export
#' @param x fitted \code{'super_RaSE'} model object.
#' @param ... additional arguments.
#' @return No value is returned.
#' @seealso \code{\link{Rase}}.
#' @examples
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' train.data <- RaModel("classification", 1, n = 100, p = 50)
#' xtrain <- train.data$x
#' ytrain <- train.data$y
#'
#' # test RaSE classifier with LDA base classifier
#' fit <- Rase(xtrain, ytrain, B1 = 50, B2 = 50, iteration = 0, cutoff = TRUE,
#' base = 'lda', cores = 2, criterion = 'ric', ranking = TRUE)
#'
#' # print the summarized results
#' print(fit)

print.super_RaSE <- function(x, ...) {
  cat("Marginal probabilities:", "\n")
  print(x$marginal)
  cat("Count of base classifier types among", x$B1, "classifiers:")
  print(table(x$base))
  cat("Criterion:", x$criterion, "\n")
  cat("B1:", x$B1, "\n")
  cat("B2:", x$B2, "\n")
  cat("D: \n")
  print(x$D)
  cat("Cutoff:", x$cutoff, "\n")
  if (!is.null(x$ranking)) {
    cat("Selected percentage of each feature appearing in B1 subspaces under different base classifier types:", "\n")
    print(x$ranking)
  }
}

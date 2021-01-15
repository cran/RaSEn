#' Print a fitted RaSE object.
#'
#' Similar to the usual print methods, this function summarizes results.
#' from a fitted \code{'RaSE'} object.
#' @export
#' @param x fitted \code{'RaSE'} model object.
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

print.RaSE <- function(x, ...) {
  cat("Marginal probabilities:", "\n")
  print(x$marginal)
  cat("Type of base classifiers:", x$base, "\n")
  cat("Criterion:", x$criterion, "\n")
  cat("B1:", x$B1, "\n")
  cat("B2:", x$B2, "\n")
  cat("D:", x$D, "\n")
  cat("Cutoff:", x$cutoff, "\n")
  if (!is.null(x$ranking)) {
    cat("Selected percentage of each feature appearing in B1 subspaces:", "\n")
    print(x$ranking)
  }
}

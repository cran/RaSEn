#' Rank the features by selected percentages provided by the output from \code{RaScreen}.
#'
#' @export
#' @param object output from \code{RaScreen}.
#' @param selected.num the number of selected variables. User can either choose from the following popular options or input an positive integer no larger than the dimension.
#' \itemize{
#' \item 'all positive': the number of variables with positive selected percentage.
#' \item 'D': floor(D), where D is the maximum of ramdom subspace size.
#' \item '1.5D': floor(1.5D).
#' \item '2D': floor(2D).
#' \item '3D': floor(3D).
#' \item 'n/logn': floor(n/logn), where n is the sample size.
#' \item '1.5n/logn': floor(1.5n/logn).
#' \item '2n/logn': floor(2n/logn).
#' \item '3n/logn': floor(3n/logn).
#' \item 'n-1': the sample size n - 1.
#' \item 'p': the dimension p.
#' }
#' @param iteration indicates results from which iteration to use. It should be an positive integer. Default = the maximal interation round used by the output from \code{RaScreen}.
#' @return Selected variables (indexes).
#' @references
#' Tian, Y. and Feng, Y., 2021(a). RaSE: A Variable Screening Framework via Random Subspace Ensembles. arXiv preprint arXiv:2102.03892.
#'
#' @examples
#' \dontrun{
#' set.seed(0, kind = "L'Ecuyer-CMRG")
#' train.data <- RaModel("screening", 1, n = 100, p = 100)
#' xtrain <- train.data$x
#' ytrain <- train.data$y
#'
#' # test RaSE screening with linear regression model and BIC
#' fit <- RaScreen(xtrain, ytrain, B1 = 100, B2 = 50, iteration = 0, model = 'lm',
#' cores = 2, criterion = 'bic')
#'
#' # Select floor(n/logn) variables
#' RaRank(fit, selected.num = "n/logn")
#' }

RaRank <- function(object, selected.num = "all positive", iteration = object$iteration) {
  if (iteration > object$iteration) {
    stop("There are not so many available iteration results! Please check the iteration number.")
  }

  iteration <- iteration+1

  if (!is.list(object$selected.perc)) {
    pos.num <- sum(object$selected.perc > 0)
  } else {
    pos.num <- sum(object$selected.perc[[iteration]] > 0)
  }


  if (!is.numeric(selected.num)) {
    if (selected.num == "all positive") {
      selected.num <- pos.num
    } else if (selected.num == "D") {
      selected.num <- floor(object$D)
    } else if (selected.num == "1.5D") {
      selected.num <- floor(1.5*object$D)
    } else if (selected.num == "2D") {
      selected.num <- floor(2*object$D)
    } else if (selected.num == "3D") {
      selected.num <- floor(3*object$D)
    } else if (selected.num == "n/logn") {
      selected.num <- floor((object$n)/log(object$n))
    } else if (selected.num == "1.5n/logn") {
      selected.num <- floor(1.5*(object$n)/log(object$n))
    } else if (selected.num == "2n/logn") {
      selected.num <- floor(2*(object$n)/log(object$n))
    } else if (selected.num == "3n/logn") {
      selected.num <- floor(3*(object$n)/log(object$n))
    } else if (selected.num == "n-1") {
      selected.num <- object$n - 1
    } else if (selected.num == "p") {
      selected.num <- object$p
    }
  }

  if (!is.list(object$selected.perc)) {
    if (selected.num > pos.num){
      L <- as.numeric(c(order(object$selected.perc, decreasing = T)[1:pos.num], sample(which(object$selected.perc == 0))[1:(selected.num-pos.num)]))
      warning(paste("Only", pos.num, "variables have positive selected percentage but request", selected.num, "ones. The last", selected.num-pos.num, "variables are randomly sampled!"))
    } else {
      L <- as.numeric(order(object$selected.perc, decreasing = T)[1:selected.num])
    }
  } else {
    if (selected.num > pos.num){
      L <- as.numeric(c(order(object$selected.perc[[iteration]], decreasing = T)[1:pos.num], sample(which(object$selected.perc[[iteration]] == 0))[1:(selected.num-pos.num)]))
      warning(paste("Only", pos.num, "variables have positive selected percentage but request", selected.num, "ones. The last", selected.num-pos.num, "variables are randomly sampled!"))
    } else {
      L <- as.numeric(order(object$selected.perc[[iteration]], decreasing = T)[1:selected.num])
    }
  }


  return(L)
}

#' Generate data \eqn{(x, y)} from 6 models.
#'
#' \code{RaModel} generates data from 6 models described in Tian, Y. and Feng, Y., 2020.
#' @export
#' @param Model.No model number, which can be 1, 2, 3, 4.
#' @param n sample size
#' @param p data dimension
#' @param p0 marginal probability of class 0. Default = 0.5. Only available when Model.No = 1, 2, 3.
#' @param sparse a logistic object indicating model sparsity. Default = TRUE. Only available when Model.No = 1, 4. When it equals to FALSE, the data is generated from model 1' or 4' as described in Tian, Y. and Feng, Y., 2020.
#' @return
#' \item{x}{n * p matrix. n observations and p features.}
#' \item{y}{n 0/1 observations.}
#' @note Models 1, 2 and 4 require \eqn{p \ge 5}. Models 1' and 3 requires \eqn{p \ge 50}. Model 4' requires \eqn{p \ge 30}.
#' @seealso \code{\link{Rase}}
#' @examples
#' train.data <- RaModel(1, n = 100, p = 50)
#' xtrain <- train.data$x
#' ytrain <- train.data$y
#'
#' @references
#' Tian, Y. and Feng, Y., 2020. RaSE: Random subspace ensemble classification. arXiv preprint arXiv:2006.08855.

RaModel <- function(Model.No, n, p, p0 = 1/2, sparse = TRUE) {
    if (Model.No == 1) {
        Y1 <- rmultinom(1, n, c(p0, 1 - p0))
        Y <- c(rep(0, Y1[1, 1]), rep(1, Y1[2, 1]))
        Sigma <- outer(1:p, 1:p, function(i, j) {
            0.5^(abs(i - j))
        })

        mu0 <- rep(0, p)
        if (sparse) {
            mu1 <- 0.556 * Sigma %*% c(3, 1.5, 0, 0, 2, rep(0, p - 5))
        } else {
            mu1 <- 0.556 * Sigma %*% c(0.9^(1:50), rep(0, p-50))
        }

        X0 <- mvrnorm(Y1[1, 1], mu0, Sigma)
        X1 <- mvrnorm(Y1[2, 1], mu1, Sigma)

        X <- rbind(X0, X1)
    }

    if (Model.No == 2) {
        Y1 <- rmultinom(1, n, c(p0, 1 - p0))
        Y <- c(rep(0, Y1[1, 1]), rep(1, Y1[2, 1]))
        a0 <- c(c(2, 1.5, 1.5, 2, 2), rep(1, p - 5))
        a1 <- c(c(2.5, 1.5, 1.5, 1, 1), rep(1, p - 5))
        b0 <- c(c(1.5, 3, 1, 1, 1), rep(3, p - 5))
        b1 <- c(c(2, 1, 3, 1, 1), rep(3, p - 5))

        X <- rbind(sapply(1:p, function(i) {
            rgamma(Y1[1, 1], shape = a0[i], scale = b0[i])
        }), sapply(1:p, function(i) {
            rgamma(Y1[2, 1], shape = a1[i], scale = b1[i])
        }))
    }


    if (Model.No == 3) {
        Y1 <- rmultinom(1, n, c(p0, 1 - p0))
        Y <- c(rep(0, Y1[1, 1]), rep(1, Y1[2, 1]))
        Sigma0 <- diag(1, p) + outer(1:p, 1:p, function(i, j) {
            0.3 * I(abs(i - j) == 1)
        })
        Sigma <- matrix(0, nrow = p, ncol = p)
        Sigma[10, 10] <- -0.3758
        Sigma[10, 30] <- 0.0616
        Sigma[30, 10] <- 0.0616
        Sigma[10, 50] <- 0.2037
        Sigma[50, 10] <- 0.2037
        Sigma[30, 30] <- -0.5482
        Sigma[30, 50] <- 0.0286
        Sigma[50, 30] <- 0.0286
        Sigma[50, 50] <- -0.4614
        Sigma1 <- solve(Sigma0 + Sigma)
        Sigma0 <- solve(Sigma0)
        mu1 <- rep(0, p)
        mu0 <- Sigma1 %*% c(0.6, 0.8, rep(0, p - 2))

        X0 <- mvrnorm(Y1[1, 1], mu0, Sigma0)
        X1 <- mvrnorm(Y1[2, 1], mu1, Sigma1)

        X <- rbind(X0, X1)
    }

    if (Model.No == 4) {
        X0 <- mvrnorm(n = 10, mu = rep(0, p), Sigma = diag(p))
        Y0 <- rep(c(0, 1), each = 5)
        if (sparse) {
            Ds <- sapply(1:n, function(o) {
                i0 <- sample(10, 1)
                c(mvrnorm(n = 1, mu = c(X0[i0, 1:5], rep(0, p - 5)), Sigma =  0.5^2*diag(p)), Y0[i0])
            })
        } else {
            Ds <- sapply(1:n, function(o) {
                i0 <- sample(10, 1)
                c(mvrnorm(n = 1, mu = c(X0[i0, 1:30], rep(0, p - 30)), Sigma =  2*diag(p)), Y0[i0])
            })
        }

        X <- t(Ds[-nrow(Ds), ])
        Y <- Ds[nrow(Ds), ]
    }



    return(list(x = X, y = Y))
}

#' Generate data \eqn{(x, y)} from various models in two papers.
#'
#' \code{RaModel} generates data from 4 models described in Tian, Y. and Feng, Y., 2021(b) and 8 models described in Tian, Y. and Feng, Y., 2021(a).
#' @export
#' @param model.type indicator of the paper covering the model, which can be 'classification' (Tian, Y. and Feng, Y., 2021(b)) or 'screening' (Tian, Y. and Feng, Y., 2021(a)).
#' @param model.no model number. It can be 1-4 when \code{model.type} = 'classification' and 1-8 when \code{model.type} = 'screening', respectively.
#' @param n sample size
#' @param p data dimension
#' @param p0 marginal probability of class 0. Default = 0.5. Only used when \code{model.type} = 'classification' and \code{model.no} = 1, 2, 3.
#' @param sparse a logistic object indicating model sparsity. Default = TRUE. Only used when \code{model.type} = 'classification' and \code{model.no} = 1, 4.
#' @return
#' \item{x}{n * p matrix. n observations and p features.}
#' \item{y}{n responses.}
#' @note When \code{model.type} = 'classification' and \code{sparse} = TRUE, models 1, 2, 4 require \eqn{p \ge 5} and model 3 requires
#'  \eqn{p \ge 50}. When \code{model.type} = 'classification' and \code{sparse} = FALSE, models 1 and 4 require \eqn{p \ge 50} and
#'  \eqn{p \ge 30}, respectively. When \code{model.type} = 'screening', models 1, 4, 5 and 7 require \eqn{p \ge 4}. Models 2 and 8 require \eqn{p \ge 5}. Model 3 requires \eqn{p \ge 22}. Model 5 requires \eqn{p \ge 2}.
#' @seealso \code{\link{Rase}}, \code{\link{RaScreen}}.
#' @examples
#' train.data <- RaModel("classification", 1, n = 100, p = 50)
#' xtrain <- train.data$x
#' ytrain <- train.data$y
#'
#' \dontrun{
#' train.data <- RaModel("screening", 2, n = 100, p = 50)
#' xtrain <- train.data$x
#' ytrain <- train.data$y
#' }
#' @references
#' Tian, Y. and Feng, Y., 2021(a). RaSE: A variable screening framework via random subspace ensembles. Journal of the American Statistical Association, (just-accepted), pp.1-30.
#'
#' Tian, Y. and Feng, Y., 2021(b). RaSE: Random subspace ensemble classification. Journal of Machine Learning Research, 22(45), pp.1-93.

RaModel <- function(model.type, model.no, n, p, p0 = 1/2, sparse = TRUE) {
    if (model.type == "classification") {
        if (model.no == 1) {
            N <- as.vector(rmultinom(1, n, c(p0, 1 - p0)))
            Y <- c(rep(0, N[1]), rep(1, N[2]))
            Sigma <- outer(1:p, 1:p, function(i, j) {
                0.5^(abs(i - j))
            })
            R <- chol(Sigma)

            mu0 <- rep(0, p)
            if (sparse) {
                mu1 <- 0.556 * Sigma %*% c(3, 1.5, 0, 0, 2, rep(0, p - 5))
            } else {
                mu1 <- 0.556 * Sigma %*% c(0.9^(1:50), rep(0, p-50))
            }

            X <- tcrossprod(matrix(rnorm(n*p), nrow = n, ncol = p), t(R)) + rbind(matrix(rep(mu0, N[1]), nrow = N[1], byrow = T), matrix(rep(mu1, N[2]), nrow = N[2], byrow = T))
        }

        if (model.no == 2) {
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


        if (model.no == 3) {
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

            # R0 <- chol(Sigma0)
            # R1 <- chol(Sigma1)
            # tcrossprod(matrix(rnorm(n*p), nrow = n, ncol = p), t(R))
            X0 <- mvrnorm(Y1[1, 1], mu0, Sigma0)
            X1 <- mvrnorm(Y1[2, 1], mu1, Sigma1)

            X <- rbind(X0, X1)
        }

        if (model.no == 4) {
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
    }

    if (model.type == "screening") {
        if (model.no == 1) { # exp in SIS
            r <- 0.5
            beta0 <- c(rep(5, 3), -15*sqrt(r))
            Sigma <- matrix(r, nrow = p, ncol = p)
            diag(Sigma) <- 1
            Sigma[-4, 4] <- sqrt(r)
            Sigma[4, -4] <- sqrt(r)
            R <- chol(Sigma)
            X <- tcrossprod(matrix(rnorm(n*p), nrow = n, ncol = p), t(R))
            Y <- as.vector(X[, 1:4]%*%beta0 + rnorm(n))
        }



        if (model.no == 2) { # knn example, continuous response
            Sigma <- outer(1:p, 1:p, function(i, j) {
                0.5^(abs(i - j))
            })
            beta <- c(rep(0.5, 5), rep(0, p-5))
            z <- sample(0:1, n, replace = TRUE, prob = c(0.5, 0.5))
            R <- chol(Sigma)
            X <- tcrossprod(matrix(rnorm(n*p), nrow = n, ncol = p), t(R))
            Y <- as.vector(X[,1:5]%*%beta[1:5] + 0.5*rt(n, df = 2))
            X[z == 0, 1:5] <- X[z == 0, 1:5] + 3
            X[z == 1, 1:5] <- X[z == 1, 1:5] - 3
        }



        if (model.no == 3) { # exp 1.c in DC-SIS
            u <- sample(0:1, 4, prob = c(0.6, 0.4), replace = T)
            Z <- abs(rnorm(4))

            beta <- (-1)^u * (Z + 4*log(n)/sqrt(n))
            Sigma <- outer(1:p, 1:p, function(i, j){
                0.8^(abs(i-j))
            })
            if (!exists("R")) {
                R <- chol(Sigma)
            }
            X <- tcrossprod(matrix(rnorm(n*p), nrow = n, ncol = p), t(R))
            Y <- 2*beta[1]*X[, 1]*X[, 2] + 3*beta[2]*I(X[, 12]<0)*X[, 22] + rnorm(n)
        }

        if (model.no == 4) { # 4-way interaction
            X <- matrix(rnorm(n*p), nrow = n)
            Y <- 2*(1.5*sqrt(abs(X[, 1])) + sqrt(abs(X[, 1]))*X[, 2]^2 + 2*sin(X[, 1])*sin(X[, 2])*sin(X[, 3])^2 + 6*sin(X[, 1])*abs(X[, 2])*sin(X[, 3])*X[, 4]^2)  + 0.5*rnorm(n)
        }


        if (model.no == 5){ #samworth model 1
            Y1 <- rmultinom(1, n, c(p0, 1 - p0))
            Y <- c(rep(0, Y1[1, 1]), rep(1, Y1[2, 1]))
            Y11 <- rmultinom(1, Y1[1, 1], c(1/2, 1/2))
            Y22 <- rmultinom(1, Y1[2, 1], c(1/2, 1/2))
            mu0 <- c(2, 2, rep(0, p - 2))
            mu1 <- c(2, -2, rep(0, p - 2))
            X0 <- rbind(t(matrix(mu0, p, Y11[1, 1])), t(matrix(-mu0, p, Y11[2, 1]))) + matrix(rnorm(Y1[1, 1] * p), Y1[1, 1], p)
            X1 <- rbind(t(matrix(mu1, p, Y22[1, 1])), t(matrix(-mu1, p, Y22[2, 1]))) + matrix(rnorm(Y1[2, 1] * p), Y1[2, 1], p)
            X <- rbind(X0, X1)
        }

        if (model.no == 6) { # exp in isis
            Xt <- cbind(matrix(runif(4*n, min = -sqrt(3), max = sqrt(3)), nrow = n), matrix(rnorm(n*(p-4)), nrow = n))
            X <- Xt
            X[, 1] <- Xt[, 1] - sqrt(2)*Xt[, 5]
            X[, 2] <- Xt[, 2] + sqrt(2)*Xt[, 5]
            X[, 3] <- Xt[, 3] - sqrt(2)*Xt[, 5]
            X[, 4] <- Xt[, 4] + sqrt(2)*Xt[, 5]
            X[, 5:p] <- X[, 5:p]*sqrt(3)
            pr <- matrix(nrow = n, ncol = 4)
            a <- 5/sqrt(3)
            pr[, 1] <- -a*Xt[, 1]+ a*Xt[, 4]
            pr[, 2] <- a*Xt[, 1] - a*Xt[, 2]
            pr[, 3] <- a*Xt[, 2] - a*Xt[, 3]
            pr[, 4] <- a*Xt[, 3] - a*Xt[, 4]
            Y <- sapply(1:n, function(i){sample(0:3, 1, prob = exp(pr[i, ]))})
        }

        if (model.no == 7) { # exp in SIS, with covariates from the rat data set
            rat <- NULL
            data(rat, envir = environment())
            r <- 0.5
            n <- length(rat$y)
            beta0 <- c(rep(5, 3), -15*sqrt(r))
            dta <- cbind(rat$x, rat$y)
            v.ind <- sample(1:ncol(dta), p)
            X <- scale(dta[, v.ind])
            Y <- as.vector(X[, 1:4]%*%beta0 + rnorm(n))
        }

        if (model.no == 8) { # knn example with mixed types of variables
            if (p != 2000) {
                stop("Screening Model 8 requires p = 2000!")
            }
            X <- matrix(nrow = n, ncol = p)
            Sigma <- outer(1:(p*4/5), 1:(p*4/5), function(i, j) {
                0.5^(abs(i - j))
            })
            R <- chol(Sigma)

            beta <- c(rep(0.5, 5), rep(0, p-5))
            z <- sample(0:1, n, replace = TRUE, prob = c(0.5, 0.5))

            X[, -seq(5,2000,5)] <- tcrossprod(matrix(rnorm(n*(p*4/5)), nrow = n, ncol = p*4/5), t(R))
            X[, seq(5,2000,5)] <- sapply(1:(p/5), function(j){
                sample(seq(-2,2,1), n, replace = TRUE)
            })
            Y <- as.vector(X[,1:5]%*%beta[1:5] + 0.5*rt(n, df = 2))

            X[z == 0, 1:5] <- X[z == 0, 1:5] + 3
            X[z == 1, 1:5] <- X[z == 1, 1:5] - 3
        }


    }


    return(list(x = X, y = Y))

}

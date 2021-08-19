scale_Rase <- function(x) {
    scale.center <- rep(0, ncol(x))
    scale.scale <- rep(0, ncol(x))
    const.ind <- sapply(1:ncol(x), function(i) {
        I(length(unique(x[, i])) == 1)
    })
    scale.center[!const.ind] <- attr(scale(x[, !const.ind]), "scaled:center")
    scale.scale[!const.ind] <- attr(scale(x[, !const.ind]), "scaled:scale")
    scale.center[const.ind] <- colMeans(x)[const.ind] - 1
    scale.scale[const.ind] <- sqrt(nrow(x)/(nrow(x) - 1))
    x[, !const.ind] <- scale(x[, !const.ind])
    x[, const.ind] <- sqrt((nrow(x) - 1)/nrow(x))
    return(list(data = x, center = scale.center, scale = scale.scale))
}


gamma_classifier <- function(t0.mle, t1.mle, p0, p1, newx, S) {
    f0 <- rowSums(sapply(S, function(i) {
        log(dgamma(newx[, i], shape = t0.mle[i, 1], scale = t0.mle[i, 2]))
    })) + log(p0)
    f1 <- rowSums(sapply(S, function(i) {
        log(dgamma(newx[, i], shape = t1.mle[i, 1], scale = t1.mle[i, 2]))
    })) + log(p1)

    return(1 * I(f1 > f0))
}


calc_ebic <- function(x, y, S, gam, weights = rep(1, length(y))/length(y)) {
    n <- nrow(x)
    p <- ncol(x)
    if(length(unique(y)) > 2) {
        fit <- multinom(y ~ x[, S, drop = F], family = "multinomial", trace = FALSE)
    } else {
        fit <- glm(y ~ x[, S, drop = F], family = "binomial", trace = FALSE, weights = weights)
    }


    return (deviance(fit) + (length(unique(y))-1)*length(S)*(log(n) + 2*gam*log(p)))
}

calc_ebic_glmnet <- function(x, y, S, gam, weights, lower.limits, upper.limits) {
    n <- nrow(x)
    p <- ncol(x)

    fit <- glmnet(x = x[, S, drop = F], y = y, alpha = 1, lambda = 0, weights = weights, family = "binomial", lower.limits = lower.limits, upper.limits = upper.limits)

    return (deviance(fit) + (length(unique(y))-1)*length(S)*(log(n) + 2*gam*log(p)))
}


calc_aic <- function(x, y, S, weights = rep(1, length(y))/length(y)) {
    n <- nrow(x)
    p <- ncol(x)
    if(length(unique(y)) > 2) {
        fit <- multinom(y ~ x[, S, drop = F], family = "multinomial", trace = FALSE)
    } else {
        fit <- glm(y ~ x[, S, drop = F], family = "binomial", trace = FALSE, weights = weights)
    }

    return (deviance(fit) + (length(unique(y))-1)*length(S)*2)
}

calc_aic_glmnet <- function(x, y, S, weights, lower.limits, upper.limits) {
    n <- nrow(x)
    p <- ncol(x)

    fit <- glmnet(x = x[, S, drop = F], y = y, alpha = 1, lambda = 0, weights = weights, family = "binomial", lower.limits = lower.limits, upper.limits = upper.limits)

    return (deviance(fit) + (length(unique(y))-1)*length(S)*2)
}

ric <- function(model, x, y, S, posterior0 = NULL, posterior1 = NULL, deg = NULL, p0 = NULL, p1 = NULL, t0.mle = NULL, t1.mle = NULL, mu0.mle = NULL,  mu1.mle = NULL, Sigma.mle = NULL, Sigma0.mle = NULL, Sigma1.mle = NULL, weights = NULL,...) {
    list2env(list(...), environment())
    n <- length(y)
    ind0 <- which(y == 0)
    ind1 <- which(y == 1)
    lik <- rep(0, n)
    if (model == "other") {
        return(-2 * p0 * sum(log(posterior0/posterior1)[ind0]*weights[ind0]) - 2 * p1 * sum(log(posterior1/posterior0)[ind1]*weights[ind1]) + deg(length(S))/sqrt(n) *
            log(log(n)))
    }

    if (model == "gamma") {
        return(-2 * sum(log(sapply(S, function(i) {
            dgamma(x[ind0, i], shape = t0.mle[i, 1], scale = t0.mle[i, 2])/dgamma(x[ind0, i], shape = t1.mle[i, 1], scale = t1.mle[i,
                2])
        })))/n - 2 * sum(log(sapply(S, function(i) {
            dgamma(x[ind1, i], shape = t1.mle[i, 1], scale = t1.mle[i, 2])/dgamma(x[ind1, i], shape = t0.mle[i, 1], scale = t0.mle[i,
                2])
        })))/n + log(log(n))/sqrt(n) * 2 * length(S))
    }

    if (model == "lda") {
        if (nrow(Sigma.mle) != length(S)) {
            return(-t(mu1.mle[S] - mu0.mle[S]) %*% solve(Sigma.mle[S, S, drop = F]) %*% (mu1.mle[S] - mu0.mle[S]) + log(log(n))/sqrt(n) *length(S))
        } else {
            return(-t(mu1.mle[S] - mu0.mle[S]) %*% solve(Sigma.mle) %*% (mu1.mle[S] - mu0.mle[S]) + log(log(n))/sqrt(n) *length(S))
        }
    }

    if (model == "qda") {
        if (nrow(Sigma0.mle) != length(S)) {
            Sigma1.inv <- solve(Sigma1.mle[S, S, drop = F])
            Sigma0.inv <- solve(Sigma0.mle[S, S, drop = F])
            TS <- sum(diag((Sigma1.inv - Sigma0.inv) %*% (p1 * Sigma1.mle[S, S, drop = F] - p0 * Sigma0.mle[S, S, drop = F])))
            MS <- (p1 - p0) * (log(det(Sigma1.mle[S, S, drop = F])) - log(det(Sigma0.mle[S, S, drop = F])))
            DS <- t(mu1.mle[S] - mu0.mle[S]) %*% (p1 * Sigma0.inv + p0 * Sigma1.inv) %*% (mu1.mle[S] - mu0.mle[S])
        } else {
            Sigma1.inv <- solve(Sigma1.mle)
            Sigma0.inv <- solve(Sigma0.mle)
            TS <- sum(diag((Sigma1.inv - Sigma0.inv) %*% (p1 * Sigma1.mle - p0 * Sigma0.mle)))
            MS <- (p1 - p0) * (log(det(Sigma1.mle)) - log(det(Sigma0.mle)))
            DS <- t(mu1.mle[S] - mu0.mle[S]) %*% (p1 * Sigma0.inv + p0 * Sigma1.inv) %*% (mu1.mle[S] - mu0.mle[S])
        }

        return(TS - DS + MS + log(log(n))/sqrt(n) * (length(S) * (length(S) + 3))/2)
    }

}


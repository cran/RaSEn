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


create_pmatrix_from_terms = function(xx, terms) {
    nt = length(terms)
    nr = nrow(xx)
    pmatrix = matrix(0, nr, 0)

    if (nt > 0)
        for (it in 1:nt) {
            term = terms[it]
            if (grepl("*", term, fixed = T)) {
                splits = strsplit(term, "*", fixed = T)[[1]]
                id1 = as.numeric(splits[1])
                id2 = as.numeric(splits[2])
                pmatrix = cbind(pmatrix, term = xx[, id1] * xx[, id2])
            } else {
                id = as.numeric(term)
                pmatrix = cbind(pmatrix, term = xx[, id])
            }
        }
    return(pmatrix)
}

calc_lda_BIC = function(xx, yy, cur_set, D, K, debug = F, gam = 0) {
    N = nrow(xx)
    D = ncol(xx)
    K = max(yy)
    d = length(cur_set)

    ll = 0
    if (d == 0) {
        p = numeric(K)
        for (i in 1:N) {
            p[yy[i]] = p[yy[i]] + 1/N
        }
        for (k in 1:K) {
            ll = ll + sum(yy == k) * log(p[k])
        }
        BIC = -2 * ll + (K - 1) * (log(N) + 2 * gam * log(D))
        return(BIC)
    } else {
        lgt = multinom(yy ~ as.matrix(xx[, cur_set]), family = "binomial", trace = F)
        BIC = lgt$deviance + (K - 1) * (1 + d) * (log(N) + 2 * gam * log(D))
        return(BIC)
    }
}


calc_BIC = function(xx, yy, terms, D, K, debug = F, gam = 0) {
    N = length(yy)
    D = ncol(xx)
    K = max(yy)
    d = length(terms)

    ll = 0
    if (d == 0) {
        p = numeric(K)
        for (i in 1:N) {
            p[yy[i]] = p[yy[i]] + 1/N
        }
        for (k in 1:K) {
            ll = ll + sum(yy == k) * log(p[k])
        }
        BIC = (K - 1) * (log(N) + 2 * gam * log(D))
        BIC = BIC - 2 * ll
        return(BIC)
    } else {
        pmatrix = create_pmatrix_from_terms(xx, terms)

        lgt = multinom(yy ~ pmatrix, family = "multinomial", trace = F)
        BIC = lgt$deviance
        BIC = BIC + (K - 1) * (1 + ncol(pmatrix)) * (log(N) + 2 * gam * log(D))

        return(BIC)
    }
}


ric <- function(model, x, y, S, posterior0 = NULL, posterior1 = NULL, deg = NULL, p0 = NULL, p1 = NULL, t0.mle = NULL, t1.mle = NULL, mu0.mle = NULL,  mu1.mle = NULL, Sigma.mle = NULL, Sigma0.mle = NULL, Sigma1.mle = NULL, ...) {
    list2env(list(...), environment())
    n <- length(y)
    ind0 <- which(y == 0)
    ind1 <- which(y == 1)
    lik <- rep(0, n)
    if (model == "other") {
        return(-2 * p0 * sum(log(posterior0/posterior1)[ind0]) - 2 * p1 * sum(log(posterior1/posterior0)[ind1]) + deg(length(S))/sqrt(n) *
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
        return(-t(mu1.mle[S] - mu0.mle[S]) %*% solve(Sigma.mle[S, S, drop = F]) %*% (mu1.mle[S] - mu0.mle[S]) + log(log(n))/sqrt(n) *
            length(S))
    }

    if (model == "qda") {
        Sigma1.inv <- solve(Sigma1.mle[S, S, drop = F])
        Sigma0.inv <- solve(Sigma0.mle[S, S, drop = F])
        TS <- sum(diag((Sigma1.inv - Sigma0.inv) %*% (p1 * Sigma1.mle[S, S, drop = F] - p0 * Sigma0.mle[S, S, drop = F])))
        MS <- (p1 - p0) * (log(det(Sigma1.mle[S, S, drop = F])) - log(det(Sigma0.mle[S, S, drop = F])))
        DS <- t(mu1.mle[S] - mu0.mle[S]) %*% (p1 * Sigma0.inv + p0 * Sigma1.inv) %*% (mu1.mle[S] - mu0.mle[S])

        return(TS - DS + MS + log(log(n))/sqrt(n) * (length(S) * (length(S) + 3))/2)
    }

}


libraries <- function(packages.names) {
    invisible(lapply(packages.names, library, character.only = TRUE))
}

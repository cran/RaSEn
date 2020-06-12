RaCutoff <- function(ytrain.pred, ytrain, p0) {
    n <- length(ytrain)
    vote0 <- rowMeans(ytrain.pred[ytrain == 0, ], na.rm = TRUE)
    vote1 <- rowMeans(ytrain.pred[ytrain == 1, ], na.rm = TRUE)
    errecdfm <- function(x) {
        (1 - p0) * ecdf(vote1)(x) + p0 * (1 - ecdf(vote0)(x))
    }
    errecdfM <- function(x) {
        (1 - p0) * ecdf(vote1)(-x) + p0 * (1 - ecdf(vote0)(-x))
    }
    alpham <- optimise(errecdfm, c(0, 1), maximum = F)$minimum
    alphaM <- optimise(errecdfM, c(-1, -0), maximum = F)$minimum
    alpha <- (alpham - alphaM)/2
    return(alpha)
}

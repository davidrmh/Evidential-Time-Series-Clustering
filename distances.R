ts_dist <- function(data, metric){
  # data is a list with tibbles.
  # All the tibbles have the same number
  # of observations
  n <- length(data)
  
  dmat <- matrix(0, nrow = n, ncol = n)
  for(i in 1:n){
    for(j in 1:n){
      if(i < j){
        dmat[i, j] <- metric(data[[i]], data[[j]])
        dmat[j, i] <- dmat[i, j]
      }
    }
  }
  dmat
}

decorate_metric <- function(metric, ...){
  wrapper <- function(x1, x2){
    metric(x1, x2, ...)
  }
  wrapper
}

easy_ts_dist <- function(x1, x2, w = NULL){
  # x1 and x2 are tibbles with time series data
  # both having the same length
  nobs <- nrow(x1)
  
  if(is.null(w)){
    w <- rep(1 / nobs, nobs)
  }
  d <- apply((x1[,] - x2[,])^2, MARGIN = 1, sum) * (w^2)
  d <- sum(d)
  sqrt(d)
}



#' @export
solve_path = function(X_list,
                      y_list,
                      censor_list,
                      lambda_1,
                      lambda_2,
                      p.factor=NULL,
                      step_size = 1,
                      B = NULL,
                      max_iter = 1500,
                      ls_beta = 1.1,
                      eps = 1e-12,
                      tol = 1e-15)
{
  stopifnot(length(lambda_1) == length(lambda_2))
  K = length(X_list)
  p = ncol(X_list[[1]])
  if(is.null(B)){
    B = matrix(rep(0.0,K*p), nrow = K, ncol = p)
  } else {
    stopifnot(nrow(B) == K, ncol(B) == p)
  }
  if(is.null(p.factor)){
    p.factor = rep(1.0, p)
  } else {
    stopifnot(length(p.factor) == p)
  }

  rankmin_list=list()
  rankmax_list=list()
  for (i in 1:K){
    n = length(y_list[[i]])
    if(nrow(X_list[[i]]) != n){
      stop(paste("The number of cases in the event time and the predictors do not match in the",
                 i, "th response"))
    }
    if(ncol(X_list[[i]]) != p){
      stop(paste("The number of variables in the", i, "th response does not match p"))
    }
    if(length(censor_list[[i]]) != n){
      stop(paste("The number of cases in the status vector and the predictors do not match in the",
                 i, "th response"))
    }

    o = order(y_list[[i]])
    y_list[[i]] = y_list[[i]][o]
    X_list[[i]] = X_list[[i]][o,]
    censor_list[[i]]= as.numeric(censor_list[[i]][o])
    rankmin_list[[i]] = as.integer(rank(y_list[[i]], ties.method="min") - 1)
    rankmax_list[[i]] = as.integer(rank(y_list[[i]], ties.method="max") - 1)
  }
  result = test(X_list,
                censor_list,
                B,
                rankmin_list,
                rankmax_list,
                step_size,
                lambda_1,
                lambda_2,
                p.factor,
                max_iter,
                ls_beta,
                eps,
                tol)
  return(result)
}


library("tibble")
library("lubridate")
library("readr")
library("dplyr")
library("stringr")

read_files <- function(path = "./data/stocks/",
                      start = "2019-01-01",
                      end = "2021-12-31",
                      new_first = TRUE){
  start <- ymd(start)
  end <- ymd(end)
  files <- list.files(path)
  data <- list()
  for(f in files){
    file_path <- paste0(path, f)
    symbol <- str_split(f, "_")[[1]][1]
    tib <- read_csv(file_path, show_col_types = FALSE)
    tib <- tib %>% filter(Date >= start & Date <= end)
    if(new_first)
      tib <- tib %>% arrange(desc(Date))
    data[[symbol]] <- tib
  }
  data
}

same_length <- function(data){
  #Data is a list created with read_files function
  n_obs <- c()
  for(s in names(data)){
    n_obs <- c(n_obs, nrow(data[[s]]))
  }
  if(length(unique(n_obs)) != 1 ){
    return(FALSE)
  }
  TRUE
}

norm_by_col <- function(data, target_cols = c("High", "Low", "Close"),
                        norm_col = "Open"){
  norm_data <- list()
  for(s in names(data)){
    norm_data[[s]] <- tibble(data[[s]][target_cols] / data[[s]][[norm_col]])
  }
  norm_data
}

transp_and_flat <- function(data){
  # data is a list with tibbles
  # all of them with the same number of rows and columns
  sym <- names(data)
  n_row <- nrow(data[[sym[1]]])
  n_col <- ncol(data[[sym[1]]])
  flat_data <- matrix(0, nrow = length(sym), ncol = n_row * n_col)
  for(i in seq_along(sym)){
    flat_data[i, ] <- unlist(data[[sym[i]]])
  }
  rownames(flat_data) <- sym
  flat_data
}

tibble_clusters <- function(y_clust, symb){
  # y_clust a vector with cluster assignment
  # for each observation
  tibble(Symbol = symb, cluster = y_clust)
}

compute_returns <- function(data, cols= c("Adj Close"),
                            keep_dates = FALSE, new_first = TRUE){
  # data is a list created with the function `read_files`
  
  ret <- list()
  for (s in names(data)){
    d <- data[[s]]
    nobs <- nrow(d)
    if(new_first){
      ret[[s]] <- as_tibble(d[1:nobs-1, cols] / d[2:nobs, cols] - 1)
      if(keep_dates){
        ret[[s]][["Date"]] <- d[["Date"]][1:nobs-1]
        ret[[s]] <- ret[[s]] %>% relocate(Date)
      }
      
    }
    else{
      ret[[s]] <- as_tibble(d[2:nobs, cols] / d[1:nobs-1, cols] - 1)
      if(keep_dates){
        ret[[s]][["Date"]] <- d[["Date"]][2:nobs]
        ret[[s]] <- ret[[s]] %>% relocate(Date)
      }
    }
  }
  ret
}

plot_cluster <- function(data, ...){
  # data is a tibble
  # with at least columns Symbol, x, y and cluster
  # ... are parameters passed to plot and text functions
  # These last parameters should be passed as key = value
  # pairs.
  params <- list(...)
  if(!("main" %in% names(params)))
    params[["main"]] <- "Cluster"
  if(!("pch" %in% names(params)))
    params[["pch"]] <- 16

  if(!("cex" %in% names(params)))
    params[["cex"]] <- 0.8

  if(!("pos" %in% names(params)))
    params[["pos"]] <- 1
  
  if(!("font" %in% names(params)))
    params[["font"]] <- 2
  
  plot(data$x, data$y, xlab = "X", ylab = "Y", main = params[["main"]],
       col = data$cluster, pch = params[["pch"]])
  
  text(data$x, data$y, labels = data$Symbol, 
       col =  data$cluster, cex = params[["cex"]],
       font = params[["font"]], pos = params[["pos"]])
  
  grid(col = "black")
}

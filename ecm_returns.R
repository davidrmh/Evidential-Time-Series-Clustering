library("evclust")
library("Rtsne")
source("utils.R")

# Inputs
path <- "./data/stocks/"
start <- "2022-01-01"
end <- "2022-01-31"
new_first <- TRUE
path_industry <- "./data/stock_industry.csv"
n_clust <- 3
perp <- 15

# Read csv with symbols names and industry
tib_indus <- read_csv(path_industry, show_col_types = FALSE)

# Read data (newest observation first)
data <- read_files(path = path, start = start, end = end, new_first = new_first)

# Validate same number of observations
if(!same_length(data))
  stop("Not every stock series has the same number of observations")

# Compute returns of Adj Close prices
ret_adj <- compute_returns(data, cols = c("Adj Close"), keep_dates = FALSE,
                           new_first = new_first)

# Compute returns of OHLC prices
ret_ohcl <- compute_returns(data,
                            cols = c("Open", "High", "Low", "Close"),
                            keep_dates = FALSE, new_first = new_first) 

# Transpose and flatten the data
# Each row represents a stock series
# Each column is an attribute
ret_adj <- transp_and_flat(ret_adj)
ret_ohcl <- transp_and_flat(ret_ohcl)

# TSNE representation of data Adj Close
tsne_adj <- Rtsne(ret_adj, dims = 2, perplexity = perp)$Y
colnames(tsne_adj) <- c("x", "y")
tsne_adj <- as_tibble(tsne_adj)
tsne_adj[["Symbol"]] <- names(data)
tsne_adj <- tsne_adj %>% relocate(Symbol)

# TSNE representation of data OHCL
tsne_ohcl <- Rtsne(ret_ohcl, dims = 2, perplexity = perp)$Y
colnames(tsne_ohcl) <- c("x", "y")
tsne_ohcl <- as_tibble(tsne_ohcl)
tsne_ohcl[["Symbol"]] <- names(data)
tsne_ohcl <- tsne_ohcl %>% relocate(Symbol)

# Classical K-Means using Adj Close prices
km_clust_adj <- kmeans(ret_adj, centers = n_clust, iter.max = 10)
km_adj <- tibble_clusters(km_clust_adj$cluster, names(data))
km_adj <- tib_indus %>% left_join(km_adj, by = "Symbol")
km_adj <- km_adj %>% left_join(tsne_adj, by = "Symbol")

# Classical K-Means using OHCL prices
km_clust_ohcl <- kmeans(ret_ohcl, centers = n_clust, iter.max = 10)
km_ohcl <- tibble_clusters(km_clust_ohcl$cluster, names(data))
km_ohcl <- tib_indus %>% left_join(km_ohcl, by = "Symbol")
km_ohcl <- km_ohcl %>% left_join(tsne_ohcl, by = "Symbol")

# Evidential c-means with Adj Close prices - Plausibility
ecm_clust_adj <- ecm(ret_adj, c = n_clust, type = "pairs", ntrials = 10, disp = FALSE)
ecm_pl_adj <- tibble_clusters(ecm_clust_adj$y.pl, names(data))
ecm_pl_adj <- tib_indus %>% left_join(ecm_pl_adj, by = "Symbol")
ecm_pl_adj <- ecm_pl_adj %>% left_join(tsne_adj, by = "Symbol")

# Evidential c-means with Adj Close prices - Belief
ecm_bel_adj <- tibble_clusters(ecm_clust_adj$y.bel, names(data))
ecm_bel_adj <- tib_indus %>% left_join(ecm_bel_adj, by = "Symbol")
ecm_bel_adj <- ecm_bel_adj %>% left_join(tsne_adj, by = "Symbol")

# Evidential c-means with OHCL prices - Plausibility
ecm_clust_ohcl <- ecm(ret_ohcl, c = n_clust, type = "pairs", ntrials = 10, disp = FALSE)
ecm_pl_ohcl <- tibble_clusters(ecm_clust_ohcl$y.pl, names(data))
ecm_pl_ohcl <- tib_indus %>% left_join(ecm_pl_ohcl, by = "Symbol")
ecm_pl_ohcl <- ecm_pl_ohcl %>% left_join(tsne_ohcl, by = "Symbol")

# Evidential c-means with OHCL prices - Belief
ecm_bel_ohcl <- tibble_clusters(ecm_clust_ohcl$y.bel, names(data))
ecm_bel_ohcl <- tib_indus %>% left_join(ecm_bel_ohcl, by = "Symbol")
ecm_bel_ohcl <- ecm_bel_ohcl %>% left_join(tsne_ohcl, by = "Symbol")

# Plot K-Means using Adj Close prices
plot(km_adj$x, km_adj$y, xlab = "X", ylab ="Y", main = "K-means-Adj Close",
     col = km_adj$cluster, pch = 19)

# Plot ECM using Adj Close Prices and Plausibility
plot(ecm_pl_adj$x, ecm_pl_adj$y, xlab = "X", ylab = "Y", 
     main = "ECM-Plausibility-Adj Close",
     col = ecm_pl_adj$cluster, pch = 19)

# Plot ECM using Adj Close Prices and Belief
plot(ecm_bel_adj$x, ecm_bel_adj$y, xlab = "X", ylab = "Y", 
     main = "ECM-Belief-Adj Close",
     col = ecm_bel_adj$cluster, pch = 19)

# Plot K-Means using OHCL prices
plot(km_ohcl$x, km_ohcl$y, xlab = "X", ylab ="Y", main = "K-means-OHCL",
     col = km_ohcl$cluster, pch = 19)

# Plot ECM using OHCL Prices and Plausibility
plot(ecm_pl_ohcl$x, ecm_pl_ohcl$y, xlab = "X", ylab = "Y", 
     main = "ECM-Plausibility-OHCL",
     col = ecm_pl_ohcl$cluster, pch = 19)

# Plot ECM using OHCL Prices and Belief
plot(ecm_bel_ohcl$x, ecm_bel_ohcl$y, xlab = "X", ylab = "Y", 
     main = "ECM-Belief-OHCL",
     col = ecm_bel_ohcl$cluster, pch = 19)

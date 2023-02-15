library("evclust")
library("Rtsne")
source("utils.R")

# Inputs
path <- "./data/stocks/"
start <- "2022-01-01"
end <- "2022-03-31"
new_first <- TRUE
norm_col <- "Open"
target_cols <- c("High", "Low", "Close")
path_industry <- "./data/stock_industry.csv"
n_clust <- 4
perp <- 15

# Read csv with symbols names and industry
tib_indus <- read_csv(path_industry, show_col_types = FALSE)

# Read data (newest observation first)
data <- read_files(path = path, start = start, end = end, new_first = new_first)

# Validate same number of observations
if(!same_length(data))
  stop("Not every stock series has the same number of observations")

# Normalize data by diving target_cols by norm_col
n_data <- norm_by_col(data, target_cols = target_cols, norm_col = norm_col)

# Transpose and flatten the data
# Each row represents a stock series
# Each column is an attribute
n_data <- transp_and_flat(n_data)

# TSNE representation of data
tsne <- Rtsne(n_data, dims = 2, perplexity = perp)$Y
colnames(tsne) <- c("x", "y")
tsne <- as_tibble(tsne)
tsne[["Symbol"]] <- names(data)
tsne <- tsne %>% relocate(Symbol)

# Classical K-Means with normalized data
km_clust <- kmeans(n_data, centers = n_clust, iter.max = 10)
km <- tibble_clusters(km_clust$cluster, names(data))
km <- tib_indus %>% left_join(km, by = "Symbol")
km <- km %>% left_join(tsne, by = "Symbol")

# Evidential c-means
# In this case, values for c less or equal than 4 seem to work
# Seems that c = 3 works quite well
ecm_clust <- ecm(n_data, c = n_clust, type = "pairs", ntrials = 10, disp = FALSE)

# Maximum plausibility
ecm_pl <- tibble_clusters(ecm_clust$y.pl, names(data))
ecm_pl <- tib_indus %>% left_join(ecm_pl, by = "Symbol")
ecm_pl <- ecm_pl %>% left_join(tsne, by = "Symbol")

# Maximum belief
ecm_bel <- tibble_clusters(ecm_clust$y.bel, names(data))
ecm_bel <- tib_indus %>% left_join(ecm_bel, by = "Symbol")
ecm_bel <- ecm_bel %>% left_join(tsne, by = "Symbol")

# Plot results for classical K-means
plot_cluster(km,
             main = "K-means-norm-open",
             pch = 16,
             cex = 0.8,
             font = 2,
             pos = 1)

# Plot results from ECM using Plausibility
plot_cluster(ecm_pl,
             main = "ECM-Plausibility-norm-open",
             pch = 16,
             cex = 0.8,
             font = 2,
             pos = 1)

# Plot results from ECM using Belief
plot_cluster(ecm_bel,
             main = "ECM-Belief-norm-open",
             pch = 16,
             cex = 0.8,
             font = 2,
             pos = 1)

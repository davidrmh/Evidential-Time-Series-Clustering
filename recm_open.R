library("evclust")
library("Rtsne")
source("utils.R")
source("distances.R")

# Inputs
path <- "./data/stocks/"
start <- "2022-01-01"
end <- "2022-03-31"
new_first <- TRUE
norm_col <- "Open"
target_cols <- c("High", "Low", "Close")
path_industry <- "./data/stock_industry.csv"
n_clust <- 3
perp <- 10

# Read csv with symbols names and industry
tib_indus <- read_csv(path_industry, show_col_types = FALSE)

# Read data (newest observation first)
data <- read_files(path = path, start = start, end = end, new_first = new_first)

# Validate same number of observations
if(!same_length(data))
  stop("Not every stock series has the same number of observations")

# Normalize data by diving target_cols by norm_col
n_data <- norm_by_col(data, target_cols = target_cols, norm_col = norm_col)

# TSNE representation of data
tsne <- Rtsne(transp_and_flat(n_data), dims = 2, perplexity = perp)$Y
colnames(tsne) <- c("x", "y")
tsne <- as_tibble(tsne)
tsne[["Symbol"]] <- names(data)
tsne <- tsne %>% relocate(Symbol)


# Relational Evidential c-means
w <- NULL
f <- decorate_metric(easy_ts_dist, w = w)
dmat <- ts_dist(n_data, f)
recm_clust <- recm(dmat, c = n_clust, type = "pairs", ntrials = 1, disp = FALSE)

# Maximum plausibility
recm_pl <- tibble_clusters(recm_clust$y.pl, names(data))
recm_pl <- tib_indus %>% left_join(recm_pl, by = "Symbol")
recm_pl <- recm_pl %>% left_join(tsne, by = "Symbol")

# Maximum belief
recm_bel <- tibble_clusters(recm_clust$y.bel, names(data))
recm_bel <- tib_indus %>% left_join(recm_bel, by = "Symbol")
recm_bel <- recm_bel %>% left_join(tsne, by = "Symbol")

# Plot results from RECM using Plausibility
plot_cluster(recm_pl,
             main = "RECM-Plausibility",
             pch = 16,
             cex = 0.8,
             font = 2,
             pos = 1)

# Plot results from ECM using Belief
plot_cluster(recm_bel,
             main = "RECM-Belief",
             pch = 16,
             cex = 0.8,
             font = 2,
             pos = 1)
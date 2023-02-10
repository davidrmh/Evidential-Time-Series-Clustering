library("evclust")
library("Rtsne")
source("utils.R")

set.seed(63523)
# Inputs
path <- "./data/stocks/"
start <- "2022-01-01"
end <- "2022-03-31"
norm_col <- "Open"
target_cols <- c("High", "Low", "Close")
path_industry <- "./data/stock_industry.csv"
n_clust <- 3
perp <- 15

# Read csv with symbols names and industry
tib_indus <- read_csv(path_industry, show_col_types = FALSE)

# Read data (newest observation first)
data <- read_files(path = path, start = start, end = end, new_first = TRUE)

# Validate same number of observations
print(same_length(data))

# Normalize data by diving target_cols by norm_col
n_data <- norm_by_col(data, target_cols = target_cols, norm_col = norm_col)

# Transpose and flatten the data
# Each row represents a stock series
# Each column is an attribute
flat_data <- transp_and_flat(n_data)
tsne_rep <- Rtsne(flat_data, dims = 2, perplexity = perp)

# Classical K-Means
km_clust <- kmeans(flat_data, centers = n_clust, iter.max = 10)
km_tsne <- Rtsne(t(flat_data), dims = 2, perplexity = perp)
km <- tibble_clusters(km_clust$cluster, names(data))
km <- tib_indus %>% left_join(km, by = "Symbol")


# Evidential c-means (attribute data)
# In this case, values for c less or equal than 4 seem to work
# Seems that c = 3 works quite well
ecm_clust <- ecm(flat_data, c = n_clust, type = "pairs", ntrials = 10, disp = FALSE)
ecm_pl <- tibble_clusters(ecm_clust$y.pl, names(data))
ecm_pl <- tib_indus %>% left_join(ecm_pl, by = "Symbol")
ecm_bel <- tibble_clusters(ecm_clust$y.bel, names(data))
ecm_bel <- tib_indus %>% left_join(ecm_bel, by = "Symbol")


plot(tsne_rep$Y, xlab = "X", ylab ="Y", main = "K-means",
     col = km_clust$cluster, pch = 19)
plot(tsne_rep$Y,xlab = "X", ylab = "Y", main = "ECM-Plausibility",
     col = ecm_clust$y.pl, pch = 19)
plot(tsne_rep$Y, xlab = "X", ylab = "Y", main = "ECM-Belif",
     col = ecm_clust$y.bel, pch = 19)



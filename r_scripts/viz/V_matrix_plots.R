library(tidyverse)
library(patchwork)
library(extrafont)
#font_import()
loadfonts(device="win") 

# Spatial UKB sample

# Read data
ukb_df <- read_csv("ukb/ukb_protein_cancer.csv")

location_df <- read_csv("ukb/ukb_cancer_location_df.csv")

# Kernel D
dist_mat <- as.matrix(dist(location_df[, 1:2], diag=T, upper=T)^2)
dim(dist_mat)
D <- 1 * exp(-dist_mat / (2 * 1))
dim(D)

# Sample some rows, take only what's needed from distance matrix
sample_n <- 1000
ukb_sample <- sample(1:nrow(ukb_df), sample_n, replace = FALSE)
ukb_s <- ukb_df[ukb_sample,]
not_in_train <- which(!sort(unique(ukb_df$location)) %in% ukb_s$location)
dist_tr <- dist_mat[-not_in_train, -not_in_train]
dim(dist_tr)

# PCA on distance matrix to get 1st PC
pca <- prcomp(dist_tr, rank. = 1)
recode_dict <- order(pca$x[,1])
names(recode_dict) <- sort(unique(ukb_s$location))

# location2 holds updated location factor
ukb_s <- ukb_s %>% mutate(location2 = recode(location, !!!recode_dict))
Z1 <- model.matrix(~0 + factor(sort(ukb_s$location2)))
dim(Z1)

# Adjust kernel
D_tr <- D[-not_in_train, -not_in_train][recode_dict, recode_dict]
dim(D_tr)

# Compute V
V <- diag(nrow(ukb_s)) + Z1 %*% D_tr %*% t(Z1)
dim(V)
print(V[1:30, 1:30], digits = 0)
print(round(V[1:20, 1:20], 2))
plot(V[,1])

# dimnames(V) <- NULL
# matplot <- function(A, ...) lattice::levelplot(A[, nrow(A):1], ...)
# matplot(round(V[1:1000, 1:1000]))
# matplot(V)

# Plot V
melted_V <- pivot_longer(as_tibble(V) %>% rowid_to_column(var = "loc1"),
                         cols = -loc1, names_to = "loc2",
                         names_transform = list(loc2 =  as.integer))

p_spatial <- melted_V %>%
  ggplot(aes(loc1, loc2, fill = value)) +
  geom_raster() +
  scale_fill_viridis_c() +
  scale_y_continuous(trans = "reverse", breaks = NULL) +
  scale_x_continuous(breaks = NULL) +
  labs(x = NULL, y = NULL, fill = "covariance") +
  coord_fixed(ratio = 1) +
  theme(
    text = element_text(family = "Century", size = 14),
    panel.border = element_blank(),
    panel.background = element_blank())

# Multiple categorical UKB sample
Z1 <- model.matrix(~0 + factor(ukb_s$diagnosis_id))
Z2 <- model.matrix(~0 + factor(ukb_s$operation_id))
Z3 <- model.matrix(~0 + factor(ukb_s$treatment_id))
Z4 <- model.matrix(~0 + factor(ukb_s$cancer_id))
Z5 <- model.matrix(~0 + factor(ukb_s$histology_id))

# Get V and perform PCA
V <- diag(nrow(ukb_s)) + 1 * Z1 %*% t(Z1) + 2 * Z2 %*% t(Z2) +
  3 * Z3 %*% t(Z3) + 4 * Z4 %*% t(Z4) + 5 * Z5 %*% t(Z5)
dim(V)
pca <- prcomp(V, rank. = 1)
ord <- order(pca$x[,1])

# Sort data by 1st PC of V
ukb_s_sorted <- ukb_s[ord,]# %>% arrange(diagnosis_id, operation_id, treatment_id, cancer_id, histology_id)

Z1 <- model.matrix(~0 + factor(ukb_s_sorted$diagnosis_id))
Z2 <- model.matrix(~0 + factor(ukb_s_sorted$operation_id))
Z3 <- model.matrix(~0 + factor(ukb_s_sorted$treatment_id))
Z4 <- model.matrix(~0 + factor(ukb_s_sorted$cancer_id))
Z5 <- model.matrix(~0 + factor(ukb_s_sorted$histology_id))
V <- diag(nrow(ukb_s)) + 1 * Z1 %*% t(Z1) + 2 * Z2 %*% t(Z2) +
  3 * Z3 %*% t(Z3) + 4 * Z4 %*% t(Z4) + 5 * Z5 %*% t(Z5)
dim(V)

# Plot V
melted_V <- pivot_longer(as_tibble(V) %>% rowid_to_column(var = "loc1"),
                         cols = -loc1, names_to = "loc2",
                         names_transform = list(loc2 =  as.integer))

p_categoricals <- melted_V %>%
  ggplot(aes(loc1, loc2, fill = value)) +
  geom_raster() +
  # scale_fill_gradient(low = "white", high = "black") +
  scale_fill_viridis_c() +
  scale_y_continuous(trans = "reverse", breaks = NULL) +
  scale_x_continuous(breaks = NULL) +
  labs(x = NULL, y = NULL, fill = "covariance") +
  coord_fixed(ratio = 1) +
  theme(
    text = element_text(family = "Century", size = 14),
    panel.border = element_blank(),
    panel.background = element_blank())

p_spatial | p_categoricals
ggsave(filename = "images/lmmnn_UKB_V_matrices.png", width = 8, height = 3, device="png", dpi=700)

# plot eigendecay
eigens_Z1 <- eigen(Z1 %*% t(Z1))$values
eigens_V <- eigen(V)$values
plot(eigens_Z1)
plot(eigens_V)

p_EigZZ <- tibble(idx = 1:length(eigens_Z1), eig = eigens_Z1) %>%
  ggplot(aes(idx, eig)) +
  geom_point(size = 2) +
  geom_line(data = tibble(idx = 1:length(eigens_Z1), eig = 1000/(idx ^ 1)), color = "red", lwd = 1) +
  ylim(range(eigens_Z1)) +
  labs(y = "Eig(ZZ')", x = NULL) +
  theme_bw() +
  theme(
    text = element_text(family = "Century", size = 14)
  )

p_EigV <- tibble(idx = 1:length(eigens_V), eig = eigens_V) %>%
  ggplot(aes(idx, eig)) +
  geom_point(size = 2) +
  geom_line(data = tibble(idx = 1:length(eigens_V), eig = 5000/(idx ^ 1)), color = "red", lwd = 1) +
  ylim(range(eigens_V)) +
  labs(y = "Eig(V)", x = NULL) +
  theme_bw() +
  theme(
    text = element_text(family = "Century", size = 14)
  )
p_EigZZ | p_EigV
ggsave(filename = "images/lmmnn_UKB_eigendecay.png", width = 8, height = 3, device="png", dpi=700)

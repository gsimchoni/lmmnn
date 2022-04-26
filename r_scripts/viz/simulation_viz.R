library(tidyverse)
library(patchwork)
library(extrafont)

font_import()
loadfonts(device="win") 


# Single categorical simulation viz ---------------------------------------

RE1 <- read_csv("data/data_for_viz/RE_sig2b1.csv")
Y1 <- read_csv("data/data_for_viz/y_test_sig2b1.csv")
RE10 <- read_csv("data/data_for_viz/RE_sig2b10.csv")
Y10 <- read_csv("data/data_for_viz/y_test_sig2b10.csv")

p_scatter_b_sig2b1 <- RE1 %>%
  ggplot(aes(b_true, b_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "", y = "Predicted RE") +
  xlim(-10, 10) +
  ylim(-10, 10) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_scatter_b_sig2b10 <- RE10 %>%
  ggplot(aes(b_true, b_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True RE", y = "Predicted RE") +
  xlim(-10, 10) +
  ylim(-10, 10) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_hist_n_sig2b1 <- RE1 %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "", y = "Frequency", title = expression(paste(sigma[b]^2," =1, ", q, "=1000"))) +
  xlim(0, 300) +
  ylim(0, 160) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20))

p_hist_n_sig2b10 <- RE10 %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "Category size", y = "Frequency", title = expression(paste(sigma[b]^2," =10, ", q, "=1000"))) +
  xlim(0, 300) +
  ylim(0, 160) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20))

p_scatter_y_sig2b1 <- Y1 %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "", y = "Predicted test y") +
  xlim(-15, 15) +
  ylim(-15, 15) +
  # geom_abline(intercept = 0, slope = 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_scatter_y_sig2b10 <- Y10 %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "Predicted test y") +
  xlim(-15, 15) +
  ylim(-15, 15) +
  # geom_abline(intercept = 0, slope = 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p1 <- (p_scatter_b_sig2b1 | p_hist_n_sig2b1 | p_scatter_y_sig2b1)
p2 <- (p_scatter_b_sig2b10 | p_hist_n_sig2b10 | p_scatter_y_sig2b10)
p <- p1 / p2

ggsave("images/sim_viz.png", p, device = "png", width = 16, height = 8, dpi = 300)

# Spatial data simulation viz ---------------------------------------------

RE1 <- read_csv("data/data_for_viz/RE_spatial_sig2b1.csv")
Y1 <- read_csv("data/data_for_viz/y_test_spatial_sig1.csv")
RE10 <- read_csv("data/data_for_viz/RE_spatial_sig2b10.csv")
Y10 <- read_csv("data/data_for_viz/y_test_spatial_sig10.csv")

min_b <- floor(min(min(RE1$b_true), min(RE1$b_pred)))
max_b <- ceiling(max(max(RE1$b_true), max(RE1$b_pred)))

p_heatmap_sig2b1_true <- RE1 %>%
  mutate(lat = as.numeric(cut_interval(lat, length = 1.0)),
         lon = as.numeric(cut_interval(lon, length = 1.0))) %>%
  group_by(lat, lon) %>%
  summarise(b_true = mean(b_true)) %>%
  ggplot(aes(lon, lat)) +
  geom_tile(aes(fill = b_true)) +
  scale_fill_binned(breaks = seq(min_b, max_b, 1)) +
  theme_bw() +
  labs(x = "", y = "latitude", subtitle = "True RE") +
  guides(fill = "none") +
  scale_x_continuous(labels = seq(-10, 10, 5)) +
  scale_y_continuous(labels = seq(-10, 10, 5)) +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.subtitle = element_text(hjust = 0))

p_heatmap_sig2b1_pred <- RE1 %>%
  mutate(lat = as.numeric(cut_interval(lat, length = 1.0)),
         lon = as.numeric(cut_interval(lon, length = 1.0))) %>%
  group_by(lat, lon) %>%
  summarise(b_pred = mean(b_pred)) %>%
  ggplot(aes(lon, lat)) +
  geom_tile(aes(fill = b_pred)) +
  scale_fill_binned(breaks = seq(min_b, max_b, 1)) +
  theme_bw() +
  labs(x = "", y = "", subtitle = "Predicted RE",
       title = expression(paste(sigma[b0]^2," =1, ", q, "=10000"))) +
  guides(fill = guide_colorbar(title="b")) +
  scale_x_continuous(labels = seq(-10, 10, 5)) +
  scale_y_continuous(labels = seq(-10, 10, 5)) +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.subtitle = element_text(hjust = 0),
        plot.title = element_text(hjust = 0.5, size = 20))

p_scatter_y_sig2b1 <- Y1 %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "", y = "Predicted test y") +
  xlim(-15, 15) +
  ylim(-15, 15) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

min_b <- floor(min(min(RE10$b_true), min(RE10$b_pred)))
max_b <- ceiling(max(max(RE10$b_true), max(RE10$b_pred)))

p_heatmap_sig2b10_true <- RE10 %>%
  mutate(lat = as.numeric(cut_interval(lat, length = 1.0)),
         lon = as.numeric(cut_interval(lon, length = 1.0))) %>%
  group_by(lat, lon) %>%
  summarise(b_true = mean(b_true)) %>%
  ggplot(aes(lon, lat)) +
  geom_tile(aes(fill = b_true)) +
  scale_fill_binned(breaks = seq(min_b, max_b, 2)) +
  theme_bw() +
  guides(fill = "none") +
  labs(x = "longitude", y = "latitude", subtitle = "True RE") +
  scale_x_continuous(labels = seq(-10, 10, 5)) +
  scale_y_continuous(labels = seq(-10, 10, 5)) +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p_heatmap_sig2b10_pred <- RE10 %>%
  mutate(lat = as.numeric(cut_interval(lat, length = 1.0)),
         lon = as.numeric(cut_interval(lon, length = 1.0))) %>%
  group_by(lat, lon) %>%
  summarise(b_pred = mean(b_pred)) %>%
  ggplot(aes(lon, lat)) +
  geom_tile(aes(fill = b_pred)) +
  scale_fill_binned(breaks = seq(min_b, max_b, 2)) +
  theme_bw() +
  guides(fill = guide_colorbar(title="b")) +
  labs(x = "longitude", y = "", subtitle = "Predicted RE",
       title = expression(paste(sigma[b0]^2," =10, ", q, "=10000"))) +
  scale_x_continuous(labels = seq(-10, 10, 5)) +
  scale_y_continuous(labels = seq(-10, 10, 5)) +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20))

p_scatter_y_sig2b10 <- Y10 %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "Predicted test y") +
  xlim(-15, 15) +
  ylim(-15, 15) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

p1 <- (p_heatmap_sig2b1_true | p_heatmap_sig2b1_pred | p_scatter_y_sig2b1)
p2 <- (p_heatmap_sig2b10_true | p_heatmap_sig2b10_pred | p_scatter_y_sig2b10)
p <- p1 / p2

ggsave("images/sim_spatial_viz.png", p, device = "png", width = 12, height = 8, dpi = 300)

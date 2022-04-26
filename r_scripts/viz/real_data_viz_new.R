library(tidyverse)
library(patchwork)
library(extrafont)

font_import()
loadfonts(device="win") 


# Multiple categorical datasets -------------------------------------------

imdb_cat <- read_csv("data/data_for_viz/imdb_ns.csv")
imdb_y <- read_csv("data/data_for_viz/imdb_y_test.csv")
news_cat <- read_csv("data/data_for_viz/news_ns.csv")
news_y <- read_csv("data/data_for_viz/news_y_test.csv")
spotify_cat <- read_csv("data/data_for_viz/spotify_ns.csv")
spotify_y <- read_csv("data/data_for_viz/spotify_y_test.csv")
ukbblood_cat <- read_csv("data/data_for_viz/ukbblood_ns.csv")
ukbblood_y <- read_csv("data/data_for_viz/ukbblood_y_test.csv")

imdb_scatter_y <- imdb_y %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "Predicted test y") +
  xlim(1, 10) +
  ylim(1, 10) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

imdb_hist_b <- imdb_cat %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "Category size", y = "Frequency", title = "Imdb", subtitle = "Categorical: director,  Y: score") +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 14))

news_scatter_y <- news_y %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "") +
  xlim(0, 10) +
  ylim(0, 10) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

news_hist_b <- news_cat %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "Category size", y = "", title = "News", subtitle = "Categorical: source,  Y: log(shares)") +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 14))

spotify_scatter_y <- spotify_y %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "") +
  xlim(0, 1) +
  ylim(0, 1) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

spotify_hist_b <- spotify_cat %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "Category size", y = "", title = "Spotify", subtitle = "Categorical: album,  Y: danceability") +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 14))

ukbblood_scatter_y <- ukbblood_y %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "") +
  xlim(-2, 6) +
  ylim(-2, 6) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

ukbblood_hist_b <- ukbblood_cat %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "Category size", y = "", title = "UKB-blood", subtitle = "Categorical: cancer type,  Y: triglyc.") +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 14))

p2 <- (imdb_hist_b / imdb_scatter_y) | (news_hist_b / news_scatter_y) | (spotify_hist_b / spotify_scatter_y) | (ukbblood_hist_b / ukbblood_scatter_y)

ggsave("images/real_viz_new_categorical.png", p2, device = "png", width = 16, height = 8, dpi = 300)

# Longitudinal datasets -------------------------------------------

rossmann_cat <- read_csv("data/data_for_viz/rossmann_ns.csv")
rossmann_y <- read_csv("data/data_for_viz/rossmann_y_test.csv")
au_cat <- read_csv("data/data_for_viz/au_ns.csv")
au_y <- read_csv("data/data_for_viz/au_y_test.csv")
ukbsbp_cat <- read_csv("data/data_for_viz/ukbsbp_ns.csv")
ukbsbp_y <- read_csv("data/data_for_viz/ukbsbp_y_test.csv")

rossmann_scatter_y <- rossmann_y %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "Predicted test y") +
  xlim(0, 8) +
  ylim(0, 8) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

rossmann_hist_b <- rossmann_cat %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "No. repeated measures", y = "Frequency", title = "Rossmann", subtitle = "Y: sales $") +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 14))

au_scatter_y <- au_y %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "") +
  xlim(0, 25) +
  ylim(0, 25) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

au_hist_b <- au_cat %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "No. repeated measures", y = "", title = "AUimport", subtitle = "Y: log(import $)") +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 14))

ukbsbp_scatter_y <- ukbsbp_y %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "") +
  xlim(0.5, 2.5) +
  ylim(0.5, 2.5) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

ukbsbp_hist_b <- ukbsbp_cat %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  scale_y_continuous(breaks = c(1:4)*100000, labels = c("100K", "200K", "300K", "400K")) +
  labs(x = "No. repeated measures", y = "", title = "UKB-SBP", subtitle = "Y: SBP") +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 14))

p2 <- (rossmann_hist_b / rossmann_scatter_y) | (au_hist_b / au_scatter_y) | (ukbsbp_hist_b / ukbsbp_scatter_y)

ggsave("images/real_viz_new_longitudinal.png", p2, device = "png", width = 12, height = 8, dpi = 300)

# Spatial datasets -------------------------------------------

income_cat <- read_csv("data/data_for_viz/income_ns.csv")
income_y <- read_csv("data/data_for_viz/income_y_test.csv")
asthma_cat <- read_csv("data/data_for_viz/asthma_ns.csv")
asthma_y <- read_csv("data/data_for_viz/asthma_y_test.csv")
airbnb_cat <- read_csv("data/data_for_viz/airbnb_ns_new.csv")
airbnb_y <- read_csv("data/data_for_viz/airbnb_y_test_new.csv")
cars_cat <- read_csv("data/data_for_viz/cars_ns.csv")
cars_y <- read_csv("data/data_for_viz/cars_y_test.csv")

income_scatter_y <- income_y %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "Predicted test y") +
  xlim(8.5, 12.5) +
  ylim(8.5, 12.5) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

income_hist_b <- income_cat %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "No. of measures in location", y = "Frequency", title = "Income", subtitle = "Y: log(income $)") +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 14))

asthma_scatter_y <- asthma_y %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "") +
  xlim(5.5, 20) +
  ylim(5.5, 20) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

asthma_hist_b <- asthma_cat %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "No. of measures in location", y = "Frequency", title = "Asthma", subtitle = "Y: asthma%") +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 14))

airbnb_scatter_y <- airbnb_y %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "") +
  xlim(2.5, 8) +
  ylim(2.5, 8) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

airbnb_hist_b <- airbnb_cat %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "No. of measures in location", y = "Frequency", title = "Airbnb", subtitle = "Y: log(price)") +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 14))

cars_scatter_y <- cars_y %>%
  ggplot(aes(y_true, y_pred)) +
  geom_point(alpha = 0.2) +
  labs(x = "True test y", y = "") +
  xlim(6.9, 12.5) +
  ylim(6.9, 12.5) +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16))

cars_hist_b <- cars_cat %>%
  ggplot(aes(n)) +
  geom_histogram(alpha = 0.5) +
  labs(x = "No. of measures in location", y = "Frequency", title = "Cars", subtitle = "Y: log(price)") +
  theme_bw() +
  theme(aspect.ratio=1, text = element_text(family = "Century", size=16),
        plot.title = element_text(hjust = 0.5, size = 20),
        plot.subtitle = element_text(hjust = 0.5, size = 14))

p2 <- (income_hist_b / income_scatter_y) | (asthma_hist_b / asthma_scatter_y) | (airbnb_hist_b / airbnb_scatter_y) | (cars_hist_b / cars_scatter_y)

ggsave("images/real_viz_new_spatial.png", p2, device = "png", width = 16, height = 8, dpi = 300)

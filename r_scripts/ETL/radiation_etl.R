library(tidyverse)
library(lubridate)
library(janitor)
library(scales)

# The Safecast Radiation Measurements from Kaggle: https://www.kaggle.com/datasets/safecast/safecast

# read radiation data in Japan (6.5M obs out of 82M)
# df <- read_csv("radiation_measurements/measurements.csv")
# 
# x_min_jpn = 128.03
# y_min_jpn = 30.22
# x_max_jpn = 148.65
# y_max_jpn = 45.83
# 
# df_japan_2017 <- df %>%
#   filter(between(Longitude, x_min_jpn, x_max_jpn),
#          between(Latitude, y_min_jpn, y_max_jpn)) %>%
#   mutate(y = year(`Captured Time`)) %>%
#   filter(y == 2017, Value > 0, Unit == "cpm") %>%
#   select(time = `Captured Time`, lat = Latitude, lon = Longitude, val = Value)
# df_japan_2017 %>% write_csv("radiation_measurements/measurements_japan.csv")

df <- read_csv("radiation_measurements/measurements_japan.csv")

# time features
df <- df %>%
  mutate(month = month(time),
         wday = wday(time),
         hour = hour(time))

# location
df <- df %>%
  mutate(lat = round(lat, 1), lon = round(lon, 1))

location_df <- df %>%
  select(lat, lon) %>%
  arrange(lat, lon) %>%
  distinct() %>%
  mutate(location_id = 0 : (n() - 1))

df <- df %>% inner_join(location_df)

df$lat <- rescale(df$lat, to = c(-10, 10))
df$lon <- rescale(df$lon, to = c(-10, 10))

df %>% select(-time) %>% write_csv("radiation_df.csv")

# 10% df
df_small <- df %>% group_by(location_id) %>% slice_sample(prop = 0.1) %>% ungroup

location_df <- df_small %>%
  select(lat, lon) %>%
  arrange(lat, lon) %>%
  distinct() %>%
  mutate(location_id = 0 : (n() - 1))

df_small <- df_small %>% select(-location_id) %>% inner_join(location_df)
df_small %>% select(-time) %>% write_csv("radiation_df_small.csv")

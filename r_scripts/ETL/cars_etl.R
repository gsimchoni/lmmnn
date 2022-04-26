library(tidyverse)
library(lubridate)
library(janitor)
library(scales)

# Used Cars from Craigslist dataset from Kaggle: https://www.kaggle.com/datasets/austinreese/craigslist-carstrucks-data

# read used cars
df <- read_csv("used_cars/vehicles.csv")

features <- c("price", "manufacturer", "model", "year", "condition", "fuel", "odometer",
              "title_status", "transmission", "drive", "size", "type",
              "paint_color", "lat", "long", "VIN")

df <- df %>%
  select(all_of(features)) %>%
  distinct(VIN, .keep_all = T) %>%
  filter(between(price, 1000, 300000),
         between(lat, 20, 50),
         between(long, -150, -50))%>%
  drop_na(price, year, lat, long)

# recode manufacturer
top10_manufacturers <- df %>% count(manufacturer, sort = T) %>% slice(1:10) %>% pull(manufacturer)
df <- df %>%
  mutate(manufacturer = ifelse(manufacturer %in% top10_manufacturers, manufacturer, "other"))
df_manufacturer <- as_tibble(model.matrix(~ 0 + manufacturer, data = df))
colnames(df_manufacturer) <- make_clean_names(colnames(df_manufacturer))

# recode model
model_df <- df %>%
  count(model) %>%
  mutate(model_id = 0 : (n() - 1)) %>%
  select(model, model_id)

df <- df %>% inner_join(model_df)

# recode location
df <- df %>% mutate(lat = round(lat, 3), long = round(long, 3))
location_df <- df %>%
  select(lat, long) %>%
  arrange(lat, long) %>%
  distinct() %>%
  mutate(location_id = 0 : (n() - 1))

df <- df %>% inner_join(location_df)
df$lat <- rescale(df$lat, to = c(-10, 10))
df$long <- rescale(df$long, to = c(-10, 10))

# recode condition
df$condition[is.na(df$condition)] <- "NA"
df_condition <- as_tibble(model.matrix(~ 0 + condition, data = df))
colnames(df_condition) <- make_clean_names(colnames(df_condition))

# recode fuel
df$fuel[is.na(df$fuel)] <- "NA"
df_fuel <- as_tibble(model.matrix(~ 0 + fuel, data = df))
colnames(df_fuel) <- make_clean_names(colnames(df_fuel))

# recode title_status
df$title_status[is.na(df$title_status)] <- "NA"
df_title_status <- as_tibble(model.matrix(~ 0 + title_status, data = df))
colnames(df_title_status) <- make_clean_names(colnames(df_title_status))

# recode transmission
df$transmission[is.na(df$transmission)] <- "NA"
df_transmission <- as_tibble(model.matrix(~ 0 + transmission, data = df))
colnames(df_transmission) <- make_clean_names(colnames(df_transmission))

# recode drive
df$drive[is.na(df$drive)] <- "NA"
df_drive <- as_tibble(model.matrix(~ 0 + drive, data = df))
colnames(df_drive) <- make_clean_names(colnames(df_drive))

# recode size
df$size[is.na(df$size)] <- "NA"
df_size <- as_tibble(model.matrix(~ 0 + size, data = df))
colnames(df_size) <- make_clean_names(colnames(df_size))

# recode type
df$type[is.na(df$type)] <- "NA"
df_type <- as_tibble(model.matrix(~ 0 + type, data = df))
colnames(df_type) <- make_clean_names(colnames(df_type))

# recode paint_color
df$paint_color[is.na(df$paint_color)] <- "NA"
df_paint_color <- as_tibble(model.matrix(~ 0 + paint_color, data = df))
colnames(df_paint_color) <- make_clean_names(colnames(df_paint_color))

# odometer
df$odometer[is.na(df$odometer)] <- median(df$odometer, na.rm = T)
df$odometer <- df$odometer/100000

# year
df$year <- as.vector(scale(df$year - 2000))

# final
df_final <- df %>%
  bind_cols(df_manufacturer, df_condition, df_fuel, df_title_status, df_transmission,
            df_drive, df_size, df_type, df_paint_color) %>%
  select(-condition, -manufacturer, -fuel, -title_status, -transmission,
         -drive, -size, -type, -paint_color, -model, -VIN)

df_final %>% write_csv("cars_df5.csv")

# df_final %>% group_by(long, lat) %>%
#   summarize(price = mean(log(price))) %>%
#   ggplot(aes(long, lat)) +
#   geom_point(aes(size = price)) +
#   theme_bw()
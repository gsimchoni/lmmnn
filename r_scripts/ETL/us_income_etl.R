library(tidyverse)
library(lubridate)
library(janitor)
library(scales)

# US Census Demographic dataset from Kaggle: https://www.kaggle.com/datasets/muonneutrino/us-census-demographic-data
# read in US census tract data
df <- read_csv("us_cencus/acs2017_census_tract_data.csv")
df_county <- read_csv("us_cencus/acs2017_county_data.csv")
df_county_lat_lon <- read_csv("us_cencus/us_county.csv")

df_county <- df_county %>% inner_join(df_county_lat_lon, by = c("CountyId" = "fips")) %>%
  select(CountyId, State, County, lat, long)
df <- df %>% inner_join(df_county)
colnames(df) <- make_clean_names(colnames(df))

# recode pop, income
df <- df %>%
  mutate(men = men/total_pop,
         women = women / total_pop,
         employed = employed / total_pop,
         voting_age_citizen = voting_age_citizen / total_pop,
         total_pop = log(total_pop)) %>%
  select(-income_err, -income_per_cap, -income_per_cap_err, -tract_id, -state, -county, -county_id)

# recode percentages
percent_cols <- c("Hispanic", "White", "Black", "Native", "Asian", "Pacific",
  "Poverty", "ChildPoverty", "Professional", "Service", "Office",
  "Construction", "Production", "Drive", "Carpool", "Transit", "Walk",
  "OtherTransp", "WorkAtHome", "MeanCommute", "PrivateWork", "PublicWork",
  "SelfEmployed", "FamilyWork", "Unemployment")

df <- df %>% mutate(across(all_of(make_clean_names(percent_cols)), ~.x/100))

# recode location
location_df <- df %>%
  select(lat, long) %>%
  arrange(lat, long) %>%
  distinct() %>%
  filter(
    between(lat, 20, 50),
    between(long, -150, -50)
  ) %>%
  mutate(location_id = 0 : (n() - 1))

df <- df %>% inner_join(location_df)

df$lat <- rescale(df$lat, to = c(-10, 10))
df$long <- rescale(df$long, to = c(-10, 10))

# no NAs
df <- df %>% drop_na()

df %>% write_csv("uscensus_df.csv")

df %>% group_by(long, lat) %>%
  summarize(income = mean(log(income))) %>%
  ggplot(aes(long, lat)) +
  geom_point(aes(size = income)) +
  theme_bw()
